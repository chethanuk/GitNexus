import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { EmbeddingConfig } from '../../src/core/embeddings/types.js';
import type { EmbeddingSidecarDevice } from '../../src/core/embeddings/embedding-sidecar-protocol.js';

/**
 * Device selection in the sidecar's local embedder. DirectML can't run on a
 * Linux CI host, so transformers.js is replaced by a fake whose vectors depend
 * on the device it was loaded for. The real initLocalEmbedder / getCurrentDevice
 * / localEmbedBatch seam is exercised end to end.
 */
const CPU_VECTOR = [1, 0, 0, 0];
const BAD_DML_VECTOR = [0, 1, 0, 0];

const { state, pipelineSpy, warnSpy } = vi.hoisted(() => ({
  state: {
    dmlVector: [0, 1, 0, 0] as number[],
    dmlProbeThrows: false,
    cpuLoadFailures: 0,
    cpuProbeFailures: 0,
    disposed: [] as string[],
  },
  pipelineSpy: vi.fn(),
  warnSpy: vi.fn(),
}));

vi.mock('@huggingface/transformers', () => ({
  pipeline: vi.fn(async (_task: string, _model: string, opts: { device: string }) => {
    pipelineSpy(opts.device);
    if (opts.device === 'cpu' && state.cpuLoadFailures > 0) {
      state.cpuLoadFailures--;
      throw new Error('cpu session failed to load');
    }
    const vector = opts.device === 'dml' ? state.dmlVector : CPU_VECTOR;
    const embed = async (texts: string[]) => {
      if (opts.device === 'dml' && state.dmlProbeThrows) throw new Error('dml inference failed');
      if (opts.device === 'cpu' && state.cpuProbeFailures > 0) {
        state.cpuProbeFailures--;
        throw new Error('cpu inference failed');
      }
      return {
        data: new Float32Array(texts.flatMap(() => vector)),
        dims: [texts.length, vector.length],
      };
    };
    return Object.assign(embed, {
      dispose: vi.fn(async () => {
        state.disposed.push(opts.device);
      }),
    });
  }),
  env: { allowLocalModels: true, cacheDir: '', remoteHost: '' },
}));

vi.mock('../../src/core/embeddings/onnxruntime-node-resolver.js', () => ({
  ensureOnnxRuntimeNodeMatchesSystem: () => {},
  isEffectiveCudaAvailable: () => false,
}));
vi.mock('../../src/core/embeddings/runtime-install.js', () => ({
  ensureEmbeddingStackResolvable: () => {},
}));
vi.mock('../../src/core/embeddings/onnxruntime-common-resolver.js', () => ({
  ensureOnnxRuntimeCommonResolvable: () => {},
}));
vi.mock('../../src/core/logger.js', () => ({
  logger: { info: vi.fn(), warn: warnSpy, error: vi.fn(), debug: vi.fn() },
}));

const ENV_KEYS = ['GITNEXUS_EMBEDDING_DEVICE'] as const;
const savedEnv = Object.fromEntries(ENV_KEYS.map((k) => [k, process.env[k]]));
const origPlatform = process.platform;
const origArch = process.arch;

beforeEach(() => {
  vi.resetModules();
  pipelineSpy.mockClear();
  warnSpy.mockClear();
  state.dmlProbeThrows = false;
  state.cpuLoadFailures = 0;
  state.cpuProbeFailures = 0;
  state.disposed = [];
  for (const key of ENV_KEYS) delete process.env[key];
  Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
  Object.defineProperty(process, 'arch', { value: 'x64', configurable: true });
});

afterEach(() => {
  Object.defineProperty(process, 'platform', { value: origPlatform, configurable: true });
  Object.defineProperty(process, 'arch', { value: origArch, configurable: true });
  for (const key of ENV_KEYS) {
    if (savedEnv[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnv[key];
  }
});

interface DeviceCase {
  name: string;
  config: Partial<EmbeddingConfig>;
  forceDevice?: EmbeddingSidecarDevice;
  dmlVector: number[];
  dmlProbeThrows?: boolean;
  cpuLoadFailures?: number;
  cpuProbeFailures?: number;
  expectedDevice: EmbeddingSidecarDevice;
  /** Exact warning logged, or none. */
  warning?: string;
  disposed: string[];
}

const cases: DeviceCase[] = [
  {
    name: 'auto picks CPU instead of DirectML',
    config: { device: 'auto' },
    dmlVector: BAD_DML_VECTOR,
    expectedDevice: 'cpu',
    disposed: [],
  },
  {
    name: 'explicit dml falls back to CPU with a warning when the probe disagrees',
    config: { device: 'dml' },
    dmlVector: BAD_DML_VECTOR,
    expectedDevice: 'cpu',
    warning: 'DirectML embeddings do not match CPU (probe cosine 0.000); using CPU instead.',
    disposed: ['dml'],
  },
  {
    name: 'forced dml falls back to CPU with a warning when the probe disagrees',
    config: {},
    forceDevice: 'dml',
    dmlVector: BAD_DML_VECTOR,
    expectedDevice: 'cpu',
    warning: 'DirectML embeddings do not match CPU (probe cosine 0.000); using CPU instead.',
    disposed: ['dml'],
  },
  {
    name: 'explicit dml falls back to CPU when the probe is NaN',
    config: { device: 'dml' },
    dmlVector: [Number.NaN, 0, 0, 0],
    expectedDevice: 'cpu',
    warning: 'DirectML embeddings do not match CPU (probe cosine NaN); using CPU instead.',
    disposed: ['dml'],
  },
  {
    name: 'explicit dml falls back to CPU when an unnormalized probe points elsewhere',
    config: { device: 'dml' },
    dmlVector: [5, 5, 0, 0],
    expectedDevice: 'cpu',
    warning: 'DirectML embeddings do not match CPU (probe cosine 0.707); using CPU instead.',
    disposed: ['dml'],
  },
  {
    name: 'explicit dml falls back to CPU just below the 0.99 cosine floor',
    config: { device: 'dml' },
    dmlVector: [1, 0.1875, 0, 0],
    expectedDevice: 'cpu',
    warning: 'DirectML embeddings do not match CPU (probe cosine 0.983); using CPU instead.',
    disposed: ['dml'],
  },
  {
    name: 'explicit dml is kept just above the 0.99 cosine floor',
    config: { device: 'dml' },
    dmlVector: [1, 0.125, 0, 0],
    expectedDevice: 'dml',
    disposed: ['cpu'],
  },
  {
    name: 'explicit dml falls back to CPU when the DirectML probe throws',
    config: { device: 'dml' },
    dmlVector: CPU_VECTOR,
    dmlProbeThrows: true,
    expectedDevice: 'cpu',
    warning: 'DirectML probe failed (dml inference failed); using CPU instead.',
    disposed: ['dml'],
  },
  {
    name: 'explicit dml releases DirectML and retries CPU when the CPU check fails to load',
    config: { device: 'dml' },
    dmlVector: CPU_VECTOR,
    cpuLoadFailures: 1,
    expectedDevice: 'cpu',
    disposed: ['dml'],
  },
  {
    name: 'explicit dml releases both pipelines and retries CPU when the CPU probe throws',
    config: { device: 'dml' },
    dmlVector: CPU_VECTOR,
    cpuProbeFailures: 1,
    expectedDevice: 'cpu',
    disposed: ['dml', 'cpu'],
  },
  {
    name: 'explicit dml is kept when the probe matches CPU',
    config: { device: 'dml' },
    dmlVector: CPU_VECTOR,
    expectedDevice: 'dml',
    disposed: ['cpu'],
  },
  {
    name: 'explicit cpu stays on CPU without loading DirectML',
    config: { device: 'cpu' },
    dmlVector: BAD_DML_VECTOR,
    expectedDevice: 'cpu',
    disposed: [],
  },
];

describe('initLocalEmbedder device selection on Windows without CUDA', () => {
  // it.each('$name') truncates titles to 40 chars, which makes several cases collide.
  it.each(cases.map((c) => [c.name, c] as const))('%s', async (_name, c) => {
    state.dmlVector = c.dmlVector;
    state.dmlProbeThrows = c.dmlProbeThrows ?? false;
    state.cpuLoadFailures = c.cpuLoadFailures ?? 0;
    state.cpuProbeFailures = c.cpuProbeFailures ?? 0;
    const { initLocalEmbedder, getCurrentDevice, localEmbedBatch } =
      await import('../../src/core/embeddings/embedding-local-init.js');

    await initLocalEmbedder(undefined, c.config, c.forceDevice);

    expect(getCurrentDevice()).toBe(c.expectedDevice);
    const [vector] = await localEmbedBatch(['function add(a, b) { return a + b; }']);
    expect(Array.from(vector)).toEqual(c.expectedDevice === 'dml' ? c.dmlVector : CPU_VECTOR);
    expect(warnSpy.mock.calls).toEqual(c.warning ? [[c.warning]] : []);
    expect(state.disposed).toEqual(c.disposed);
    if (c.config.device === 'cpu' || c.config.device === 'auto') {
      expect(pipelineSpy).not.toHaveBeenCalledWith('dml');
    }
  });
});
