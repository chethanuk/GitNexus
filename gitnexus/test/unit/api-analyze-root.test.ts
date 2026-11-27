/**
 * Unit tests for checkAnalyzeRoot, the GITNEXUS_ANALYZE_ROOT confinement on
 * POST /api/analyze, against real temp directories. The route wiring and the
 * escape cases are covered over HTTP in
 * test/integration/server-analyze-root.test.ts; these cover the configuration
 * edges that would otherwise each need their own server.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { checkAnalyzeRoot, isPathWithin } from '../../src/server/api.js';

describe('checkAnalyzeRoot', () => {
  let tmpDir = '';
  let root = '';

  beforeAll(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'gitnexus-analyze-root-unit-'));
    root = path.join(tmpDir, 'allowed');
    fs.mkdirSync(path.join(root, 'repo'), { recursive: true });
    fs.symlinkSync(root, path.join(tmpDir, 'root-link'), 'junction');
    fs.mkdirSync(path.join(root, 'repo', 'actuator'));
    fs.mkdirSync(path.join(tmpDir, 'outside'));
    fs.symlinkSync(path.join(tmpDir, 'outside'), path.join(root, 'repo', 'spec-link'), 'junction');
  });

  afterAll(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it.each([
    { name: 'no root is configured', setting: () => undefined },
    { name: 'the root is blank', setting: () => '   ' },
  ])('skips the check when $name', async ({ setting }) => {
    // An undefined setting falls back to the env var, so clear it for this case.
    vi.stubEnv('GITNEXUS_ANALYZE_ROOT', undefined);
    try {
      expect(await checkAnalyzeRoot(path.join(root, 'repo'), 'path', setting())).toBeNull();
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it.each([
    { name: 'the target is the root itself', setting: () => root, target: () => root },
    { name: 'the root has a trailing separator', setting: () => root + path.sep },
    { name: 'the root is the filesystem root', setting: () => path.parse(root).root },
    {
      name: 'the root is a symlink and the target uses the real path',
      setting: () => path.join(tmpDir, 'root-link'),
      target: () => path.join(fs.realpathSync(root), 'repo'),
    },
  ])('allows the target when $name', async ({ setting, target }) => {
    const localPath = target ? target() : path.join(root, 'repo');
    expect(await checkAnalyzeRoot(localPath, 'path', setting())).toEqual({
      realPath: fs.realpathSync(localPath),
    });
  });

  it.each([
    {
      name: 'a relative root',
      setting: () => 'allowed',
      expected: { status: 500, error: 'GITNEXUS_ANALYZE_ROOT must be an absolute path' },
    },
    {
      name: 'a root that does not exist',
      setting: () => path.join(tmpDir, 'missing'),
      target: () => path.join(tmpDir, 'missing', 'repo'),
      expected: { status: 500, error: 'GITNEXUS_ANALYZE_ROOT does not exist' },
    },
    {
      name: 'a missing root and a target outside it',
      setting: () => path.join(tmpDir, 'missing'),
      target: () => path.join(tmpDir, 'elsewhere'),
      expected: { status: 500, error: 'GITNEXUS_ANALYZE_ROOT does not exist' },
    },
  ])('refuses every target with $name', async ({ setting, target, expected }) => {
    const localPath = target ? target() : path.join(root, 'repo');
    expect(await checkAnalyzeRoot(localPath, 'path', setting())).toEqual(expected);
  });

  it('resolves an in-root symlink to its target so the job analyzes what was checked', async () => {
    const link = path.join(root, 'alias');
    fs.symlinkSync(path.join(root, 'repo'), link, 'junction');
    expect(await checkAnalyzeRoot(link, 'path', root)).toEqual({
      realPath: fs.realpathSync(path.join(root, 'repo')),
    });
  });

  // The route checks spec inputs as path.resolve(repoPath, spec) with the
  // field name, so these rows cover the spec branches without more HTTP calls
  // (the integration suite is near the 10/min rate limit).
  it.each([
    {
      name: 'refuses a spec symlink inside the repo that points outside the root',
      spec: 'spec-link',
      expected: () => ({
        status: 403,
        error: '"springActuatorPath" is outside GITNEXUS_ANALYZE_ROOT',
      }),
    },
    {
      name: 'reports a missing spec path',
      spec: 'no-such-spec',
      expected: () => ({ status: 400, error: '"springActuatorPath" does not exist' }),
    },
    {
      name: 'accepts a spec inside the repo and returns its realpath',
      spec: 'actuator',
      expected: () => ({ realPath: fs.realpathSync(path.join(root, 'repo', 'actuator')) }),
    },
  ])('$name', async ({ spec, expected }) => {
    const repo = path.join(root, 'repo');
    expect(await checkAnalyzeRoot(path.resolve(repo, spec), 'springActuatorPath', root)).toEqual(
      expected(),
    );
  });

  it('names the request field in the error', async () => {
    expect(await checkAnalyzeRoot(tmpDir, 'asyncApiSpecPath', root)).toEqual({
      status: 403,
      error: '"asyncApiSpecPath" is outside GITNEXUS_ANALYZE_ROOT',
    });
  });
});

describe('isPathWithin', () => {
  it.each([
    {
      name: 'win32: a child cased differently from the root',
      p: path.win32,
      root: 'C:\\Repos',
      target: 'c:\\repos\\app',
      within: true,
    },
    {
      name: 'win32: the root itself cased differently',
      p: path.win32,
      root: 'C:\\Repos',
      target: 'C:\\REPOS',
      within: true,
    },
    {
      name: 'win32: a sibling sharing the prefix',
      p: path.win32,
      root: 'C:\\Repos',
      target: 'c:\\repos-evil',
      within: false,
    },
    {
      name: 'win32: another drive',
      p: path.win32,
      root: 'C:\\Repos',
      target: 'D:\\Repos\\app',
      within: false,
    },
    {
      name: 'posix: a sibling sharing the prefix',
      p: path.posix,
      root: '/root',
      target: '/root-evil',
      within: false,
    },
    { name: 'posix: a parent', p: path.posix, root: '/root/a', target: '/root', within: false },
    {
      name: 'posix: a child cased differently',
      p: path.posix,
      root: '/root',
      target: '/ROOT/app',
      within: false,
    },
    {
      name: 'posix: a child whose name starts with ..',
      p: path.posix,
      root: '/root',
      target: '/root/..cache',
      within: true,
    },
    {
      name: 'posix: the filesystem root as root',
      p: path.posix,
      root: '/',
      target: '/srv/app',
      within: true,
    },
  ])('$name -> $within', ({ p, root, target, within }) => {
    expect(isPathWithin(root, target, p)).toBe(within);
  });
});
