/**
 * End-to-end HTTP test of the GITNEXUS_ANALYZE_ROOT confinement on
 * POST /api/analyze.
 *
 * With the variable set, the route resolves the requested "path" (including
 * `..` segments and symlinks) and refuses anything that lands outside the
 * configured root. Rejections return before any job starts, so those cases
 * are hermetic. The accepted case starts a job on an empty temp directory.
 *
 * Own file because /api/analyze is rate-limited to 10 requests/minute per IP.
 * Mirrors the spawn+health-poll harness in server-analyze-branch-validation.test.ts.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DIST_CLI = path.join(REPO_ROOT, 'dist', 'cli', 'index.js');
const STARTUP_BUDGET_MS = process.env.CI ? 30_000 : 15_000;

// dupe-ok: same per-file serve harness as the other server-analyze-* suites
const allocateFreePort = (): Promise<number> =>
  new Promise((resolve, reject) => {
    const probe = http.createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const addr = probe.address();
      if (typeof addr !== 'object' || !addr) {
        probe.close();
        reject(new Error('could not allocate ephemeral port'));
        return;
      }
      const port = addr.port;
      probe.close((err) => (err ? reject(err) : resolve(port)));
    });
  });

// dupe-ok: same per-file serve harness as the other server-analyze-* suites
const httpJson = (
  port: number,
  method: string,
  reqPath: string,
  body?: unknown,
): Promise<{ status: number; body: string }> =>
  new Promise((resolve, reject) => {
    const payload = body === undefined ? undefined : JSON.stringify(body);
    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path: reqPath,
        method,
        headers: payload
          ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) }
          : {},
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () =>
          resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf8') }),
        );
      },
    );
    req.on('error', reject);
    req.setTimeout(5_000, () => {
      req.destroy();
      reject(new Error(`${method} ${reqPath} timed out`));
    });
    if (payload) req.write(payload);
    req.end();
  });

const postAnalyze = (port: number, body: unknown) => httpJson(port, 'POST', '/api/analyze', body);

// Spawned `serve` on Windows can report ready before the socket is reachable
// from the parent (see server-http-startup.test.ts), and creating symlinks
// there needs extra privileges.
const describeBlock = process.platform === 'win32' ? describe.skip : describe;

describeBlock('POST /api/analyze with GITNEXUS_ANALYZE_ROOT (real server)', () => {
  let proc: ChildProcessWithoutNullStreams | undefined;
  let tmpDir = '';
  let root = '';
  let outside = '';
  let port = 0;

  beforeAll(async () => {
    if (!fs.existsSync(DIST_CLI)) {
      throw new Error(`Missing ${DIST_CLI} — run npm run build before integration tests`);
    }

    port = await allocateFreePort();
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'gitnexus-analyze-root-'));
    const homeDir = path.join(tmpDir, 'home');
    root = path.join(tmpDir, 'allowed');
    outside = path.join(tmpDir, 'outside');
    for (const dir of [homeDir, path.join(root, 'repo'), outside, `${root}-evil`]) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.symlinkSync(outside, path.join(root, 'escape'), 'dir');
    fs.symlinkSync(path.join(root, 'repo'), path.join(root, 'alias'), 'dir');

    proc = spawn(
      process.execPath,
      [DIST_CLI, 'serve', '--port', String(port), '--host', '127.0.0.1'],
      {
        cwd: REPO_ROOT,
        env: {
          ...process.env,
          GITNEXUS_HOME: homeDir,
          GITNEXUS_ANALYZE_ROOT: root,
          NODE_OPTIONS: '',
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );

    let stderr = '';
    proc.stderr.on('data', (buf) => {
      stderr += buf.toString();
    });

    const startedAt = Date.now();
    while (Date.now() - startedAt < STARTUP_BUDGET_MS) {
      if (proc.exitCode !== null) {
        throw new Error(`serve exited ${proc.exitCode} before ready.\nstderr:\n${stderr}`);
      }
      try {
        const { status } = await httpJson(port, 'GET', '/api/health');
        if (status === 200) return;
      } catch {
        // Server still starting — retry until budget expires.
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error(
      `serve did not become ready within ${STARTUP_BUDGET_MS}ms.\nstderr:\n${stderr}`,
    );
  }, 60_000);

  afterAll(async () => {
    if (proc && !proc.killed) {
      proc.kill('SIGTERM');
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          proc?.kill('SIGKILL');
          resolve();
        }, 3_000);
        proc?.on('exit', () => {
          clearTimeout(timer);
          resolve();
        });
      });
    }
    proc = undefined;
    if (tmpDir) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
      tmpDir = '';
    }
  });

  it.each([
    { name: 'a directory outside the root', body: () => ({ path: outside }), field: 'path' },
    {
      name: 'a ".." escape from inside the root',
      // Built by hand: path.join would normalize the `..` away before sending.
      body: () => ({ path: `${root}${path.sep}..${path.sep}outside` }),
      field: 'path',
    },
    {
      name: 'a symlink inside the root that points outside',
      body: () => ({ path: path.join(root, 'escape') }),
      field: 'path',
    },
    {
      name: 'a sibling that only shares the root as a prefix',
      body: () => ({ path: `${root}-evil` }),
      field: 'path',
    },
    { name: 'the filesystem root', body: () => ({ path: path.parse(root).root }), field: 'path' },
    {
      name: 'a springActuatorPath that climbs out of the root',
      body: () => ({ path: path.join(root, 'repo'), springActuatorPath: '../../outside' }),
      field: 'springActuatorPath',
    },
    {
      name: 'an asyncApiSpecPath for a cloned repo',
      body: () => ({ url: 'https://github.com/owner/repo', asyncApiSpecPath: 'specs' }),
      error: '"asyncApiSpecPath" needs a local "path" when GITNEXUS_ANALYZE_ROOT is set',
    },
  ])('rejects $name with 403', async ({ body: makeBody, field, error }) => {
    const { status, body } = await postAnalyze(port, makeBody());
    expect(status).toBe(403);
    expect(JSON.parse(body).error).toBe(error ?? `"${field}" is outside GITNEXUS_ANALYZE_ROOT`);
  });

  it('rejects a path inside the root that does not exist', async () => {
    const { status, body } = await postAnalyze(port, { path: path.join(root, 'missing') });
    expect(status).toBe(400);
    expect(JSON.parse(body).error).toBe('"path" does not exist');
  });

  it('accepts an in-root symlink and queues the job on its resolved target', async () => {
    const { status, body } = await postAnalyze(port, { path: path.join(root, 'alias') });
    expect(status).toBe(202);
    const { jobId } = JSON.parse(body);
    // The job's repo name comes from the path it was queued with, so it reads
    // "repo" (the validated realpath) rather than "alias" (the request).
    const job = await httpJson(port, 'GET', `/api/analyze/${jobId}`);
    expect(JSON.parse(job.body).repoName).toBe('repo');
    // Cancel and wait for the worker to exit so afterAll does not delete its
    // directory mid-run. 400 means the job already finished.
    const cancel = await httpJson(port, 'DELETE', `/api/analyze/${jobId}`);
    expect([200, 400]).toContain(cancel.status);
    const deadline = Date.now() + 20_000;
    let jobStatus = JSON.parse(job.body).status;
    while (jobStatus !== 'complete' && jobStatus !== 'failed' && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 200));
      jobStatus = JSON.parse((await httpJson(port, 'GET', `/api/analyze/${jobId}`)).body).status;
    }
    expect(['complete', 'failed']).toContain(jobStatus);
  });
});
