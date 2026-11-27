import { describe, expect, it, vi } from 'vitest';
import { createGraphRAGTools, type GraphRAGBackend } from '../../src/core/llm/tools';
import { BackendError } from '../../src/services/backend-client';

const SOURCE = ['line1', 'line2', 'line3', 'line4', 'line5'].join('\n') + '\n';

function makeBackend(readFile = vi.fn(async (_path: string) => SOURCE)) {
  const backend = {
    executeQuery: vi.fn(async () => []),
    search: vi.fn(async () => ({ results: [] })),
    grep: vi.fn(async () => ({ results: [], timedOut: false })),
    readFile,
  };
  return { backend, readFile };
}

async function suggest(
  args: { startLine: number; endLine: number; replacement: string },
  backend: GraphRAGBackend = makeBackend().backend,
): Promise<string> {
  const tool = createGraphRAGTools(backend).find((candidate) => candidate.name === 'suggest');
  if (!tool) throw new Error('suggest tool is not registered');
  return tool.invoke({ filePath: 'src/auth.ts', reason: 'test', ...args });
}

/** The diff lines after the hunk header, without the final patch terminator. */
const hunkBody = (output: string) => {
  expect(output.endsWith('\n')).toBe(true);
  const lines = output.split('\n');
  return lines.slice(lines.findIndex((l) => l.startsWith('@@')) + 1, -1);
};

describe('suggest tool (read-only diff proposals)', () => {
  it('is registered alongside the other agent tools', () => {
    expect(createGraphRAGTools(makeBackend().backend).map((t) => t.name)).toContain('suggest');
  });

  it.each([
    {
      name: 'replaces a middle range',
      startLine: 2,
      endLine: 3,
      replacement: 'new2\nnew3',
      header: '@@ -2,2 +2,2 @@',
      body: ['-line2', '-line3', '+new2', '+new3'],
    },
    {
      name: 'replaces the last line of a file that ends in a newline',
      startLine: 5,
      endLine: 5,
      replacement: 'last',
      header: '@@ -5,1 +5,1 @@',
      body: ['-line5', '+last'],
    },
    {
      name: 'ignores a trailing newline in the replacement',
      startLine: 1,
      endLine: 1,
      replacement: 'first\n',
      header: '@@ -1,1 +1,1 @@',
      body: ['-line1', '+first'],
    },
    {
      name: 'deletes a range when the replacement is empty',
      startLine: 2,
      endLine: 3,
      replacement: '',
      header: '@@ -2,2 +1,0 @@',
      body: ['-line2', '-line3'],
    },
  ])('returns a diff hunk for a line range: $name', async (c) => {
    const output = await suggest(c);
    expect(output).toContain('--- a/src/auth.ts\n+++ b/src/auth.ts');
    expect(output).toContain(c.header);
    expect(hunkBody(output)).toEqual(c.body);
  });

  it('keeps CRLF line endings so the hunk matches the file bytes', async () => {
    const { backend } = makeBackend(vi.fn(async () => 'a\r\nb\r\nc\r\n'));
    const output = await suggest({ startLine: 2, endLine: 2, replacement: 'B\r\n' }, backend);
    expect(hunkBody(output)).toEqual(['-b\r', '+B\r']);
  });

  it('leaves no carriage return on an unterminated last line of a CRLF file', async () => {
    const { backend } = makeBackend(vi.fn(async () => 'a\r\nb'));
    const output = await suggest({ startLine: 1, endLine: 2, replacement: 'A\nB' }, backend);
    expect(hunkBody(output)).toEqual([
      '-a\r',
      '-b',
      '\\ No newline at end of file',
      '+A\r',
      '+B',
      '\\ No newline at end of file',
    ]);
  });

  it.each([
    {
      name: 'replacing it',
      replacement: 'Z',
      body: ['-c', '\\ No newline at end of file', '+Z', '\\ No newline at end of file'],
    },
    { name: 'deleting it', replacement: '', body: ['-c', '\\ No newline at end of file'] },
  ])('marks an unterminated last line when $name', async (c) => {
    const { backend } = makeBackend(vi.fn(async () => 'a\nb\nc'));
    const output = await suggest({ startLine: 3, endLine: 3, replacement: c.replacement }, backend);
    expect(hunkBody(output)).toEqual(c.body);
  });

  it('rejects any range in an empty file', async () => {
    const { backend } = makeBackend(vi.fn(async () => ''));
    const output = await suggest({ startLine: 1, endLine: 1, replacement: 'x' }, backend);
    expect(output).toBe(
      'Invalid line range 1-1: "src/auth.ts" has 0 lines. Use read to check line numbers.',
    );
  });

  it('only reads the file: no other backend call is made', async () => {
    const { backend, readFile } = makeBackend();
    await suggest({ startLine: 1, endLine: 1, replacement: 'x' }, backend);
    expect(readFile).toHaveBeenCalledExactlyOnceWith('src/auth.ts');
    expect(backend.executeQuery).not.toHaveBeenCalled();
    expect(backend.search).not.toHaveBeenCalled();
    expect(backend.grep).not.toHaveBeenCalled();
  });

  it.each([
    { name: 'past the end of the file', startLine: 4, endLine: 99 },
    { name: 'onto the phantom line after a trailing newline', startLine: 6, endLine: 6 },
    { name: 'starting before line 1', startLine: 0, endLine: 2 },
    { name: 'ending before it starts', startLine: 3, endLine: 2 },
  ])('rejects a range $name without throwing', async (c) => {
    const output = await suggest({ ...c, replacement: 'x' });
    expect(output).toBe(
      `Invalid line range ${c.startLine}-${c.endLine}: "src/auth.ts" has 5 lines. Use read to check line numbers.`,
    );
  });

  it('refuses a range whose diff would flood the context', async () => {
    const big = Array.from({ length: 2000 }, (_, i) => `const value${i} = ${'x'.repeat(30)};`).join(
      '\n',
    );
    const { backend } = makeBackend(vi.fn(async () => big));
    const output = await suggest({ startLine: 1, endLine: 2000, replacement: 'x' }, backend);
    expect(output).toMatch(/too large for one suggestion/);
  });

  it('points the model at grep when the file does not exist', async () => {
    const { backend } = makeBackend(
      vi.fn(async () => {
        throw new BackendError('File not found', 404, 'not_found');
      }),
    );
    const output = await suggest({ startLine: 1, endLine: 1, replacement: 'x' }, backend);
    expect(output).toBe('File not found: "src/auth.ts". Use grep to search for the correct path.');
  });
});
