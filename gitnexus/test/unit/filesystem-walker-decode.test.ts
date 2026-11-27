import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { readFileContents } from '../../src/core/ingestion/filesystem-walker.js';

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })),
  );
});

describe('readFileContents decoding', () => {
  const cases: Array<{ name: string; bytes: Buffer; expected: string }> = [
    {
      name: 'keeps valid UTF-8 Japanese text unchanged',
      bytes: Buffer.from('void 日本語() {}', 'utf8'),
      expected: 'void 日本語() {}',
    },
    {
      name: 'decodes Shift-JIS bytes instead of producing mojibake',
      // "void " + Shift-JIS 日本語 (93 FA 96 7B 8C EA) + "() {}"
      bytes: Buffer.concat([
        Buffer.from('void '),
        Buffer.from([0x93, 0xfa, 0x96, 0x7b, 0x8c, 0xea]),
        Buffer.from('() {}'),
      ]),
      expected: 'void 日本語() {}',
    },
    {
      name: 'keeps a leading UTF-8 BOM',
      bytes: Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from('int x;')]),
      expected: '\uFEFFint x;',
    },
    {
      name: 'falls back to lossy UTF-8 when bytes are neither UTF-8 nor Shift-JIS',
      bytes: Buffer.concat([Buffer.from('int x; // '), Buffer.from([0xff])]),
      expected: 'int x; // \uFFFD',
    },
    {
      name: 'keeps UTF-8 text in a mostly UTF-8 file with one stray byte',
      // 0x93 is a cp1252 quote; this whole byte stream also decodes as valid Shift-JIS
      bytes: Buffer.concat([Buffer.from('/* 日本語x'), Buffer.from([0x93]), Buffer.from('x */')]),
      expected: '/* 日本語x\uFFFDx */',
    },
    {
      name: 'does not count a literal U+FFFD as a decoding error',
      bytes: Buffer.concat([Buffer.from('/* あ\uFFFDx'), Buffer.from([0x93]), Buffer.from('x */')]),
      expected: '/* あ\uFFFDx\uFFFDx */',
    },
    {
      name: 'keeps plain ASCII unchanged',
      bytes: Buffer.from('int main() { return 0; }'),
      expected: 'int main() { return 0; }',
    },
  ];

  it.each(cases)('$name', async ({ bytes, expected }) => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'gitnexus-decode-'));
    temporaryRoots.push(root);
    await fs.writeFile(path.join(root, 'a.cpp'), bytes);
    const contents = await readFileContents(root, ['a.cpp']);
    expect(contents.get('a.cpp')).toBe(expected);
  });
});
