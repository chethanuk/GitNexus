import { it, expect } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { getNodesByLabel, runPipelineFromRepo } from './resolvers/helpers.js';

it('indexes a Japanese function name from a Shift-JIS encoded C++ file', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gn-shift-jis-'));
  try {
    // "void " + Shift-JIS 日本語 (93 FA 96 7B 8C EA) + "() {}"
    fs.writeFileSync(
      path.join(root, 'legacy.cpp'),
      Buffer.concat([
        Buffer.from('void '),
        Buffer.from([0x93, 0xfa, 0x96, 0x7b, 0x8c, 0xea]),
        Buffer.from('() {}\n'),
      ]),
    );
    const result = await runPipelineFromRepo(root, () => {}, {});
    expect(getNodesByLabel(result, 'Function')).toContain('日本語');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}, 120000);
