import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { cliAnalysisEnv, CLI_SPAWN_PREFIX } from '../helpers/cli-entry.js';

// Files GitNexus has no language parser for still become File nodes, so the
// analyze summary must say which extensions produced no symbols.
describe('analyze summary names files without a language parser', () => {
  it.each([
    {
      name: 'shell scripts next to parsed code',
      files: {
        'src/app.ts': 'export function run(): number { return 1; }\n',
        'scripts/build.sh': '#!/bin/sh\nbuild() { echo ok; }\n',
        'scripts/deploy.sh': '#!/bin/sh\ndeploy() { echo ok; }\n',
        'config.json': '{}\n',
      },
      expected: 'Note: 3 file(s) have no language parser: .sh (2), .json (1)',
    },
    {
      name: 'more extensions than the summary lists',
      files: {
        'src/app.ts': 'export const one = 1;\n',
        'a.sh': 'true\n',
        'b.sh': 'true\n',
        'c.yml': 'a: 1\n',
        'd.toml': 'a = 1\n',
        'e.ini': 'a=1\n',
        'f.txt': 'x\n',
        Makefile: 'all:\n\ttrue\n',
      },
      expected:
        'Note: 7 file(s) have no language parser: .sh (2), .ini (1), .toml (1), .txt (1), .yml (1), +1 more file(s)',
    },
    {
      name: 'exactly five extensions, mixed case, with a template',
      files: {
        'src/app.ts': 'export const one = 1;\n',
        'views/index.html': '<a href="/home">home</a>\n',
        'scripts/BUILD.SH': 'true\n',
        'b.yml': 'a: 1\n',
        'c.toml': 'a = 1\n',
        'd.ini': 'a=1\n',
        Makefile: 'all:\n\ttrue\n',
      },
      expected:
        'Note: 5 file(s) have no language parser: .ini (1), .sh (1), .toml (1), .yml (1), Makefile (1)',
    },
    {
      name: 'inputs handled outside tree-sitter',
      files: {
        'src/app.ts': 'export const one = 1;\n',
        'jobs/nightly.jcl': '//NIGHTLY JOB\n//STEP1 EXEC PGM=MAIN\n',
        'copy/record.copybook': '       01 REC PIC X(10).\n',
        'src/main/resources/application-dev.yml': 'server:\n  port: 8080\n',
        'src/main/resources/other.yml': 'a: 1\n',
        'run.sh': 'true\n',
      },
      expected: 'Note: 2 file(s) have no language parser: .sh (1), .yml (1)',
    },
    {
      name: 'GitNexus-managed dot directory left out',
      files: {
        'src/app.ts': 'export const one = 1;\n',
        '.claude/settings.json': '{}\n',
        'run.sh': 'true\n',
      },
      env: { GITNEXUS_INCLUDE_DOT_DIRS: '.claude' },
      expected: 'Note: 1 file(s) have no language parser: .sh (1)',
    },
    {
      name: 'only parsed code and markdown',
      files: {
        'src/app.ts': 'export const one = 1;\n',
        'README.md': '# Title\n',
      },
      expected: undefined,
    },
  ])(
    '$name',
    async ({
      files,
      env,
      expected,
    }: {
      files: Record<string, string>;
      env?: Record<string, string>;
      expected?: string;
    }) => {
      const root = await fs.mkdtemp(path.join(os.tmpdir(), 'gn-unparsed-note-'));
      const repo = path.join(root, 'repo');
      try {
        for (const [rel, content] of Object.entries(files)) {
          await fs.mkdir(path.dirname(path.join(repo, rel)), { recursive: true });
          await fs.writeFile(path.join(repo, rel), content);
        }
        const output = execFileSync(
          process.execPath,
          [
            ...CLI_SPAWN_PREFIX,
            'analyze',
            repo,
            '--index-only',
            '--no-share',
            '--force',
            '--skip-git',
          ],
          {
            cwd: repo,
            env: cliAnalysisEnv({
              GITNEXUS_HOME: path.join(root, 'home'),
              GITNEXUS_SHARED_STORE: 'off',
              GITNEXUS_WORKER_READY_TIMEOUT_MS: '60000',
              ...env,
            }),
            encoding: 'utf8',
            timeout: 90_000,
            stdio: ['ignore', 'pipe', 'pipe'],
          },
        );
        expect(output).toContain('Repository indexed successfully');
        if (expected) expect(output).toContain(expected);
        else expect(output).not.toContain('no language parser');
      } finally {
        await fs.rm(root, { recursive: true, force: true });
      }
    },
    120_000,
  );
});
