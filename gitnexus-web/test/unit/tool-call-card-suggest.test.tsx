import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ToolCallCard } from '../../src/components/ToolCallCard';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const diff = [
  'Suggested change (not applied): rename',
  '',
  '--- a/src/auth.ts',
  '+++ b/src/auth.ts',
  '@@ -2,1 +2,1 @@',
  '-const oldName = 1;',
  '+const newName = 1;',
  '---- removed SQL comment',
  '+++counter;',
].join('\n');

function renderCard(name: string, result: string) {
  render(
    <ToolCallCard
      toolCall={{ id: 't1', name, args: {}, result, status: 'completed' }}
      defaultExpanded
    />,
  );
}

describe('ToolCallCard suggest diff', () => {
  it.each([
    { line: '+const newName = 1;', color: 'text-emerald-400' },
    { line: '-const oldName = 1;', color: 'text-rose-400' },
    { line: '@@ -2,1 +2,1 @@', color: 'text-sky-400' },
    { line: '---- removed SQL comment', color: 'text-rose-400' },
    { line: '+++counter;', color: 'text-emerald-400' },
  ])('colors the $line line', ({ line, color }) => {
    renderCard('suggest', diff);
    expect(screen.getByText(line)).toHaveClass(color);
  });

  it('leaves the file header lines uncolored', () => {
    renderCard('suggest', diff);
    expect(screen.getByText('+++ b/src/auth.ts').className).toBe('');
  });

  it('shows the end of a long diff instead of cutting it at 3000 characters', () => {
    const long =
      diff + '\n' + Array.from({ length: 200 }, (_, i) => `+line ${i} padding`).join('\n');
    renderCard('suggest', long);
    expect(screen.getByText('+line 199 padding')).toBeInTheDocument();
    expect(screen.queryByText(/common:progress.truncated/)).not.toBeInTheDocument();
  });

  it('keeps the 3000-character cut for other tools', () => {
    renderCard('read', 'x'.repeat(4000));
    expect(screen.getByText(/common:progress.truncated/)).toBeInTheDocument();
  });
});
