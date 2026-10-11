import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync, readdirSync } from 'node:fs';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { normalizeExplanationMarkdown } from '../src/app/practice/explanationMarkdown.ts';

function render(source) {
  return renderToStaticMarkup(createElement(ReactMarkdown, {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex],
  }, normalizeExplanationMarkdown(source)));
}

test('preserves prose and heading choices while rendering both model math delimiters', () => {
  const source = String.raw`This code maps each input to \(y_i = x_i + 1\).

## The operation

\[
y = x + 1
\]

The output keeps the input shape.`;
  const normalized = normalizeExplanationMarkdown(source);
  assert.ok(normalized.startsWith('This code maps each input to $y_i = x_i + 1$.'));
  assert.ok(normalized.includes('## The operation'));
  assert.ok(normalized.endsWith('The output keeps the input shape.'));
  const html = render(source);
  assert.equal((html.match(/class="katex"/g) ?? []).length, 2);
  assert.equal((html.match(/class="katex-display"/g) ?? []).length, 1);
  assert.ok(!html.includes('katex-error'));
  assert.ok(!html.includes('<h1'));
  assert.ok(html.includes('<h2>The operation</h2>'));
});

test('protects fenced code with backtick and tilde fences, including longer fences', () => {
  const source = [
    'Before \\(x_i\\).',
    '````python',
    String.raw`formula = r"\(x_i\)"`,
    '```',
    String.raw`block = r"\[y\]"`,
    '````',
    '  ~~~text',
    String.raw`$\text{batch_size}$ and \(z\)`,
    '  ~~~',
    'After \\(y_i\\).',
  ].join('\n');
  const expected = source.replace('Before \\(x_i\\).', 'Before $x_i$.').replace('After \\(y_i\\).', 'After $y_i$.');
  assert.equal(normalizeExplanationMarkdown(source), expected);
  const html = render(source);
  assert.equal((html.match(/class="katex"/g) ?? []).length, 2);
  assert.ok(html.includes('language-python'));
  assert.ok(html.includes('language-text'));
});

test('protects inline code spans with matching backtick runs and multiline content', () => {
  const code = ['``', String.raw`\(x_i\) and $\text{batch_size}$`, '` inside', String.raw`\[y_i\]`, '``'].join('\n');
  const actual = 'Use \\(x_i\\), then ' + code + ' and `\\(z_i\\)`.';
  assert.equal(normalizeExplanationMarkdown(actual), 'Use $x_i$, then ' + code + ' and `\\(z_i\\)`.');
  assert.equal((render(actual).match(/class="katex"/g) ?? []).length, 1);
});

test('protects indented code without confusing indented list prose with code', () => {
  const code = '    ' + String.raw`formula = "\(x_i\) and $\text{batch_size}$"`;
  const source = 'Before \\(x_i\\).\n\n' + code + '\n\nAfter \\(y_i\\).';
  assert.equal(normalizeExplanationMarkdown(source), 'Before $x_i$.\n\n' + code + '\n\nAfter $y_i$.');
  assert.equal((render(source).match(/class="katex"/g) ?? []).length, 2);

  const list = '- A step\n\n    An indented explanation uses \\(x_i\\).\n\n    ```python\n    value = "\\(y_i\\)"\n    ```';
  assert.equal(normalizeExplanationMarkdown(list), list.replace('uses \\(x_i\\)', 'uses $x_i$'));
  assert.equal((render(list).match(/class="katex"/g) ?? []).length, 1);
});

test('leaves escaped or unmatched delimiters and unclosed code fences intact', () => {
  const source = 'Escaped \\\\(x\\\\), unmatched \\(x, and an unmatched backtick `';
  assert.equal(normalizeExplanationMarkdown(source), source);
  const unclosed = '```python\n' + String.raw`\(x_i\) and $\text{batch_size}$`;
  assert.equal(normalizeExplanationMarkdown(unclosed), unclosed);
});

test('escapes literal text underscores while preserving mathematical subscripts and existing escapes', () => {
  const source = String.raw`\(x_i = \text{batch_size} + \texttt{d_model} + \text{already\_escaped}\)

$x_i + \text{outer {inner_label}}$

$$
\operatorname{my_op}(x_i)
$$`;
  const normalized = normalizeExplanationMarkdown(source);
  assert.ok(normalized.includes(String.raw`x_i = \text{batch\_size} + \texttt{d\_model} + \text{already\_escaped}`));
  assert.ok(normalized.includes(String.raw`\text{outer {inner\_label}}`));
  assert.ok(normalized.includes(String.raw`\operatorname{my\_op}(x_i)`));
  const html = render(source);
  assert.equal((html.match(/class="katex"/g) ?? []).length, 3);
  assert.ok(!html.includes('katex-error'));
});

test('keeps GFM tables and Mermaid source usable beside math', () => {
  const source = '## Shapes\n\n| Input | Output |\n| --- | --- |\n| `x` | \\(y_i\\) |\n\n```mermaid\nflowchart LR\n  A["\\(x\\)"] --> B["y"]\n```';
  const html = render(source);
  assert.ok(html.includes('<table>'));
  assert.ok(html.includes('language-mermaid'));
  assert.equal((html.match(/class="katex"/g) ?? []).length, 1);
  assert.ok(normalizeExplanationMarkdown(source).includes('A["\\(x\\)"]'));
});

test('renders every saved explanation without a TeX parse error', () => {
  const directory = new URL('../../explanations/', import.meta.url);
  const files = readdirSync(directory).filter(file => /^\d+_.*\.md$/.test(file));
  assert.equal(files.length, 41);
  for (const file of files) {
    const html = render(readFileSync(new URL(file, directory), 'utf8'));
    assert.ok(!html.includes('katex-error'), `${file} contains math that failed to render`);
  }
});
