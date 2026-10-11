import { fromMarkdown } from 'mdast-util-from-markdown';

type MarkdownNode = {
  type: string;
  position?: { start: { offset?: number }; end: { offset?: number } };
  children?: MarkdownNode[];
};

/** Adapt model math notation for remark-math without changing the saved answer. */
export function normalizeExplanationMarkdown(source: string): string {
  const codeRanges: { start: number; end: number }[] = [];
  function collectCode(node: MarkdownNode) {
    if (node.type === 'code' || node.type === 'inlineCode') {
      const start = node.position?.start.offset;
      const end = node.position?.end.offset;
      if (start !== undefined && end !== undefined) codeRanges.push({ start, end });
    } else {
      node.children?.forEach(collectCode);
    }
  }
  // Use Markdown's own code boundaries, including indentation and nested lists.
  collectCode(fromMarkdown(source));
  let output = '';
  let start = 0;
  for (const code of codeRanges) {
    output += normalizeMath(source.slice(start, code.start)) + source.slice(code.start, code.end);
    start = code.end;
  }
  return output + normalizeMath(source.slice(start));
}

function escaped(source: string, index: number): boolean {
  let slashes = 0;
  while (index > 0 && source[--index] === '\\') slashes++;
  return slashes % 2 === 1;
}

function normalizeMath(source: string): string {
  let output = '';
  for (let index = 0; index < source.length;) {
    const pair = source.slice(index, index + 2);
    const slashMath = (pair === '\\(' || pair === '\\[') && !escaped(source, index);
    const dollarMath = source[index] === '$' && !escaped(source, index);
    if (!slashMath && !dollarMath) {
      output += source[index++];
      continue;
    }
    const opening = slashMath ? pair : source[index + 1] === '$' ? '$$' : '$';
    const closing = slashMath ? pair === '\\(' ? '\\)' : '\\]' : opening;
    let end = source.indexOf(closing, index + opening.length);
    while (end !== -1 && (escaped(source, end) || (!slashMath && (source[end - 1] === '$' || source[end + closing.length] === '$')))) end = source.indexOf(closing, end + closing.length);
    if (end === -1) {
      output += source[index++];
      continue;
    }
    const body = escapeTextUnderscores(source.slice(index + opening.length, end));
    if (slashMath) output += pair === '\\[' ? `\n\n$$\n${body.trim()}\n$$\n\n` : `$${body}$`;
    else output += opening + body + closing;
    index = end + closing.length;
  }
  return output;
}

// KaTeX expects literal underscores in text commands to be escaped. Mathematical
// subscripts (x_i), and underscores already escaped by the model, stay intact.
function escapeTextUnderscores(source: string): string {
  const commands = /\\(?:text(?:normal|rm|sf|tt|bf|it)?|operatorname\*?)\s*\{/g;
  let output = '';
  let start = 0;
  let command: RegExpExecArray | null;
  while ((command = commands.exec(source))) {
    if (escaped(source, command.index)) continue;
    output += source.slice(start, commands.lastIndex);
    let depth = 1;
    let index = commands.lastIndex;
    for (; index < source.length && depth; index++) {
      const character = source[index];
      if (character === '\\' && index + 1 < source.length) output += character + source[++index];
      else {
        if (character === '{') depth++;
        if (character === '}') depth--;
        output += character === '_' ? '\\_' : character;
      }
    }
    start = index;
    commands.lastIndex = index;
  }
  return output + source.slice(start);
}
