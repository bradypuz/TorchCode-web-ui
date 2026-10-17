'use client';

import { createContext, useContext, useEffect, useId, useRef, useState } from 'react';
import type { CodeTheme } from '../useCodeTheme';
import s from './mermaid-diagram.module.css';

// Visibility changes should not remount the other interactive markdown widgets.
export const MermaidVisibilityContext = createContext(false);
export const MermaidThemeContext = createContext<CodeTheme>('dark');

let mermaidPromise: Promise<typeof import('mermaid')['default']> | undefined;
let renderQueue: Promise<unknown> = Promise.resolve();

function loadMermaid() {
  mermaidPromise ??= import('mermaid').then(({ default: mermaid }) => mermaid).catch(error => {
    mermaidPromise = undefined;
    throw error;
  });
  return mermaidPromise;
}

function renderDiagram(mermaid: Awaited<ReturnType<typeof loadMermaid>>, id: string, source: string, theme: CodeTheme, container: HTMLDivElement) {
  // Mermaid's theme configuration is global. Keep initialization and rendering
  // together so simultaneous diagrams always receive their requested palette.
  const rendering = renderQueue.then(() => {
    const dark = theme === 'dark';
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      suppressErrorRendering: true,
      htmlLabels: false,
      flowchart: { rankSpacing: 24, nodeSpacing: 28 },
      theme: 'base',
      fontFamily: 'Arial, sans-serif',
      themeVariables: {
        darkMode: dark,
        fontSize: '14px',
        primaryColor: dark ? '#2f3d2b' : '#edf4df',
        primaryTextColor: dark ? '#e0e8da' : '#263d30',
        primaryBorderColor: dark ? '#8fb76d' : '#85a763',
        lineColor: dark ? '#b3c0aa' : '#647655',
        textColor: dark ? '#e0e8da' : '#263d30',
        secondaryColor: dark ? '#283027' : '#f4f7ed',
        tertiaryColor: dark ? '#354331' : '#e9f0db',
        background: dark ? '#1c211d' : '#fcfef8',
        edgeLabelBackground: dark ? '#232b24' : '#fcfef8',
      },
    });
    return mermaid.render(id, source, container);
  });
  renderQueue = rendering.catch(() => undefined);
  return rendering;
}

type RenderedDiagram = { source: string; theme: CodeTheme; svg: string; failed: boolean };

function prepareSvg(svg: string) {
  const document = new DOMParser().parseFromString(svg, 'image/svg+xml');
  const element = document.documentElement;
  if (element.tagName !== 'svg') throw new Error('Invalid diagram SVG.');
  element.setAttribute('role', 'img');
  if (!element.hasAttribute('aria-label') && !element.hasAttribute('aria-labelledby')) {
    element.setAttribute('aria-label', 'Solution explanation diagram. Diagram source is available below.');
  }
  // Keep natural dimensions so wide diagrams scroll instead of shrinking their text.
  const width = Number(element.getAttribute('viewBox')?.split(/\s+/)[2]);
  if (Number.isFinite(width) && width > 0) {
    element.setAttribute('width', String(width));
    element.setAttribute('style', `width: ${width}px; max-width: none; height: auto;`);
  }
  return new XMLSerializer().serializeToString(element);
}

export default function MermaidDiagram({ source }: { source: string }) {
  const visible = useContext(MermaidVisibilityContext);
  const theme = useContext(MermaidThemeContext);
  const id = useId();
  const renderId = `solution-diagram-${id.replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const viewportRef = useRef<HTMLDivElement>(null);
  const sequenceRef = useRef(0);
  const [rendered, setRendered] = useState<RenderedDiagram | null>(null);
  const current = rendered?.source === source && rendered.theme === theme ? rendered : null;

  useEffect(() => {
    if (!visible || current) return;
    let cancelled = false;
    const sequence = ++sequenceRef.current;

    async function render() {
      let staging: HTMLDivElement | undefined;
      try {
        const mermaid = await loadMermaid();
        await document.fonts.ready;
        if (cancelled || !viewportRef.current) return;

        // An attached, measurable container inherits the viewer's font styles.
        // It also contains any temporary Mermaid output, including parse errors.
        staging = document.createElement('div');
        staging.className = s.staging;
        staging.setAttribute('aria-hidden', 'true');
        viewportRef.current.appendChild(staging);
        const { svg } = await renderDiagram(mermaid, `${renderId}-${sequence}`, source, theme, staging);
        if (!cancelled) setRendered({ source, theme, svg: prepareSvg(svg), failed: false });
      } catch {
        if (!cancelled) setRendered({ source, theme, svg: '', failed: true });
      } finally {
        staging?.remove();
      }
    }

    void render();
    return () => { cancelled = true; };
  }, [source, theme, visible, renderId, current]);

  return <figure className={s.figure}>
    <figcaption className={s.caption}>Explanation diagram</figcaption>
    <div
      ref={viewportRef}
      className={s.viewport}
      tabIndex={current?.svg ? 0 : undefined}
      role="region"
      aria-label="Explanation diagram"
      aria-busy={visible && !current}
    >
      {current?.svg
        ? <div className={s.diagram} dangerouslySetInnerHTML={{ __html: current.svg }} />
        : <p className={s.status} role="status">{current?.failed ? 'We couldn’t draw this diagram. Its source is available below.' : 'Loading diagram…'}</p>}
    </div>
    <details className={s.source} open={current?.failed || undefined}>
      <summary>Diagram source</summary>
      <pre><code>{source}</code></pre>
    </details>
  </figure>;
}
