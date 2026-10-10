'use client';

import { useEffect, useRef, useState } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { ArrowRight, BookOpen, Check, Code2, Copy, RefreshCw, Sparkles, X } from 'lucide-react';
import s from './solution.module.css';

type Solution = {
  id: string;
  title: string;
  difficulty: string;
  code: string;
  explanation: string;
};

type Props = { taskId: string; user: string; onClose: () => void };

export default function SolutionDialog({ taskId, user, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [solution, setSolution] = useState<Solution | null>(null);
  const [tab, setTab] = useState<'code' | 'explanation'>('code');
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState('');
  const codeTabRef = useRef<HTMLButtonElement>(null);
  const explanationTabRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog?.showModal();
    return () => {
      dialog?.close();
      opener?.focus();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`http://localhost:8000/api/tasks/${encodeURIComponent(taskId)}/solution`, { signal: controller.signal })
      .then(async response => {
        const body = await response.json();
        if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'We couldn’t load this solution.');
        return body as Solution;
      })
      .then(result => { if (!controller.signal.aborted) { setSolution(result); setError(''); } })
      .catch(err => { if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'We couldn’t load this solution.'); });
    return () => controller.abort();
  }, [taskId, retry]);

  async function copyCode() {
    if (!solution) return;
    try {
      await navigator.clipboard.writeText(solution.code);
      setCopied(true); setCopyError('');
    } catch {
      setCopyError('Copy isn’t available here. You can select and copy the code below.');
    }
  }

  function handleTabKeys(event: React.KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'code' : event.key === 'End' ? 'explanation' : tab === 'code' ? 'explanation' : 'code';
    setTab(next);
    (next === 'code' ? codeTabRef : explanationTabRef).current?.focus();
  }

  return <dialog
    ref={dialogRef}
    className={s.dialog}
    aria-labelledby="solution-title"
    onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => {
      if (event.target !== event.currentTarget) return;
      const rect = event.currentTarget.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose();
    }}
  >
    <header className={s.header}>
      <span className={s.headerIcon}><BookOpen size={23} /></span>
      <div><span className={s.eyebrow}>A LITTLE GUIDANCE. A BIG AHA.</span><h2 id="solution-title">{solution?.title || 'Your solution guide'}</h2><p>{solution ? `${solution.difficulty} · Reference implementation & walkthrough` : 'Opening your reference notebook…'}</p></div>
      <button onClick={onClose} className={s.closeButton} aria-label="Close solution"><X size={21} /></button>
    </header>

    <div className={s.tabs} role="tablist" aria-label="Solution guide" onKeyDown={handleTabKeys}>
      <button ref={codeTabRef} id="solution-code-tab" role="tab" aria-selected={tab === 'code'} aria-controls="solution-code-panel" tabIndex={tab === 'code' ? 0 : -1} className={tab === 'code' ? s.activeTab : ''} onClick={() => setTab('code')}><Code2 size={17} />Solution Code</button>
      <button ref={explanationTabRef} id="solution-explanation-tab" role="tab" aria-selected={tab === 'explanation'} aria-controls="solution-explanation-panel" tabIndex={tab === 'explanation' ? 0 : -1} className={tab === 'explanation' ? s.activeTab : ''} onClick={() => setTab('explanation')}><Sparkles size={17} />Explanation</button>
    </div>

    {!solution ? <div className={s.status} role={error ? 'alert' : 'status'}><BookOpen size={33} /><h3>{error ? 'Let’s try that again.' : 'A little clarity is on its way…'}</h3><p>{error || 'Loading the reference code and its explanation.'}</p>{error && <button onClick={() => { setError(''); setRetry(value => value + 1); }}><RefreshCw size={15} />Try again</button>}</div> : <>
      <section id="solution-code-panel" role="tabpanel" aria-labelledby="solution-code-tab" hidden={tab !== 'code'} className={s.codePanel}>
        <div className={s.codeToolbar}><span><i /> Python · PyTorch</span><button onClick={() => void copyCode()}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? 'Copied!' : 'Copy code'}</button></div>
        {copyError && <p className={s.copyError} role="status">{copyError}</p>}
        <div className={s.editor}><Editor language="python" theme="vs-dark" value={solution.code} loading={<span className={s.editorLoading}>Loading code…</span>} options={{ readOnly: true, domReadOnly: true, minimap: { enabled: false }, automaticLayout: true, fontSize: 13, lineHeight: 23, scrollBeyondLastLine: false, padding: { top: 17, bottom: 17 }, renderLineHighlight: 'none', contextmenu: false, wordWrap: 'on', scrollbar: { verticalScrollbarSize: 7, horizontalScrollbarSize: 7 } }} /></div>
      </section>
      <section id="solution-explanation-panel" role="tabpanel" aria-labelledby="solution-explanation-tab" hidden={tab !== 'explanation'} className={s.explanationPanel} tabIndex={0}>
        <div className={s.explanation}><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{solution.explanation}</ReactMarkdown></div>
      </section>
    </>}

    <footer className={s.footer}><span><Sparkles size={15} />Understand the idea. Then make it your own.</span><a href={`http://localhost:3000/?task=${encodeURIComponent(taskId)}&user=${encodeURIComponent(user)}`} target="_blank" rel="noreferrer">Back to practice<ArrowRight size={15} /></a></footer>
  </dialog>;
}
