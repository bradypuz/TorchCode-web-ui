'use client';

import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AlertCircle, ArrowRight, Check, Clock3, Code2, History, LoaderCircle, RefreshCw, RotateCcw } from 'lucide-react';

const API = 'http://localhost:8000/api/practice/submissions';

type Submission = {
  id: number;
  task_id: string;
  success: boolean;
  passed: number;
  total: number;
  created_at: string;
  hint_used: boolean;
  has_code: boolean;
};

export type HistoricalSubmission = Submission & { code: string | null };
type HistoryPage = { submissions: Submission[]; next_before_id: number | null };
type Props = {
  userId: string;
  taskId: string;
  title: string;
  profileName: string;
  disabled: boolean;
  loadedId: number | null;
  canUndo: boolean;
  onLoad: (submission: HistoricalSubmission) => void;
  onUndo: () => void;
  onLoadingChange: (loading: boolean) => void;
};

async function readResponse<T>(response: Response): Promise<T> {
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Could not load submission history.');
  return body as T;
}

export default function SubmissionHistory({ userId, taskId, title, profileName, disabled, loadedId, canUndo, onLoad, onUndo, onLoadingChange }: Props) {
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [nextId, setNextId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [loadingId, setLoadingId] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [loadError, setLoadError] = useState('');
  const [retry, setRetry] = useState(0);
  const moreController = useRef<AbortController | null>(null);
  const detailController = useRef<AbortController | null>(null);
  const latestOnLoad = useRef(onLoad);

  useLayoutEffect(() => { latestOnLoad.current = onLoad; }, [onLoad]);

  useEffect(() => {
    const controller = new AbortController();
    const query = new URLSearchParams({ user_id: userId, task_id: taskId, limit: '20' });
    fetch(`${API}?${query}`, { signal: controller.signal })
      .then(readResponse<HistoryPage>)
      .then(page => {
        if (controller.signal.aborted) return;
        setSubmissions(page.submissions); setNextId(page.next_before_id); setError('');
      })
      .catch(err => { if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Could not load submission history.'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => {
      controller.abort();
      moreController.current?.abort();
      detailController.current?.abort();
      onLoadingChange(false);
    };
  }, [userId, taskId, retry, onLoadingChange]);

  async function loadMore() {
    if (nextId === null || loadingMore) return;
    const controller = new AbortController();
    moreController.current?.abort(); moreController.current = controller;
    setLoadingMore(true); setError('');
    const query = new URLSearchParams({ user_id: userId, task_id: taskId, limit: '20', before_id: String(nextId) });
    try {
      const page = await fetch(`${API}?${query}`, { signal: controller.signal }).then(readResponse<HistoryPage>);
      if (controller.signal.aborted) return;
      setSubmissions(previous => [...previous, ...page.submissions]); setNextId(page.next_before_id);
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Could not load older submissions.');
    } finally {
      if (!controller.signal.aborted) setLoadingMore(false);
    }
  }

  async function loadSubmission(submission: Submission) {
    if (disabled || loadingId !== null || !submission.has_code || loadedId === submission.id) return;
    const controller = new AbortController();
    detailController.current?.abort(); detailController.current = controller;
    setLoadingId(submission.id); setLoadError('');
    onLoadingChange(true);
    const query = new URLSearchParams({ user_id: userId, task_id: taskId });
    try {
      const detail = await fetch(`${API}/${submission.id}?${query}`, { signal: controller.signal }).then(readResponse<HistoricalSubmission>);
      if (controller.signal.aborted) return;
      if (detail.code === null) throw new Error('This older submission has test results only. Its code was not saved.');
      // The editor may change while this request is in flight. The current
      // callback saves the latest draft, rather than the version at click time.
      latestOnLoad.current(detail);
    } catch (err) {
      if (!controller.signal.aborted) setLoadError(err instanceof Error ? err.message : 'Could not load this code.');
    } finally {
      if (!controller.signal.aborted) { setLoadingId(null); onLoadingChange(false); }
    }
  }

  return <section id="submission-history-pane" aria-labelledby="submission-history-heading" className="space-y-5">
    <div>
      <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[var(--workspace-accent)]"><History className="h-4 w-4" />Your previous attempts</div>
      <h2 id="submission-history-heading" className="text-2xl font-semibold text-[var(--workspace-ink)]">Submission history</h2>
      <p className="mt-2 text-sm leading-6 text-[var(--workspace-muted)]">{title} · {profileName}<br />Choose an attempt to load its code into the editor.</p>
    </div>

    {canUndo && <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-orange-400/25 bg-orange-400/10 px-4 py-3">
      <span className="text-sm text-[var(--workspace-accent)]" role="status">{loadedId !== null ? `Loaded submission #${loadedId}. Ready to revisit.` : 'Your previous draft is saved for this session.'}</span>
      <button onClick={onUndo} disabled={disabled || loadingId !== null} className="flex items-center gap-2 rounded-md border border-orange-300/30 px-3 py-1.5 text-xs font-medium text-[var(--workspace-accent)] transition-colors hover:bg-orange-400/15 disabled:opacity-50"><RotateCcw className="h-3.5 w-3.5" />Undo load</button>
    </div>}

    {loadError && <p className="flex items-start gap-2 rounded-lg border border-red-500/25 bg-red-500/10 p-3 text-sm text-[var(--workspace-danger)]" role="alert"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{loadError}</p>}

    {loading ? <div className="flex items-center gap-3 py-10 text-sm text-[var(--workspace-muted)]" role="status"><LoaderCircle className="h-5 w-5 animate-spin" />Loading your submissions…</div> : <>
      {!error && !submissions.length && <div className="rounded-xl border border-dashed border-[var(--workspace-border-strong)] bg-[var(--workspace-surface)] px-6 py-10 text-center"><Code2 className="mx-auto mb-4 h-9 w-9 text-[var(--workspace-faint)]" /><h3 className="text-base font-medium text-[var(--workspace-text)]">Your first attempt starts here.</h3><p className="mt-2 text-sm leading-6 text-[var(--workspace-muted)]">Run Tests to save your code and test results.<br />They’ll be here whenever you want to look back.</p></div>}

      {submissions.some(item => !item.has_code) && <p className="rounded-lg border border-[var(--workspace-border)] bg-[var(--workspace-surface)] p-3 text-xs leading-5 text-[var(--workspace-muted)]">Older submissions have test results only. Their code was not saved.</p>}

      <div className="space-y-3">{submissions.map(submission => {
        const loaded = loadedId === submission.id;
        const passedFraction = submission.total > 0 ? Math.min(100, Math.max(0, submission.passed / submission.total * 100)) : 0;
        const status = submission.success ? 'Accepted' : submission.total === 0 ? 'Execution failed' : 'Keep practicing';
        return <button
          key={submission.id}
          disabled={disabled || loadingId !== null || !submission.has_code || loaded}
          onClick={() => void loadSubmission(submission)}
          aria-label={`Load submission #${submission.id}, ${submission.passed} of ${submission.total} tests passed`}
          aria-pressed={loaded}
          className={`group w-full rounded-xl border p-4 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-orange-400 disabled:cursor-default ${loaded ? 'border-orange-400/50 bg-orange-400/10' : 'border-[var(--workspace-border)] bg-[var(--workspace-raised)] enabled:hover:border-[var(--workspace-faint)] enabled:hover:bg-[var(--workspace-control)]'} ${!submission.has_code ? 'opacity-70' : ''}`}
        >
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2"><span className="text-sm font-semibold text-[var(--workspace-text)]">Submission #{submission.id}</span><span className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${submission.success ? 'bg-emerald-400/10 text-[var(--workspace-emerald)]' : 'bg-amber-400/10 text-[var(--workspace-warning)]'}`}>{submission.success && <Check className="h-3.5 w-3.5" />}{status}</span></div>
          <div className="flex items-baseline gap-2"><span className={`text-xl font-semibold ${submission.success ? 'text-[var(--workspace-emerald)]' : 'text-[var(--workspace-text)]'}`}>{submission.passed}<span className="text-sm font-normal text-[var(--workspace-muted)]"> / {submission.total}</span></span><span className="text-xs text-[var(--workspace-muted)]">tests passed</span></div>
          <div className="my-3 h-1 overflow-hidden rounded-full bg-[var(--workspace-border)]" aria-hidden="true"><div className={`h-full rounded-full ${submission.success ? 'bg-emerald-400' : 'bg-amber-400'}`} style={{ width: `${passedFraction}%` }} /></div>
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs"><span className="flex items-center gap-1.5 text-[var(--workspace-muted)]"><Clock3 className="h-3.5 w-3.5" /><time dateTime={submission.created_at}>{new Date(submission.created_at).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' })}</time>{submission.hint_used && <span className="text-[var(--workspace-warning)]">· Hint used</span>}</span><span className={`flex items-center gap-1.5 font-medium ${loaded ? 'text-[var(--workspace-accent)]' : 'text-[var(--workspace-subtle)]'}`}>{loadingId === submission.id ? <><LoaderCircle className="h-3.5 w-3.5 animate-spin" />Loading…</> : !submission.has_code ? 'Code unavailable' : loaded ? <><Check className="h-3.5 w-3.5" />Loaded in editor</> : <>Load code<ArrowRight className="h-3.5 w-3.5" /></>}</span></div>
        </button>;
      })}</div>

      {error && <div className="rounded-lg border border-red-500/25 bg-red-500/10 p-4 text-sm text-[var(--workspace-danger)]" role="alert"><p>{error}</p>{!submissions.length && <button className="mt-3 flex items-center gap-2 rounded-md bg-[var(--workspace-hover)] px-3 py-2 text-[var(--workspace-text)] hover:bg-[var(--workspace-hover-strong)]" onClick={() => { setLoading(true); setError(''); setRetry(value => value + 1); }}><RefreshCw className="h-4 w-4" />Try again</button>}</div>}
      {nextId !== null && <button onClick={() => void loadMore()} disabled={loadingMore} className="flex w-full items-center justify-center gap-2 rounded-lg border border-[var(--workspace-border)] bg-[var(--workspace-surface)] py-3 text-sm text-[var(--workspace-subtle)] transition-colors hover:bg-[var(--workspace-control)] disabled:opacity-50">{loadingMore ? <><LoaderCircle className="h-4 w-4 animate-spin" />Loading…</> : <><History className="h-4 w-4" />Load older submissions</>}</button>}
    </>}
  </section>;
}
