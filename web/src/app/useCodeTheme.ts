'use client';

import { useMemo, useSyncExternalStore } from 'react';

export type CodeTheme = 'light' | 'dark';

type Snapshot = {
  theme: CodeTheme;
  loading: boolean;
  saving: boolean;
  error: string | null;
};

const DEFAULT_SNAPSHOT: Snapshot = { theme: 'dark', loading: false, saving: false, error: null };
const API = 'http://localhost:8000/api/practice/appearance';
const POLL_INTERVAL = 2500;

function isTheme(value: unknown): value is CodeTheme {
  return value === 'light' || value === 'dark';
}

function cacheKey(userId: string) {
  return `torchcode:code-theme:${encodeURIComponent(userId)}`;
}

function cachedTheme(userId: string): CodeTheme {
  try {
    const cached = window.localStorage.getItem(cacheKey(userId));
    return isTheme(cached) ? cached : 'dark';
  } catch {
    return 'dark';
  }
}

function cacheTheme(userId: string, theme: CodeTheme) {
  try {
    window.localStorage.setItem(cacheKey(userId), theme);
  } catch {
    // The backend remains the source of truth when browser storage is unavailable.
  }
}

async function responseTheme(response: Response): Promise<CodeTheme> {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body?.detail === 'string' ? body.detail : 'Appearance settings are unavailable.');
  }
  if (!isTheme(body?.theme)) throw new Error('The appearance setting could not be read.');
  return body.theme;
}

export function createCodeThemeStore(userId: string) {
  let snapshot: Snapshot = { ...DEFAULT_SNAPSHOT, theme: cachedTheme(userId), loading: true };
  let confirmedTheme = snapshot.theme;
  let saveError: string | null = null;
  let mutationVersion = 0;
  let pendingSaves = 0;
  let saveQueue = Promise.resolve();
  let readVersion = 0;
  let readController: AbortController | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const listeners = new Set<() => void>();

  function publish(patch: Partial<Snapshot>) {
    const next = { ...snapshot, ...patch };
    if (next.theme === snapshot.theme && next.loading === snapshot.loading && next.saving === snapshot.saving && next.error === snapshot.error) return;
    snapshot = next;
    listeners.forEach(listener => listener());
  }

  function clearTimer() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  }

  function cancelRead() {
    readVersion += 1;
    readController?.abort();
    readController = null;
  }

  function scheduleRefresh() {
    clearTimer();
    if (listeners.size && document.visibilityState !== 'hidden') {
      timer = setTimeout(() => { void refresh(); }, POLL_INTERVAL);
    }
  }

  async function refresh() {
    clearTimer();
    if (!listeners.size || document.visibilityState === 'hidden' || pendingSaves) return;
    cancelRead();
    const requestVersion = readVersion;
    const controller = new AbortController();
    readController = controller;
    try {
      const response = await fetch(`${API}?${new URLSearchParams({ user_id: userId })}`, {
        signal: controller.signal,
        cache: 'no-store',
      });
      const theme = await responseTheme(response);
      if (controller.signal.aborted || requestVersion !== readVersion || pendingSaves) return;
      confirmedTheme = theme;
      cacheTheme(userId, theme);
      publish({ theme, loading: false, error: saveError });
    } catch (error) {
      if (controller.signal.aborted || requestVersion !== readVersion || pendingSaves) return;
      publish({ loading: false, error: saveError ?? (error instanceof Error ? error.message : 'Appearance settings are unavailable.') });
    } finally {
      if (requestVersion === readVersion) {
        readController = null;
        scheduleRefresh();
      }
    }
  }

  function onVisibilityChange() {
    if (document.visibilityState === 'hidden') {
      clearTimer();
      cancelRead();
    } else {
      void refresh();
    }
  }

  function onFocus() {
    void refresh();
  }

  async function setTheme(theme: CodeTheme) {
    if (!isTheme(theme)) return;
    const version = ++mutationVersion;
    pendingSaves += 1;
    saveError = null;
    clearTimer();
    cancelRead();
    publish({ theme, loading: false, saving: true, error: null });

    // Serialize writes so rapid changes cannot reach the backend out of order.
    const save = saveQueue.then(async () => {
      try {
        const response = await fetch(API, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, theme }),
        });
        confirmedTheme = await responseTheme(response);
        cacheTheme(userId, confirmedTheme);
        if (version === mutationVersion) publish({ theme: confirmedTheme, error: null });
      } catch (error) {
        if (version === mutationVersion) {
          saveError = error instanceof Error ? error.message : 'We couldn’t save your appearance setting.';
          publish({ theme: confirmedTheme, error: saveError });
        }
      } finally {
        pendingSaves -= 1;
        publish({ saving: pendingSaves > 0 });
        if (!pendingSaves) scheduleRefresh();
      }
    });
    saveQueue = save;
    await save;
  }

  return {
    getSnapshot: () => snapshot,
    setTheme,
    subscribe(listener: () => void) {
      listeners.add(listener);
      if (listeners.size === 1) {
        document.addEventListener('visibilitychange', onVisibilityChange);
        window.addEventListener('focus', onFocus);
        void refresh();
      }
      return () => {
        listeners.delete(listener);
        if (!listeners.size) {
          clearTimer();
          cancelRead();
          document.removeEventListener('visibilitychange', onVisibilityChange);
          window.removeEventListener('focus', onFocus);
        }
      };
    },
  };
}

const stores = new Map<string, ReturnType<typeof createCodeThemeStore>>();
const emptyStore = {
  getSnapshot: () => DEFAULT_SNAPSHOT,
  setTheme: async () => {},
  subscribe: () => () => {},
};

export function useCodeTheme(userId: string | null): Snapshot & { setTheme: (theme: CodeTheme) => Promise<void> } {
  const store = useMemo(() => {
    if (!userId || typeof window === 'undefined') return emptyStore;
    let current = stores.get(userId);
    if (!current) {
      current = createCodeThemeStore(userId);
      stores.set(userId, current);
    }
    return current;
  }, [userId]);
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, () => DEFAULT_SNAPSHOT);
  return { ...snapshot, setTheme: store.setTheme };
}
