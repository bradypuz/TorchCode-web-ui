import assert from 'node:assert/strict';
import test from 'node:test';
import { createCodeThemeStore } from '../src/app/useCodeTheme.ts';

function browserEnvironment(t, { storageAvailable = true } = {}) {
  const requests = [];
  const timers = new Map();
  const storage = new Map();
  const cleanups = [];
  let timerId = 0;
  const document = Object.assign(new EventTarget(), { visibilityState: 'visible' });
  const window = Object.assign(new EventTarget(), {
    localStorage: {
      getItem(key) {
        if (!storageAvailable) throw new Error('Storage is blocked');
        return storage.get(key) ?? null;
      },
      setItem(key, value) {
        if (!storageAvailable) throw new Error('Storage is blocked');
        storage.set(key, value);
      },
    },
  });
  const globals = {
    window,
    document,
    fetch(url, options = {}) {
      return new Promise((resolve, reject) => {
        requests.push({
          url: String(url),
          method: options.method ?? 'GET',
          signal: options.signal,
          body: options.body && JSON.parse(options.body),
          respond(body, status = 200) { resolve(new Response(JSON.stringify(body), { status })); },
          reject,
        });
      });
    },
    setTimeout(callback, delay) {
      const id = ++timerId;
      timers.set(id, { callback, delay });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
  };
  const originals = new Map();
  for (const [key, value] of Object.entries(globals)) {
    originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  t.after(() => {
    for (const cleanup of cleanups.reverse()) cleanup();
    for (const [key, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  return {
    requests,
    timers,
    storage,
    watch(store, listener = () => {}) {
      const unsubscribe = store.subscribe(listener);
      cleanups.push(unsubscribe);
      return unsubscribe;
    },
    nextRequest(method) {
      const request = requests.shift();
      assert.ok(request, `Expected a ${method} request`);
      assert.equal(request.method, method);
      return request;
    },
    runTimer() {
      const [id, timer] = timers.entries().next().value ?? [];
      assert.ok(timer, 'Expected a scheduled refresh');
      assert.equal(timer.delay, 2500);
      timers.delete(id);
      timer.callback();
    },
    setVisibility(state) {
      document.visibilityState = state;
      document.dispatchEvent(new Event('visibilitychange'));
    },
    focus() { window.dispatchEvent(new Event('focus')); },
  };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test('a saved change updates every subscriber and an older GET cannot undo it', async t => {
  const env = browserEnvironment(t);
  const store = createCodeThemeStore('local');
  const changes = [[], []];
  env.watch(store, () => changes[0].push(store.getSnapshot().theme));
  env.watch(store, () => changes[1].push(store.getSnapshot().theme));
  env.nextRequest('GET').respond({ theme: 'dark' });
  await settle();

  env.runTimer();
  const staleRead = env.nextRequest('GET');
  const save = store.setTheme('light');
  assert.equal(store.getSnapshot().theme, 'light');
  assert.equal(store.getSnapshot().saving, true);
  assert.equal(staleRead.signal.aborted, true);
  await settle();
  const write = env.nextRequest('PUT');
  assert.deepEqual(write.body, { user_id: 'local', theme: 'light' });
  write.respond({ theme: 'light' });
  await save;
  staleRead.respond({ theme: 'dark' });
  await settle();

  assert.deepEqual(store.getSnapshot(), { theme: 'light', loading: false, saving: false, error: null });
  assert.deepEqual(changes[0], changes[1]);
  assert.ok(changes[0].includes('light'));
  assert.equal(env.storage.get('torchcode:code-theme:local'), 'light');
});

test('a failed save restores the last confirmed preference and keeps its error visible', async t => {
  const env = browserEnvironment(t);
  const store = createCodeThemeStore('local');
  env.watch(store);
  env.nextRequest('GET').respond({ theme: 'light' });
  await settle();
  const save = store.setTheme('dark');
  await settle();
  env.nextRequest('PUT').respond({ detail: 'Preferences unavailable' }, 503);
  await save;

  assert.deepEqual(store.getSnapshot(), {
    theme: 'light', loading: false, saving: false, error: 'Preferences unavailable',
  });
  assert.equal(env.storage.get('torchcode:code-theme:local'), 'light');
  env.runTimer();
  env.nextRequest('GET').respond({ theme: 'light' });
  await settle();
  assert.equal(store.getSnapshot().error, 'Preferences unavailable');
});

test('rapid writes are serialized and a later failure preserves the earlier successful save', async t => {
  const env = browserEnvironment(t);
  const store = createCodeThemeStore('local');
  env.watch(store);
  env.nextRequest('GET').respond({ theme: 'dark' });
  await settle();
  const firstSave = store.setTheme('light');
  const secondSave = store.setTheme('dark');
  await settle();
  const firstWrite = env.nextRequest('PUT');
  assert.equal(firstWrite.body.theme, 'light');
  assert.equal(env.requests.length, 0);
  firstWrite.respond({ theme: 'light' });
  await firstSave;
  assert.equal(store.getSnapshot().theme, 'dark');
  assert.equal(store.getSnapshot().saving, true);
  await settle();
  const secondWrite = env.nextRequest('PUT');
  assert.equal(secondWrite.body.theme, 'dark');
  secondWrite.reject(new Error('Connection lost'));
  await secondSave;
  assert.deepEqual(store.getSnapshot(), { theme: 'light', loading: false, saving: false, error: 'Connection lost' });
});

test('different profiles have isolated requests, cached values, and optimistic updates', async t => {
  const env = browserEnvironment(t);
  env.storage.set('torchcode:code-theme:alice', 'light');
  const alice = createCodeThemeStore('alice');
  const bob = createCodeThemeStore('bob');
  assert.equal(alice.getSnapshot().theme, 'light');
  assert.equal(bob.getSnapshot().theme, 'dark');
  const stopAlice = env.watch(alice);
  env.watch(bob);
  const aliceRead = env.nextRequest('GET');
  const bobRead = env.nextRequest('GET');
  assert.equal(new URL(aliceRead.url).searchParams.get('user_id'), 'alice');
  assert.equal(new URL(bobRead.url).searchParams.get('user_id'), 'bob');
  stopAlice();
  bobRead.respond({ theme: 'light' });
  aliceRead.respond({ theme: 'light' });
  await settle();
  assert.equal(bob.getSnapshot().theme, 'light');
  const save = alice.setTheme('dark');
  assert.equal(alice.getSnapshot().theme, 'dark');
  assert.equal(bob.getSnapshot().theme, 'light');
  assert.equal(bob.getSnapshot().saving, false);
  await settle();
  const write = env.nextRequest('PUT');
  assert.equal(write.body.user_id, 'alice');
  write.respond({ theme: 'dark' });
  await save;
  assert.equal(bob.getSnapshot().theme, 'light');
  assert.equal(env.storage.get('torchcode:code-theme:bob'), 'light');
});

test('hidden pages stop polling, visibility and focus refresh, and the last unsubscribe stops work', async t => {
  const env = browserEnvironment(t);
  const store = createCodeThemeStore('local');
  const unsubscribe = env.watch(store);
  env.nextRequest('GET').respond({ theme: 'dark' });
  await settle();
  assert.equal(env.timers.size, 1);
  env.setVisibility('hidden');
  assert.equal(env.timers.size, 0);
  env.focus();
  assert.equal(env.requests.length, 0);
  env.setVisibility('visible');
  env.nextRequest('GET').respond({ theme: 'light' });
  await settle();
  assert.equal(store.getSnapshot().theme, 'light');
  assert.equal(env.timers.size, 1);
  env.focus();
  const focusedRead = env.nextRequest('GET');
  unsubscribe();
  assert.equal(focusedRead.signal.aborted, true);
  focusedRead.respond({ theme: 'dark' });
  await settle();
  assert.equal(store.getSnapshot().theme, 'light');
  assert.equal(env.timers.size, 0);
  env.focus();
  assert.equal(env.requests.length, 0);
});

test('backend preferences load and save when browser storage is unavailable', async t => {
  const env = browserEnvironment(t, { storageAvailable: false });
  const store = createCodeThemeStore('local');
  assert.equal(store.getSnapshot().theme, 'dark');
  env.watch(store);
  env.nextRequest('GET').respond({ theme: 'light' });
  await settle();
  assert.equal(store.getSnapshot().theme, 'light');
  const save = store.setTheme('dark');
  await settle();
  env.nextRequest('PUT').respond({ theme: 'dark' });
  await save;
  assert.deepEqual(store.getSnapshot(), { theme: 'dark', loading: false, saving: false, error: null });
});
