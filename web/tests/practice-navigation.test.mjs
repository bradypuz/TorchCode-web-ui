import assert from 'node:assert/strict';
import test from 'node:test';
import {
  getEditorUrl,
  getPracticeReturnUrl,
  parsePracticeLocation,
  withPracticeLocation,
} from '../src/app/practiceNavigation.ts';

const dashboard = 'http://localhost:4000/practice';
const emptyLocation = { section: 'today', query: '', filter: 'all', week: null, lesson: null, scrollTop: 0 };

test('all four dashboard views round-trip their applicable state', () => {
  for (const section of ['today', 'review', 'history', 'library']) {
    const location = {
      section,
      query: section === 'library' || section === 'history' ? 'GRPO & policy' : '',
      filter: section === 'library' ? 'hard' : section === 'history' ? 'working' : 'all',
      week: section === 'history' ? 4 : null,
      lesson: section === 'history' ? 'grpo_complete_loss' : null,
      scrollTop: 1642.6,
    };
    const url = withPracticeLocation(dashboard, location);
    assert.deepEqual(parsePracticeLocation(url), { ...location, scrollTop: 1643 });
  }
});

test('missing or malformed location fields use safe defaults', () => {
  for (const href of [dashboard, 'not a URL']) assert.deepEqual(parsePracticeLocation(href), emptyLocation);
  assert.deepEqual(parsePracticeLocation(`${dashboard}?section=elsewhere&filter=hard&week=5&lesson=bad%2Fid&scroll=Infinity`), emptyLocation);
  for (const value of ['0', '5', '-1', '1.5', 'NaN', '1e0']) {
    assert.equal(parsePracticeLocation(`${dashboard}?week=${value}`).week, null);
  }
  for (const value of ['-1', 'Infinity', 'NaN', '10000001']) {
    assert.equal(parsePracticeLocation(`${dashboard}?scroll=${value}`).scrollTop, 0);
  }
  assert.equal(parsePracticeLocation(`${dashboard}?scroll=12.5`).scrollTop, 12.5);
  assert.equal(parsePracticeLocation(`${dashboard}?scroll=10000000`).scrollTop, 10_000_000);
});

test('filters are interpreted only in the dashboard section that supports them', () => {
  assert.equal(parsePracticeLocation(`${dashboard}?section=history&filter=solved`).filter, 'solved');
  assert.equal(parsePracticeLocation(`${dashboard}?section=history&filter=hard`).filter, 'all');
  assert.equal(parsePracticeLocation(`${dashboard}?section=library&filter=hard`).filter, 'hard');
  assert.equal(parsePracticeLocation(`${dashboard}?section=library&filter=solved`).filter, 'all');
  assert.equal(parsePracticeLocation(`${dashboard}?section=review&filter=working`).filter, 'all');
  assert.equal(parsePracticeLocation(`${dashboard}?filter=easy`).filter, 'all');
});

test('partial updates preserve the path, unrelated query parameters, and hash', () => {
  const before = `${dashboard}/?user=other&section=history&q=old&filter=solved&week=4&lesson=grpo_loss&scroll=300&extra=keep#problem-submissions-grpo_loss`;
  const after = new URL(withPracticeLocation(before, { query: 'new search', scrollTop: 454.6 }));
  assert.equal(after.origin, 'http://localhost:4000');
  assert.equal(after.pathname, '/practice/');
  assert.equal(after.hash, '#problem-submissions-grpo_loss');
  assert.equal(after.searchParams.get('extra'), 'keep');
  assert.equal(after.searchParams.get('user'), 'other');
  assert.deepEqual(parsePracticeLocation(after.toString()), {
    section: 'history', query: 'new search', filter: 'solved', week: 4, lesson: 'grpo_loss', scrollTop: 455,
  });
});

test('explicit empty values remove stale view parameters without erasing other state', () => {
  const before = `${dashboard}?user=local&section=history&q=old&filter=solved&week=4&lesson=grpo_loss&scroll=300`;
  const after = new URL(withPracticeLocation(before, { query: '', filter: 'all', week: null, lesson: null, scrollTop: 0 }));
  for (const key of ['q', 'filter', 'week', 'lesson', 'scroll']) assert.equal(after.searchParams.has(key), false);
  assert.equal(after.searchParams.get('user'), 'local');
  assert.equal(after.searchParams.get('section'), 'history');
  assert.deepEqual(parsePracticeLocation(after.toString()), { ...emptyLocation, section: 'history' });
});

test('an editor URL carries its own encoded dashboard destination and profile', () => {
  const user = 'learner + & / 中文';
  const returnTo = withPracticeLocation(`${dashboard}?unrelated=%252F#details`, {
    section: 'library', query: 'GRPO + KL & 20% / 中文', filter: 'hard', user,
  });
  const editor = new URL(getEditorUrl('grpo_complete_loss', user, returnTo));
  assert.equal(editor.origin, 'http://localhost:3000');
  assert.equal(editor.searchParams.get('task'), 'grpo_complete_loss');
  assert.equal(editor.searchParams.get('user'), user);
  const restored = new URL(getPracticeReturnUrl(editor.searchParams.get('returnTo'), user));
  assert.equal(restored.searchParams.get('user'), user);
  assert.equal(restored.searchParams.get('q'), 'GRPO + KL & 20% / 中文');
  assert.equal(restored.searchParams.get('unrelated'), '%2F');
  assert.equal(restored.hash, '#details');
  assert.equal(parsePracticeLocation(restored.toString()).filter, 'hard');
});

test('independent editor tabs retain distinct launch destinations when tasks change', () => {
  const first = new URL(getEditorUrl('grpo_loss', 'local', withPracticeLocation(dashboard, {
    section: 'history', week: 4, lesson: 'grpo_loss', scrollTop: 750,
  })));
  const second = new URL(getEditorUrl('relu', 'local', withPracticeLocation(dashboard, { section: 'review', scrollTop: 200 })));
  first.searchParams.set('task', 'ppo_loss');
  assert.deepEqual(parsePracticeLocation(first.searchParams.get('returnTo')), {
    ...emptyLocation, section: 'history', week: 4, lesson: 'grpo_loss', scrollTop: 750,
  });
  assert.deepEqual(parsePracticeLocation(second.searchParams.get('returnTo')), {
    ...emptyLocation, section: 'review', scrollTop: 200,
  });
});

test('only absolute URLs on supported dashboard paths can be return destinations', () => {
  const user = 'name + &';
  const fallback = `${dashboard}?${new URLSearchParams({ user })}`;
  const invalid = [
    undefined, null, '', 'not a URL', '/practice?section=history', '//localhost:4000/practice',
    'javascript:alert(1)', 'https://localhost:4000/practice', 'http://localhost:3000/practice',
    'http://localhost:4001/practice', 'http://127.0.0.1:4000/practice', 'http://example.com/practice',
    'http://localhost.example.com:4000/practice', 'http://localhost:4000/elsewhere',
    'http://localhost:4000/practice/extra', 'http://localhost:4000/%70ractice',
    'http://someone@localhost:4000/practice', 'http://someone:secret@localhost:4000/practice',
    encodeURIComponent(`${dashboard}?section=review`),
  ];
  for (const raw of invalid) assert.equal(getPracticeReturnUrl(raw, user), fallback, String(raw));
  for (const path of ['/', '/practice', '/practice/']) {
    const restored = new URL(getPracticeReturnUrl(`http://localhost:4000${path}?section=review&user=other#end`, user));
    assert.equal(restored.pathname, path);
    assert.equal(restored.searchParams.get('section'), 'review');
    assert.equal(restored.searchParams.get('user'), user);
    assert.equal(restored.hash, '#end');
  }
});

test('the generic editor link omits an absent task and validates its destination', () => {
  const editor = new URL(getEditorUrl(undefined, 'local', 'http://example.com/'));
  assert.equal(editor.searchParams.has('task'), false);
  assert.equal(editor.searchParams.get('user'), 'local');
  assert.equal(editor.searchParams.get('returnTo'), `${dashboard}?user=local`);
});
