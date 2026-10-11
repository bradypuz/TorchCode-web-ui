import assert from 'node:assert/strict';
import test from 'node:test';
import { lessonState, currentJourneyWeeks, isManuallyFlagged } from '../src/app/practice/journeyProgress.ts';

const solved = { id: 'relu', status: 'solved', needs_review: false, attempts: 3 };
const history = { task_id: 'relu', solved: true, needs_review: false, attempts: 3 };

test('a review flag replaces the solved state without changing historical success', () => {
  assert.equal(lessonState({ ...solved, needs_review: true }, history), 'Needs more practice');
  assert.equal(history.solved, true);
  assert.equal(lessonState(undefined, { ...history, needs_review: true }), 'Needs more practice');
});

test('clearing a flag restores a solved lesson even when earlier history was flagged', () => {
  assert.equal(lessonState(solved, { ...history, needs_review: true }), 'Solved');
  assert.equal(lessonState(undefined, history), 'Solved');
});

test('weekly completion drops when a previously solved problem needs more practice', () => {
  const weeks = [{ week: 1, task_ids: ['relu', 'softmax'], solved: 2, total: 2 }];
  const progress = [solved, { ...solved, id: 'softmax', needs_review: true }];
  const result = currentJourneyWeeks(weeks, progress, [history]);
  assert.equal(result[0].solved, 1);
  assert.equal(result[0].total, 2);
  assert.equal(weeks[0].solved, 2);
});

test('unstarted and unsuccessful lessons remain distinct', () => {
  assert.equal(lessonState(), 'Not started');
  assert.equal(lessonState({ ...solved, status: 'attempted' }), 'In progress');
  assert.equal(lessonState(undefined, { ...history, solved: false }), 'In progress');
});

test('automatic review recommendations do not act like a manual bookmark', () => {
  const automatic = { ...solved, needs_review: true, manual_review: false };
  assert.equal(lessonState(automatic, history), 'Needs more practice');
  assert.equal(isManuallyFlagged(automatic, { needs_review: true, manual_review: true }), false);
  assert.equal(isManuallyFlagged({ ...automatic, manual_review: true }), true);
  assert.equal(isManuallyFlagged({ needs_review: true }), true);
});
