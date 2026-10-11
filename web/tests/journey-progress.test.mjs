import assert from 'node:assert/strict';
import test from 'node:test';
import { lessonState, currentJourneyWeeks, isManuallyFlagged } from '../src/app/practice/journeyProgress.ts';

const solved = { id: 'relu', status: 'solved', needs_review: false, attempts: 3 };
const history = { task_id: 'relu', solved: true, needs_review: false, attempts: 3 };

test('practice flags keep an earned solved state', () => {
  assert.equal(lessonState({ ...solved, needs_review: true }, history), 'Solved');
  assert.equal(history.solved, true);
  assert.equal(lessonState(undefined, { ...history, needs_review: true }), 'Solved');
});

test('clearing a practice flag leaves completion intact even when earlier history was flagged', () => {
  assert.equal(lessonState(solved, { ...history, needs_review: true }), 'Solved');
  assert.equal(lessonState(undefined, history), 'Solved');
});

test('weekly completion includes solved problems with practice reminders', () => {
  const weeks = [{ week: 1, task_ids: ['relu', 'softmax'], solved: 2, total: 2 }];
  const progress = [solved, { ...solved, id: 'softmax', needs_review: true }];
  const result = currentJourneyWeeks(weeks, progress, [history]);
  assert.equal(result[0].solved, 2);
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
  assert.equal(lessonState(automatic, history), 'Solved');
  assert.equal(isManuallyFlagged(automatic, { needs_review: true, manual_review: true }), false);
  assert.equal(isManuallyFlagged({ ...automatic, manual_review: true }), true);
  assert.equal(isManuallyFlagged({ needs_review: true }), true);
});

test('later failed submissions preserve a previous successful completion', () => {
  const latestFailure = { ...history, success: false, passed: 2, total: 4, needs_review: true };
  assert.equal(lessonState({ ...solved, needs_review: true }, latestFailure), 'Solved');
  assert.equal(lessonState(undefined, latestFailure), 'Solved');
  const weeks = [{ task_ids: ['relu'], solved: 0, total: 1 }];
  assert.equal(currentJourneyWeeks(weeks, [], [latestFailure])[0].solved, 1);
});

test('never-solved practice flags do not complete a lesson', () => {
  const unfinished = { ...solved, status: 'attempted', needs_review: true };
  const failed = { ...history, solved: false, success: false, needs_review: true };
  assert.equal(lessonState(unfinished, failed), 'Needs more practice');
  assert.equal(currentJourneyWeeks([{ task_ids: ['relu'], solved: 0, total: 1 }], [unfinished], [failed])[0].solved, 0);
});

test('imported solves and a recorded successful submission count as completion', () => {
  assert.equal(lessonState({ ...solved, attempts: 0, needs_review: true }), 'Solved');
  assert.equal(lessonState(undefined, { ...history, solved: false, success: true, needs_review: true }), 'Solved');
});
