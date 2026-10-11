export type LessonState = 'Solved' | 'Needs more practice' | 'In progress' | 'Not started';
type Progress = { id: string; status: string; needs_review: boolean; attempts: number };
type History = { task_id: string; solved: boolean; needs_review: boolean; attempts: number };
type Week = { task_ids: string[]; solved: number; total: number };
type ReviewFlags = { needs_review: boolean; manual_review?: boolean };

export function isManuallyFlagged(progress?: ReviewFlags, history?: ReviewFlags): boolean {
  return progress?.manual_review ?? history?.manual_review ?? progress?.needs_review ?? history?.needs_review ?? false;
}

export function lessonState(progress?: Progress, history?: History): LessonState {
  // Current progress is authoritative when a learner clears an earlier flag.
  if (progress?.needs_review ?? history?.needs_review ?? false) return 'Needs more practice';
  if (progress?.status === 'solved' || history?.solved) return 'Solved';
  if (progress?.status === 'attempted' || (progress?.attempts ?? 0) > 0 || (history?.attempts ?? 0) > 0) return 'In progress';
  return 'Not started';
}

export function currentJourneyWeeks<T extends Week>(weeks: T[], progress: Progress[], history: History[]): T[] {
  const progressById = new Map(progress.map(item => [item.id, item]));
  const historyById = new Map(history.map(item => [item.task_id, item]));
  return weeks.map(week => ({ ...week, solved: week.task_ids.filter(id => lessonState(progressById.get(id), historyById.get(id)) === 'Solved').length }));
}
