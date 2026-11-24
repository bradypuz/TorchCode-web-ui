export type PracticeSection = 'today' | 'review' | 'history' | 'library';

export type PracticeLocation = {
  section: PracticeSection;
  query: string;
  filter: string;
  week: number | null;
  lesson: string | null;
  scrollTop: number;
};

export type PracticeLocationUpdate = Partial<PracticeLocation> & { user?: string };

const PRACTICE_ORIGIN = 'http://localhost:4000';
const MAX_SCROLL_TOP = 10_000_000;
const LESSON_ID = /^[a-z0-9_]+$/;

function practiceSection(value: string | null): PracticeSection {
  return value === 'review' || value === 'history' || value === 'library' ? value : 'today';
}

function practiceFilter(section: PracticeSection, value: string | null): string {
  if (section === 'history' && (value === 'solved' || value === 'working')) return value;
  if (section === 'library' && (value === 'easy' || value === 'medium' || value === 'hard')) return value;
  return 'all';
}

function validScrollTop(value: number): number {
  return Number.isFinite(value) && value >= 0 && value <= MAX_SCROLL_TOP ? value : 0;
}

export function parsePracticeLocation(href: string): PracticeLocation {
  let params: URLSearchParams;
  try {
    params = new URL(href).searchParams;
  } catch {
    params = new URLSearchParams();
  }
  const section = practiceSection(params.get('section'));
  const week = params.get('week');
  const lesson = params.get('lesson');
  return {
    section,
    query: params.get('q') || '',
    filter: practiceFilter(section, params.get('filter')),
    week: week !== null && /^[1-4]$/.test(week) ? Number(week) : null,
    lesson: lesson && LESSON_ID.test(lesson) ? lesson : null,
    scrollTop: validScrollTop(Number(params.get('scroll') || 0)),
  };
}

export function withPracticeLocation(href: string, updates: PracticeLocationUpdate): string {
  const url = new URL(href);
  const params = url.searchParams;
  if (updates.section !== undefined) params.set('section', practiceSection(updates.section));
  if (updates.query !== undefined) {
    if (updates.query) params.set('q', updates.query);
    else params.delete('q');
  }
  if (updates.filter !== undefined) {
    const filter = practiceFilter(practiceSection(params.get('section')), updates.filter);
    if (filter !== 'all') params.set('filter', filter);
    else params.delete('filter');
  }
  if (updates.week !== undefined) {
    if (updates.week !== null && Number.isInteger(updates.week) && updates.week >= 1 && updates.week <= 4) {
      params.set('week', String(updates.week));
    } else params.delete('week');
  }
  if (updates.lesson !== undefined) {
    if (updates.lesson && LESSON_ID.test(updates.lesson)) params.set('lesson', updates.lesson);
    else params.delete('lesson');
  }
  if (updates.scrollTop !== undefined) {
    const scroll = Math.round(validScrollTop(updates.scrollTop));
    if (scroll) params.set('scroll', String(scroll));
    else params.delete('scroll');
  }
  if (updates.user !== undefined) params.set('user', updates.user);
  return url.toString();
}

export function getPracticeReturnUrl(raw: string | null | undefined, user: string): string {
  const fallback = new URL('/practice', PRACTICE_ORIGIN);
  fallback.searchParams.set('user', user);
  if (!raw) return fallback.toString();
  try {
    const url = new URL(raw);
    if (url.origin !== PRACTICE_ORIGIN || url.username || url.password
      || !['/', '/practice', '/practice/'].includes(url.pathname)) return fallback.toString();
    url.searchParams.set('user', user);
    return url.toString();
  } catch {
    return fallback.toString();
  }
}

export function getEditorUrl(taskId: string | undefined, user: string, returnTo: string): string {
  const url = new URL('http://localhost:3000/');
  if (taskId !== undefined) url.searchParams.set('task', taskId);
  url.searchParams.set('user', user);
  url.searchParams.set('returnTo', getPracticeReturnUrl(returnTo, user));
  return url.toString();
}
