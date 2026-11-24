'use client';

import { useSyncExternalStore } from 'react';
import { withPracticeLocation, type PracticeLocationUpdate } from './practiceNavigation';

const LOCATION_EVENT = 'torchcode-practice-location';

function subscribe(onChange: () => void) {
  window.addEventListener('popstate', onChange);
  window.addEventListener(LOCATION_EVENT, onChange);
  return () => {
    window.removeEventListener('popstate', onChange);
    window.removeEventListener(LOCATION_EVENT, onChange);
  };
}

export function usePracticeHref() {
  return useSyncExternalStore(subscribe, () => window.location.href, () => '');
}

export function practiceScrollTop() {
  return document.getElementById('practice-scroll')?.scrollTop || 0;
}

export function updatePracticeLocation(updates: PracticeLocationUpdate, push = false) {
  const next = withPracticeLocation(window.location.href, updates);
  if (next === window.location.href) return;
  if (push) window.history.pushState(window.history.state, '', next);
  else window.history.replaceState(window.history.state, '', next);
  window.dispatchEvent(new Event(LOCATION_EVENT));
}
