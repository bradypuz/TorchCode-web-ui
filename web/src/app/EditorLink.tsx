'use client';

import type { AnchorHTMLAttributes, SyntheticEvent } from 'react';
import { getEditorUrl, getPracticeReturnUrl, withPracticeLocation } from './practiceNavigation';
import { practiceScrollTop, usePracticeHref } from './usePracticeNavigation';

type Props = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> & {
  taskId?: string;
  user: string;
};

export default function EditorLink({ taskId, user, onClick, onAuxClick, onPointerDown, onContextMenu, onKeyDown, ...props }: Props) {
  const href = usePracticeHref();
  const returnTo = getPracticeReturnUrl(href || null, user);

  function rememberOrigin(event: SyntheticEvent<HTMLAnchorElement>) {
    // Capture the launch position on this link, keeping separate editor tabs
    // independent even when the dashboard is used again afterward.
    const origin = withPracticeLocation(window.location.href, { user, scrollTop: practiceScrollTop() });
    event.currentTarget.href = getEditorUrl(taskId, user, origin);
  }

  return <a {...props} href={getEditorUrl(taskId, user, returnTo)} target={props.target ?? '_blank'} rel={props.rel ?? 'noreferrer'}
    onPointerDown={event => { rememberOrigin(event); onPointerDown?.(event); }}
    onContextMenu={event => { rememberOrigin(event); onContextMenu?.(event); }}
    onKeyDown={event => { if (event.key === 'Enter') rememberOrigin(event); onKeyDown?.(event); }}
    onClick={event => { rememberOrigin(event); onClick?.(event); }}
    onAuxClick={event => { rememberOrigin(event); onAuxClick?.(event); }}
  />;
}
