'use client';

import { useEffect, useRef, useState } from 'react';
import { Moon, Sun, SunMoon, X } from 'lucide-react';
import { useCodeTheme } from '../useCodeTheme';
import s from './appearance.module.css';

export default function AppearanceSettings({ user, name }: { user: string | null; name: string }) {
  const { theme, setTheme, loading, saving, error } = useCodeTheme(user);
  const [open, setOpen] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    if (open) dialog.current?.showModal();
    else dialog.current?.close();
  }, [open]);

  return <>
    <button className={s.trigger} onClick={() => setOpen(true)} aria-haspopup="dialog" aria-label="Appearance settings" disabled={!user}>
      <SunMoon size={18} /><span>Appearance</span>
    </button>
    <dialog ref={dialog} className={s.dialog} aria-labelledby="appearance-title" onCancel={() => setOpen(false)} onClick={event => {
      if (event.target !== event.currentTarget) return;
      const rect = event.currentTarget.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) setOpen(false);
    }}>
      <div className={s.heading}><span className={s.icon}><SunMoon size={23} /></span><button onClick={() => setOpen(false)} aria-label="Close appearance settings"><X size={21} /></button></div>
      <h2 id="appearance-title">Make yourself comfortable.</h2>
      <p>Choose the appearance of your coding workspace and solution guides.</p>
      <fieldset className={s.choices} disabled={loading || saving || !user}>
        <legend>Code &amp; solution appearance</legend>
        {(['light', 'dark'] as const).map(value => <label key={value} className={`${s.choice} ${theme === value ? s.selected : ''}`}>
          <input type="radio" name="code-appearance" value={value} checked={theme === value} onChange={() => void setTheme(value)} />
          <span className={s.preview} data-preview={value} aria-hidden="true"><span /><span /><span /></span>
          <span className={s.choiceName}>{value === 'light' ? <Sun size={17} /> : <Moon size={17} />}{value === 'light' ? 'Light' : 'Dark'}</span>
        </label>)}
      </fieldset>
      <div className={s.status} aria-live="polite">{error ? <p className={s.error} role="alert">{error}</p> : <p>{loading ? 'Loading your preference…' : saving ? 'Saving your preference…' : `Saved for ${name}. Applies to open workspaces too.`}</p>}</div>
    </dialog>
  </>;
}
