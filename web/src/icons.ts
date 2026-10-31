/** Line icons (24 × 24, stroke = currentColor), in the spirit of dcv/icons.py: no emoji in toolbars. */
const svg = (body: string) =>
  `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const ICONS: Record<string, string> = {
  open: svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 10v6M9 13l3-3 3 3"/>'),
  series: svg('<rect x="3" y="4" width="7" height="7" rx="1"/><rect x="3" y="13" width="7" height="7" rx="1"/><path d="M13 6h8M13 9h5M13 15h8M13 18h5"/>'),
  wl: svg('<circle cx="12" cy="12" r="8"/><path d="M12 4v16" /><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/>'),
  pan: svg('<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>'),
  zoom: svg('<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5 21 21M8 10.5h5M10.5 8v5"/>'),
  scroll: svg('<rect x="6" y="3" width="12" height="18" rx="2"/><path d="M12 7v10M9.5 9.5 12 7l2.5 2.5M9.5 14.5 12 17l2.5-2.5"/>'),
  length: svg('<path d="M4 20 20 4"/><path d="M7 17l-2-2M10 14l-1.5-1.5M13 11l-2-2M16 8l-1.5-1.5"/><circle cx="4" cy="20" r="1.4"/><circle cx="20" cy="4" r="1.4"/>'),
  presets: svg('<path d="M4 6h16M4 12h10M4 18h7"/><circle cx="18" cy="15" r="3"/>'),
  dsa: svg('<rect x="3" y="5" width="8" height="14" rx="1"/><rect x="13" y="5" width="8" height="14" rx="1"/><path d="M15 9c2 2 2 4 0 6M7 8v8"/>'),
  mpr: svg('<rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><path d="M13 17h8M17 13v8"/>'),
  vr: svg('<path d="M12 3 20 7.5v9L12 21 4 16.5v-9z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/>'),
  cine: svg('<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9.5v5l4.5-2.5z" fill="currentColor"/>'),
  fit: svg('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/><rect x="8" y="8" width="8" height="8" rx="1"/>'),
  layout: svg('<rect x="3" y="4" width="18" height="16" rx="1.5"/><path d="M9 4v16M15 4v16"/>'),
  locate: svg('<circle cx="12" cy="12" r="6"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>'),
  collage: svg('<rect x="3" y="3" width="18" height="18" rx="1.5"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>'),
  seed: svg('<circle cx="12" cy="12" r="2.2" fill="currentColor"/><path d="M12 4.5c4.5 0 7.5 3 7.5 7.5s-3 7.5-7.5 7.5-7.5-3-7.5-7.5" stroke-dasharray="3 2.4"/>'),
  info: svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>'),
  play: svg('<path d="M7 5v14l12-7z" fill="currentColor"/>'),
  pause: svg('<path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z" fill="currentColor"/>'),
  trash: svg('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>'),
  invert: svg('<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 0 0 16z" fill="currentColor"/>'),
};
