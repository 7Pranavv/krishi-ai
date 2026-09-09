/* Shared UI pieces: page shell, theme, form helpers, state rendering.
 * Every page imports this so the header, footer and states stay identical. */

import { LANGUAGES, applyTranslations, currentLanguage, initLanguage, setLanguage, t, tc } from './i18n.js';
import { api, auth } from './api.js';

/* Every tool. `key` is the short nav label; title/blurb keys are the longer
 * forms used on the home page cards. All resolved through t() at render time
 * so a language change re-labels them without a reload. */
export const TOOLS = [
  { id: 'chat',       href: '/chat',       icon: '💬', key: 'nav_chat' },
  { id: 'crop',       href: '/crop',       icon: '🌾', key: 'nav_crop' },
  { id: 'disease',    href: '/disease',    icon: '🍃', key: 'nav_disease' },
  { id: 'fertilizer', href: '/fertilizer', icon: '🧪', key: 'nav_fertilizer' },
  { id: 'water',      href: '/water',      icon: '💧', key: 'nav_water' },
  { id: 'rainfall',   href: '/rainfall',   icon: '🌧️', key: 'nav_rainfall' },
  { id: 'market',     href: '/market',     icon: '🧺', key: 'nav_market' },
].map((tool) => ({
  ...tool,
  get title() { return t(`tool_${this.id}_title`); },
  get blurb() { return t(`tool_${this.id}_blurb`); },
}));

/* --------------------------------------------------------------- theme */
const THEME_KEY = 'krishi.theme';

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch { /* storage unavailable */ }
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.textContent = theme === 'dark' ? '☀️' : '🌙';
    btn.setAttribute('aria-label', t(theme === 'dark' ? 'theme_light' : 'theme_dark'));
  }
}

function initTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem(THEME_KEY);
  } catch { /* storage unavailable */ }
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
}

/* --------------------------------------------------------------- shell */
export async function renderShell(activeId) {
  // Load the catalogue first so the header paints in the chosen language
  // rather than flashing English.
  await initLanguage();

  const header = document.createElement('header');
  header.className = 'site-header';
  header.innerHTML = `
    <a class="brand" href="/"><span class="brand-mark" aria-hidden="true">🌱</span>Krishi.AI</a>
    <button class="icon-btn" id="nav-toggle" aria-expanded="false" aria-controls="site-nav"
            data-i18n-label="nav_toggle" aria-label="Toggle navigation">☰</button>
    <nav class="site-nav" id="site-nav" aria-label="Main">
      <a href="/" data-i18n="nav_home">Home</a>
      ${TOOLS.map((tool) => `<a href="${tool.href}" data-nav="${tool.id}" data-i18n="${tool.key}">${escapeHtml(t(tool.key))}</a>`).join('')}
      <a href="/profile" data-nav="profile" data-i18n="nav_profile">Profile</a>
    </nav>
    <span class="spacer"></span>
    <div class="header-tools">
      <label class="sr-only" for="lang-select" data-i18n="language">Language</label>
      <select id="lang-select" style="width:auto;padding:7px 10px">
        ${LANGUAGES.map((l) => `<option value="${l.code}">${l.native}</option>`).join('')}
      </select>
      <button class="icon-btn" id="theme-toggle" type="button">🌙</button>
    </div>`;
  document.body.prepend(header);

  const footer = document.createElement('footer');
  footer.className = 'site-footer';
  footer.innerHTML =
    `<span data-i18n="footer_tagline">${escapeHtml(t('footer_tagline'))}</span> ` +
    `<span data-i18n="footer_disclaimer">${escapeHtml(t('footer_disclaimer'))}</span>`;
  document.body.append(footer);

  const nav = header.querySelector('#site-nav');
  const toggle = header.querySelector('#nav-toggle');
  const isMobile = () => window.matchMedia('(max-width: 860px)').matches;
  const closeNav = () => {
    if (isMobile()) {
      nav.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
    }
  };
  toggle.addEventListener('click', () => {
    nav.hidden = !nav.hidden;
    toggle.setAttribute('aria-expanded', String(!nav.hidden));
  });
  window.addEventListener('resize', () => {
    nav.hidden = isMobile() ? nav.hidden : false;
  });
  nav.addEventListener('click', closeNav);
  closeNav();

  const active = activeId === 'home'
    ? nav.querySelector('a[href="/"]')
    : nav.querySelector(`[data-nav="${activeId}"]`);
  active?.setAttribute('aria-current', 'page');

  initTheme();
  header.querySelector('#theme-toggle').addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });

  const select = header.querySelector('#lang-select');
  select.value = currentLanguage();
  select.addEventListener('change', async (event) => {
    const code = event.target.value;
    await setLanguage(code);
    // The assistant replies in the language stored on the profile, so the
    // choice has to reach the server too. A preference is not worth blocking
    // or reporting on if the write fails.
    if (auth.token) {
      api.updateProfile({ language: code })
        .then((user) => auth.setUser(user))
        .catch(() => {});
    }
  });
  applyTranslations();
}

/* --------------------------------------------------------------- states */
export function stateMarkup({ icon = 'ℹ️', title = '', body = '' }) {
  return `<div class="state">
    <div class="state-icon" aria-hidden="true">${icon}</div>
    ${title ? `<h3>${escapeHtml(title)}</h3>` : ''}
    ${body ? `<p>${escapeHtml(body)}</p>` : ''}
  </div>`;
}

export function showEmpty(target, title, body, icon = '🌱') {
  target.innerHTML = stateMarkup({ icon, title, body });
}

/* Renders any error, with a distinct look for "temporarily unavailable"
 * so a farmer knows to retry rather than to change their input. */
export function showError(target, error) {
  const unavailable = error?.isUnavailable;
  const fallback = t('common_error');
  target.innerHTML = `<div class="alert ${unavailable ? 'alert-warning' : 'alert-error'}" role="alert">
    <span aria-hidden="true">${unavailable ? '⏳' : '⚠️'}</span>
    <div>
      <strong>${escapeHtml(error?.message || fallback)}</strong>
      ${error?.detail ? `<span>${escapeHtml(error.detail)}</span>` : ''}
    </div>
  </div>`;
}

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

/* Model replies come back as Markdown. Rather than pull in a parser, escape
 * everything first and then re-introduce only what actually shows up in
 * answers: headings, bold, italics and bullet markers. Escaping first means no
 * model output can inject markup. Line breaks are handled by white-space:
 * pre-wrap in the CSS.
 *
 * Order matters: headings before bullets (### vs *), and bold before italics,
 * so ** is consumed before single * is considered. */
export function formatReply(text) {
  return escapeHtml(text)
    .replace(/^\s{0,3}#{1,6}\s+(.+?)\s*$/gm, '<strong>$1</strong>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    // The \S guards stop "2 * 3 * 4" from turning into italics.
    .replace(/(^|[\s(])\*(\S(?:[^*\n]*\S)?)\*(?=[\s.,;:)!?]|$)/gm, '$1<em>$2</em>')
    .replace(/^\s{0,3}[*-]\s+/gm, '• ');
}

/* -------------------------------------------------------------- forms */

/* Wires a range input to its <output> so the current value is always visible. */
export function bindRange(input) {
  const output = input.parentElement.querySelector('output');
  if (!output) return;
  const suffix = input.dataset.suffix || '';
  const sync = () => { output.textContent = `${input.value}${suffix}`; };
  input.addEventListener('input', sync);
  sync();
}

export function initRanges(root = document) {
  root.querySelectorAll('input[type="range"]').forEach(bindRange);
}

/* Fills a <select> from an /options response ({values, labels}). */
export function fillSelect(select, options, preferred) {
  select.innerHTML = options.values
    .map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(options.labels?.[v] ?? v)}</option>`)
    .join('');
  if (preferred && options.values.includes(preferred)) select.value = preferred;
  select.disabled = false;
}

/* Reads a form into a plain object, coercing numeric inputs.
 * Returns null and marks the fields when the browser reports them invalid. */
export function readForm(form) {
  if (!form.reportValidity()) return null;
  const data = {};
  for (const el of form.elements) {
    if (!el.name || el.disabled) continue;
    if (el.type === 'checkbox') data[el.name] = el.checked;
    else if (el.type === 'number' || el.type === 'range') data[el.name] = Number(el.value);
    else data[el.name] = el.value;
  }
  return data;
}

/* Prevents duplicate submissions: the button is disabled and shows a spinner
 * for as long as the request is in flight, whatever its outcome. */
export function withBusy(button, label, task) {
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<span class="spinner"></span>${escapeHtml(label)}`;
  return Promise.resolve()
    .then(task)
    .finally(() => {
      button.disabled = false;
      button.innerHTML = original;
    });
}

export function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(undefined, {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
  });
}

export { t, tc };
