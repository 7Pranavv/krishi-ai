/* Interface language.
 *
 * Strings live in frontend/i18n/<code>.json, generated from en.json by
 * tools/translate_ui.py. English loads first and is the source of truth; other
 * languages load on demand and are cached for the session.
 *
 * Any key missing from a translation falls back to English rather than showing
 * a raw key name, so a partial catalogue degrades into a mixed-language page
 * instead of a broken one.
 *
 * The choice also travels to the user's profile, which is what the assistant
 * reads to decide which language to answer in.
 */

export const LANGUAGES = [
  { code: 'en', native: 'English',   english: 'English' },
  { code: 'hi', native: 'हिंदी',      english: 'Hindi' },
  { code: 'bn', native: 'বাংলা',      english: 'Bengali' },
  { code: 'mr', native: 'मराठी',      english: 'Marathi' },
  { code: 'te', native: 'తెలుగు',     english: 'Telugu' },
  { code: 'ta', native: 'தமிழ்',      english: 'Tamil' },
  { code: 'gu', native: 'ગુજરાતી',    english: 'Gujarati' },
  { code: 'kn', native: 'ಕನ್ನಡ',      english: 'Kannada' },
  { code: 'ml', native: 'മലയാളം',    english: 'Malayalam' },
  { code: 'pa', native: 'ਪੰਜਾਬੀ',     english: 'Punjabi' },
  { code: 'or', native: 'ଓଡ଼ିଆ',      english: 'Odia' },
  { code: 'as', native: 'অসমীয়া',    english: 'Assamese' },
  { code: 'ur', native: 'اردو',       english: 'Urdu', rtl: true },
];

const STORAGE_KEY = 'krishi.lang';

let english = {};
let active = {};
let current = 'en';
const cache = new Map();

export function currentLanguage() {
  return current;
}

export function isRtl(code = current) {
  return Boolean(LANGUAGES.find((l) => l.code === code)?.rtl);
}

/* Translate a key. Falls back to English, then to the key itself. */
export function t(key) {
  return active[key] ?? english[key] ?? key;
}

/* Translate with {placeholders} filled in, e.g. tf('water_day', {n: 3}). */
export function tf(key, values = {}) {
  return String(t(key)).replace(/\{(\w+)\}/g, (whole, name) =>
    (name in values ? String(values[name]) : whole));
}

/* ------------------------------------------------- agronomic content
 * The ML service returns advice as English prose (crop tips, irrigation
 * notes, fertilizer schedules, rainfall advisories). Those sentences are the
 * part a farmer acts on, so they are translated too - looked up by their
 * English text rather than by a key, because that is what the API sends.
 *
 * Unknown text passes through unchanged, so a new tip added server-side shows
 * in English until the catalogue is regenerated, never as a blank or an error.
 */
let contentMap = {};

export function tc(text) {
  if (!text) return text;
  return contentMap[String(text).trim()] ?? text;
}

async function loadContent(code) {
  if (code === 'en') return {};
  const cacheKey = `content:${code}`;
  if (cache.has(cacheKey)) return cache.get(cacheKey);
  let byText = {};
  try {
    const [source, target] = await Promise.all([
      fetch('/i18n/content.en.json').then((r) => (r.ok ? r.json() : {})),
      fetch(`/i18n/content.${code}.json`).then((r) => (r.ok ? r.json() : {})),
    ]);
    // The files are keyed by slug; index by the English sentence instead.
    for (const [key, english] of Object.entries(source)) {
      if (target[key]) byText[String(english).trim()] = target[key];
    }
  } catch {
    byText = {};
  }
  cache.set(cacheKey, byText);
  return byText;
}

async function loadCatalogue(code) {
  if (cache.has(code)) return cache.get(code);
  let data = {};
  try {
    const response = await fetch(`/i18n/${code}.json`);
    if (response.ok) data = await response.json();
  } catch {
    // A missing or broken catalogue must not break the page; English stands in.
  }
  cache.set(code, data);
  return data;
}

export function applyTranslations(root = document) {
  root.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  root.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  root.querySelectorAll('[data-i18n-label]').forEach((el) => {
    el.setAttribute('aria-label', t(el.dataset.i18nLabel));
  });
  root.querySelectorAll('[data-i18n-title]').forEach((el) => {
    el.title = t(el.dataset.i18nTitle);
  });
}

export async function setLanguage(code) {
  const entry = LANGUAGES.find((l) => l.code === code);
  if (!entry) return;

  [active, contentMap] = await Promise.all([
    code === 'en' ? Promise.resolve(english) : loadCatalogue(code),
    loadContent(code),
  ]);
  current = code;
  try {
    localStorage.setItem(STORAGE_KEY, code);
  } catch { /* storage unavailable */ }

  document.documentElement.lang = code;
  document.documentElement.dir = entry.rtl ? 'rtl' : 'ltr';
  applyTranslations();
  // Pages listen for this to re-render whatever they built in JS, which
  // data-i18n attributes cannot reach.
  document.dispatchEvent(new CustomEvent('krishi:language', { detail: { code } }));
}

/* Pages build result panels and tables with innerHTML, so their data-i18n
 * elements appear after applyTranslations() has already run. Watching for
 * insertions translates them the moment they land, which is both fewer call
 * sites than translating after every render and impossible to forget when a
 * new panel is added. */
function watchForInsertedNodes() {
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        if (node.matches?.('[data-i18n], [data-i18n-placeholder], [data-i18n-label], [data-i18n-title]')) {
          applyTranslations(node.parentElement ?? node);
        } else if (node.querySelector?.('[data-i18n], [data-i18n-placeholder], [data-i18n-label], [data-i18n-title]')) {
          applyTranslations(node);
        }
      }
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

export async function initLanguage() {
  english = await loadCatalogue('en');

  let saved = null;
  try {
    saved = localStorage.getItem(STORAGE_KEY);
  } catch { /* storage unavailable */ }

  await setLanguage(saved && LANGUAGES.some((l) => l.code === saved) ? saved : 'en');
  watchForInsertedNodes();
}
