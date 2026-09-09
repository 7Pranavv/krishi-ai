/* Krishi.AI API client.
 *
 * One place that knows the base URL, the auth token, and how the backend
 * reports errors. Every page uses this instead of calling fetch directly.
 */

/* The API is served from the same origin as these pages by default. Set
 * window.KRISHI_API_BASE before this script (see index.html) when the
 * frontend is hosted separately from the backend. */
export const API_BASE = (window.KRISHI_API_BASE || '/api').replace(/\/$/, '');

const TOKEN_KEY = 'krishi.token';
const USER_KEY = 'krishi.user';

/* An error the UI can render directly: it always carries a human message. */
export class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }

  /* True when the feature is temporarily unusable but the rest of the app is
   * fine - shown as an amber "try again" state, not a red error. 429 is here
   * because the AI's daily quota is a wait-and-retry condition, not a fault. */
  get isUnavailable() {
    return [429, 502, 503, 504].includes(this.status);
  }
}

function store(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* Private mode or blocked storage: the session just won't persist. */
  }
}

function read(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export const auth = {
  get token() {
    return read(TOKEN_KEY);
  },
  get user() {
    try {
      return JSON.parse(read(USER_KEY) || 'null');
    } catch {
      return null;
    }
  },
  save({ access_token, user }) {
    store(TOKEN_KEY, access_token);
    store(USER_KEY, JSON.stringify(user));
    return user;
  },
  setUser(user) {
    store(USER_KEY, JSON.stringify(user));
  },
  clear() {
    store(TOKEN_KEY, null);
    store(USER_KEY, null);
  },
};

async function parseError(response) {
  let body = {};
  try {
    body = await response.json();
  } catch {
    /* Non-JSON error (proxy timeout, HTML error page). */
  }
  const message =
    body.error ||
    body.detail ||
    (response.status >= 500
      ? 'Something went wrong on our side. Please try again.'
      : `Request failed (${response.status}).`);
  return new ApiError(String(message), {
    status: response.status,
    detail: typeof body.detail === 'string' ? body.detail : null,
  });
}

/* Core request. `body` may be a plain object (sent as JSON) or FormData. */
export async function request(path, { method = 'GET', body, signal, auth: needsAuth = true } = {}) {
  const headers = {};
  const token = auth.token;
  if (needsAuth && token) headers.Authorization = `Bearer ${token}`;

  let payload;
  if (body instanceof FormData) {
    payload = body; // let the browser set the multipart boundary
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { method, headers, body: payload, signal });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError('Cannot reach Krishi.AI. Check your internet connection.', { status: 0 });
  }

  if (response.status === 401) {
    // The token is gone or expired. Drop it so the next call re-provisions.
    auth.clear();
    throw new ApiError('Your session has expired. Reload the page to continue.', { status: 401 });
  }
  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return null;
  return response.json();
}

/* Make sure we have a usable identity.
 *
 * A farmer should not have to sign up before asking a question, so the first
 * visit silently provisions a guest account. Registering later replaces it. */
let ensuring = null;
export function ensureSession() {
  if (auth.token) return Promise.resolve(auth.user);
  if (!ensuring) {
    ensuring = request('/auth/guest', { method: 'POST', auth: false })
      .then((data) => auth.save(data))
      .finally(() => {
        ensuring = null;
      });
  }
  return ensuring;
}

export const api = {
  health: () => request('/health', { auth: false }),

  register: (payload) => request('/auth/register', { method: 'POST', body: payload, auth: false }),
  login: (payload) => request('/auth/login', { method: 'POST', body: payload, auth: false }),
  me: () => request('/auth/me'),
  updateProfile: (payload) => request('/auth/me', { method: 'PATCH', body: payload }),

  dashboard: () => request('/dashboard'),
  predictions: (tool) => request(`/predictions${tool ? `?tool=${encodeURIComponent(tool)}` : ''}`),

  chat: (message, sessionId, signal) =>
    request('/chat', { method: 'POST', body: { message, session_id: sessionId }, signal }),
  conversations: () => request('/chat/conversations'),
  conversation: (sessionId) => request(`/chat/conversations/${encodeURIComponent(sessionId)}`),
  deleteConversation: (sessionId) =>
    request(`/chat/conversations/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),

  mlOptions: (tool) => request(`/ml/${tool}/options`),

  marketOptions: () => request('/market/options'),
  marketPrices: (commodity, state) => {
    const q = new URLSearchParams();
    if (commodity) q.set('commodity', commodity);
    if (state) q.set('state', state);
    return request(`/market/prices${q.toString() ? `?${q}` : ''}`);
  },
  harvestValue: (payload) =>
    request('/market/harvest-value', { method: 'POST', body: payload }),
  predict: (tool, payload, query) => {
    const q = new URLSearchParams(
      Object.entries(query || {}).filter(([, v]) => v !== '' && v != null));
    return request(`/ml/${tool}/predict${q.toString() ? `?${q}` : ''}`,
                   { method: 'POST', body: payload });
  },
  predictDisease: (formData, signal) =>
    request('/ml/disease/predict', { method: 'POST', body: formData, signal }),
};
