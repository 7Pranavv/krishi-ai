/* Conversation controller shared by the full chat page and the floating widget.
 *
 * Handles: empty input, in-flight de-duplication, abort on navigation,
 * error bubbles, session continuity, and autoscroll. */

import { api, ensureSession } from './api.js';
import { escapeHtml, formatDate, formatReply, t } from './ui.js';

export function createChat({ log, input, sendButton, onSession }) {
  let sessionId = null;
  let pending = null; // AbortController while a reply is in flight

  function scroll() {
    log.scrollTop = log.scrollHeight;
  }

  function bubble(role, text, meta) {
    // The welcome panel and its suggestions belong to an empty log only.
    log.querySelectorAll('[data-welcome]').forEach((node) => node.remove());
    const el = document.createElement('div');
    el.className = `bubble bubble-${role}`;
    // The farmer's own text is never formatted; only replies are.
    if (role === 'bot') el.innerHTML = formatReply(text);
    else el.textContent = text;
    if (meta) {
      const note = document.createElement('div');
      note.className = 'bubble-meta';
      note.textContent = meta;
      el.append(note);
    }
    log.append(el);
    scroll();
    return el;
  }

  function typing() {
    const el = document.createElement('div');
    el.className = 'bubble bubble-bot';
    el.innerHTML = `<span class="spinner" role="status" aria-label="${escapeHtml(t('common_working'))}"></span>`;
    log.append(el);
    scroll();
    return el;
  }

  async function send(text) {
    const message = (text ?? input.value).trim();
    if (!message) {
      input.focus();
      return;
    }
    if (pending) return; // one question at a time

    input.value = '';
    input.style.height = 'auto';
    bubble('user', message);

    const placeholder = typing();
    pending = new AbortController();
    sendButton.disabled = true;

    try {
      await ensureSession();
      const data = await api.chat(message, sessionId, pending.signal);
      sessionId = data.session_id;
      onSession?.(sessionId);
      placeholder.remove();
      bubble('bot', data.reply, data.source === 'faq' ? t('chat_saved_answer') : null);
    } catch (error) {
      placeholder.remove();
      if (error.name === 'AbortError') return;
      const el = bubble('error', error.message);
      if (error.detail) {
        const note = document.createElement('div');
        note.className = 'bubble-meta';
        note.textContent = error.detail;
        el.append(note);
      }
    } finally {
      pending = null;
      sendButton.disabled = false;
      input.focus();
    }
  }

  function reset() {
    pending?.abort();
    pending = null;
    sessionId = null;
    log.replaceChildren();
    onSession?.(null);
  }

  async function load(id) {
    pending?.abort();
    pending = null;
    sessionId = id;
    log.innerHTML = '<div class="state"><div class="spinner"></div></div>';
    try {
      const data = await api.conversation(id);
      log.replaceChildren();
      data.messages.forEach((m) =>
        bubble(m.role === 'user' ? 'user' : 'bot', m.content, formatDate(m.created_at)));
      onSession?.(id);
    } catch (error) {
      log.innerHTML = `<div class="alert alert-error" role="alert">${escapeHtml(error.message)}</div>`;
    }
  }

  // Enter sends, Shift+Enter makes a new line.
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  });
  sendButton.addEventListener('click', () => send());

  return {
    send,
    reset,
    load,
    get sessionId() { return sessionId; },
    set sessionId(id) { sessionId = id; },
  };
}
