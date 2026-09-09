/* Floating assistant, available on every page except the full chat page.
 * Its conversation is a normal conversation - it shows up in chat history. */

import { createChat } from './chat-core.js';
import { t } from './i18n.js';

const OPEN_KEY = 'krishi.widget.open';

export function mountChatWidget() {
  if (document.getElementById('krishi-chat-fab')) return;

  const fab = document.createElement('button');
  fab.id = 'krishi-chat-fab';
  fab.className = 'chat-fab';
  fab.type = 'button';
  fab.textContent = '💬';
  fab.setAttribute('aria-label', t('chat_open'));
  fab.setAttribute('aria-expanded', 'false');

  const panel = document.createElement('section');
  panel.className = 'chat-panel';
  panel.id = 'krishi-chat-panel';
  panel.hidden = true;
  panel.setAttribute('aria-label', 'Krishi.AI assistant');
  panel.innerHTML = `
    <div class="chat-panel-head">
      <span aria-hidden="true">🌱</span>
      <span>Krishi.AI</span>
      <span class="spacer"></span>
      <a class="btn-quiet" href="/chat" style="text-decoration:none"
         data-i18n="chat_open_full">${t('chat_open_full')}</a>
      <button class="icon-btn" type="button" data-close data-i18n-label="chat_close"
              aria-label="${t('chat_close')}" style="width:30px;height:30px">✕</button>
    </div>
    <div class="chat-log" id="widget-log"></div>
    <div class="chat-composer">
      <label class="sr-only" for="widget-input" data-i18n="chat_your_question">${t('chat_your_question')}</label>
      <textarea id="widget-input" rows="1" data-i18n-placeholder="assistant_prompt"
                placeholder="${t('assistant_prompt')}"></textarea>
      <button class="btn" id="widget-send" type="button" data-i18n-label="chat_send"
              aria-label="${t('chat_send')}">➤</button>
    </div>`;

  fab.setAttribute('aria-controls', panel.id);
  document.body.append(panel, fab);

  const log = panel.querySelector('#widget-log');
  const chat = createChat({
    log,
    input: panel.querySelector('#widget-input'),
    sendButton: panel.querySelector('#widget-send'),
  });

  const greet = () => {
    if (log.childElementCount) return;
    const el = document.createElement('div');
    el.className = 'bubble bubble-bot';
    el.textContent = t('welcome_message');
    log.append(el);
  };

  function setOpen(open) {
    panel.hidden = !open;
    fab.textContent = open ? '✕' : '💬';
    fab.setAttribute('aria-expanded', String(open));
    try {
      sessionStorage.setItem(OPEN_KEY, open ? '1' : '0');
    } catch { /* storage unavailable */ }
    if (open) {
      greet();
      panel.querySelector('#widget-input').focus();
    }
  }

  // The greeting bubble is plain text, so it has to be rewritten on a
  // language change - data-i18n cannot reach it.
  document.addEventListener('krishi:language', () => {
    const first = log.firstElementChild;
    if (log.childElementCount === 1 && first?.classList.contains('bubble-bot')) {
      first.textContent = t('welcome_message');
    }
  });

  fab.addEventListener('click', () => setOpen(panel.hidden));
  panel.querySelector('[data-close]').addEventListener('click', () => setOpen(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !panel.hidden) setOpen(false);
  });

  let wasOpen = false;
  try {
    wasOpen = sessionStorage.getItem(OPEN_KEY) === '1';
  } catch { /* storage unavailable */ }
  setOpen(wasOpen);

  return chat;
}
