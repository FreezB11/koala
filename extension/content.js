// content.js — just listens for INJECT messages from the background service worker

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "INJECT" && msg.text) {
    const ok = injectText(msg.text);
    sendResponse({ ok });
  }
  return true;
});

function injectText(text) {
  const selectors = [
    'div.ql-editor[contenteditable="true"]',
    'rich-textarea div[contenteditable="true"]',
    '#prompt-textarea',
    'div[contenteditable="true"].ProseMirror',
    'div[contenteditable="true"]',
    'textarea',
  ];

  let el = null;
  for (const sel of selectors) {
    el = document.querySelector(sel);
    if (el) break;
  }

  if (!el) {
    console.error("[AgentInjector] No input element found");
    return false;
  }

  el.focus();

  if (el.tagName === "TEXTAREA") {
    el.value = text;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  } else {
    document.execCommand("selectAll", false, null);
    document.execCommand("insertText", false, text);
    el.dispatchEvent(new InputEvent("input", { bubbles: true, data: text }));
  }

  console.log("[AgentInjector] Injected:", text.slice(0, 60));

  setTimeout(() => {
    const submitSelectors = [
      'button[aria-label*="Send"]',
      'button[data-testid="send-button"]',
      'button[aria-label="Submit"]',
      'button[jsname="Utb3Nb"]',
    ];
    for (const sel of submitSelectors) {
      const btn = document.querySelector(sel);
      if (btn && !btn.disabled) {
        btn.click();
        console.log("[AgentInjector] Submitted via", sel);
        return;
      }
    }
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", code: "Enter", bubbles: true }));
  }, 500);

  return true;
}