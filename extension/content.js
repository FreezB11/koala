// Content script runs on AI chat pages to provide backup injection capability
console.log('AI Prompt Filler: Content script loaded');

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'fillInput') {
    try {
      const result = fillInputOnPage(request.text, request.site, request.autoSubmit);
      sendResponse({ success: true, result });
    } catch (error) {
      sendResponse({ success: false, error: error.message });
    }
    return true; // Keep channel open for async
  }
});

function fillInputOnPage(text, site, autoSubmit) {
  // Same logic as in popup.js but accessible via messaging
  const inputSelectors = {
    chatgpt: [
      '#prompt-textarea',
      '[data-testid="text-input"]',
      'div[contenteditable="true"]',
      'textarea[placeholder*="Message"]'
    ],
    gemini: [
      '.rich-text-editor',
      'div[contenteditable="true"]',
      'input-area textarea',
      '[placeholder*="Ask"]'
    ],
    claude: [
      '.ProseMirror',
      'div[contenteditable="true"]',
      '[role="textbox"]'
    ]
  };

  const selectors = inputSelectors[site] || [];
  let element = null;

  for (const selector of selectors) {
    element = document.querySelector(selector);
    if (element) break;
  }

  if (!element) {
    throw new Error('Input field not found');
  }

  // Fill the field
  element.focus();
  
  if (element.tagName === 'TEXTAREA' || element.tagName === 'INPUT') {
    element.value = text;
  } else {
    element.innerHTML = text.replace(/\n/g, '<br>');
  }

  // Trigger events
  ['focus', 'input', 'change', 'keyup'].forEach(eventType => {
    const event = new Event(eventType, { bubbles: true });
    element.dispatchEvent(event);
  });

  if (autoSubmit) {
    setTimeout(() => {
      const keyEvent = new KeyboardEvent('keydown', {
        key: 'Enter',
        code: 'Enter',
        keyCode: 13,
        which: 13,
        bubbles: true
      });
      element.dispatchEvent(keyEvent);
    }, 100);
  }

  return { filled: true, element: element.tagName };
}