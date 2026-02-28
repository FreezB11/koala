let currentSite = 'unknown';
let lastResponse = '';

document.addEventListener('DOMContentLoaded', async () => {
  // Detect site
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab.url || '';
    
    if (url.includes('chatgpt.com') || url.includes('chat.openai.com')) {
      currentSite = 'chatgpt';
    } else if (url.includes('gemini.google.com')) {
      currentSite = 'gemini';
    } else if (url.includes('claude.ai')) {
      currentSite = 'claude';
    } else if (url.includes('kimi.com')) {
      currentSite = 'kimi';
    }
    
    document.getElementById('siteBadge').textContent = `Site: ${currentSite.toUpperCase()}`;
    
    // Load stored response
    chrome.storage.local.get([`response_${currentSite}`], (result) => {
      if (result[`response_${currentSite}`]) {
        lastResponse = result[`response_${currentSite}`];
        document.getElementById('responseBox').textContent = lastResponse;
      }
    });
  } catch (e) {}
  
  // Inject buttons
  document.getElementById('injectBtn').addEventListener('click', () => inject(false));
  document.getElementById('forceInject').addEventListener('click', () => inject(true));
  
  // Capture button
  document.getElementById('captureBtn').addEventListener('click', captureResponse);
  
  // Copy button
  document.getElementById('copyBtn').addEventListener('click', () => {
    if (!lastResponse) return showStatus('Nothing to copy');
    navigator.clipboard.writeText(lastResponse).then(() => showStatus('Copied!'));
  });
  
  // Debug buttons
  document.getElementById('testSelector').addEventListener('click', testSelector);
  document.getElementById('findInputs').addEventListener('click', findInputs);
});

async function inject(aggressive) {
  const text = document.getElementById('prompt').value;
  const autoSubmit = document.getElementById('autoSubmit').checked;
  const clearFirst = document.getElementById('clearFirst').checked;
  
  if (!text.trim()) return showStatus('Enter a prompt first', 'error');

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: injectPrompt,
      args: [text, autoSubmit, clearFirst, aggressive, currentSite]
    });
    
    showStatus('Injected!', 'success');
    // REMOVED: window.close() - extension stays open now
  } catch (err) {
    showStatus('Error: ' + err.message, 'error');
  }
}

async function captureResponse() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: captureFromPage,
      args: [currentSite]
    });
    
    if (results[0].result.success) {
      lastResponse = results[0].result.text;
      document.getElementById('responseBox').textContent = lastResponse;
      
      // Save to storage
      chrome.storage.local.set({ [`response_${currentSite}`]: lastResponse });
      showStatus(`Captured ${lastResponse.length} chars`, 'success');
    } else {
      showStatus('No response found. Wait for AI to finish.', 'error');
    }
  } catch (err) {
    showStatus('Error: ' + err.message, 'error');
  }
}

async function testSelector() {
  const selector = document.getElementById('customSelector').value.trim();
  if (!selector) return showStatus('Enter selector first', 'error');
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (sel) => {
        const el = document.querySelector(sel);
        if (el) {
          el.style.border = '3px solid red';
          setTimeout(() => el.style.border = '', 2000);
          return `Found: ${el.tagName}#${el.id}.${el.className.split(' ')[0]}`;
        }
        return 'Not found';
      },
      args: [selector]
    });
    showStatus(results[0].result, 'info');
  } catch (err) {
    showStatus('Error: ' + err.message, 'error');
  }
}

async function findInputs() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const inputs = [];
        document.querySelectorAll('div, textarea, input').forEach((el, i) => {
          if (el.offsetHeight > 30 && el.offsetWidth > 100) {
            const isEditable = el.contentEditable === 'true' || 
                              el.tagName === 'TEXTAREA' || 
                              el.tagName === 'INPUT';
            if (isEditable) {
              inputs.push({
                index: i,
                tag: el.tagName,
                id: el.id,
                class: el.className.split(' ')[0],
                placeholder: el.placeholder?.substring(0, 20)
              });
            }
          }
        });
        console.log('Inputs found:', inputs);
        return inputs.slice(0, 10);
      }
    });
    showStatus(`Found ${results[0].result.length} inputs. Check console.`, 'info');
  } catch (err) {
    showStatus('Error: ' + err.message, 'error');
  }
}

function showStatus(msg, type) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.color = type === 'error' ? '#e94560' : type === 'success' ? '#10b981' : '#aaa';
  setTimeout(() => el.textContent = '', 3000);
}

// ============ INJECTION FUNCTION ============
function injectPrompt(text, autoSubmit, clearFirst, aggressive, site) {
  console.log('[Inject] Starting for', site);
  
  let target = null;
  const selectors = {
    chatgpt: ['#prompt-textarea', '[data-testid="text-input"]', 'div[contenteditable="true"]'],
    gemini: ['.rich-text-editor', 'div[contenteditable="true"]'],
    claude: ['.ProseMirror', '[contenteditable="true"].ProseMirror'],
    kimi: ['#chat-input', '#input', 'div[contenteditable="true"]']
  };
  
  // Try site selectors
  for (const sel of selectors[site] || []) {
    try {
      const el = document.querySelector(sel);
      if (el && isVisible(el)) {
        target = el;
        console.log('[Inject] Found:', sel);
        break;
      }
    } catch (e) {}
  }
  
  // Aggressive mode
  if (!target && aggressive) {
    const candidates = Array.from(document.querySelectorAll('div[contenteditable="true"], textarea'))
      .filter(el => isVisible(el) && el.offsetHeight > 40)
      .sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top);
    
    if (candidates.length > 0) target = candidates[0];
  }
  
  if (!target) throw new Error('No input found');
  
  // Inject
  target.focus();
  target.click();
  
  if (clearFirst) {
    if (target.tagName === 'TEXTAREA') target.value = '';
    else target.innerHTML = '';
  }
  
  if (target.tagName === 'TEXTAREA') {
    target.value = text;
  } else {
    // Handle Kimi/others with p tag
    if (site === 'kimi' && target.querySelector('p')) {
      target.querySelector('p').textContent = text;
    } else {
      target.innerHTML = text.replace(/\n/g, '<br>');
    }
  }
  
  // Trigger events
  ['input', 'change', 'keyup'].forEach(evt => {
    target.dispatchEvent(new Event(evt, { bubbles: true }));
  });
  
  // Submit
  if (autoSubmit) {
    setTimeout(() => {
      target.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      // Try click send button
      document.querySelector('button[type="submit"], button svg, [data-testid="send-button"]')?.click();
    }, 100);
  }
  
  return true;
  
  function isVisible(el) {
    const style = window.getComputedStyle(el);
    return style.display !== 'none' && style.visibility !== 'hidden' && el.offsetParent !== null;
  }
}

// ============ CAPTURE FUNCTION - FIXED FOR CLAUDE ============
function captureFromPage(site) {
  console.log('[Capture] Capturing from', site);
  let text = '';
  let found = false;
  
  // Get all messages first
  let messages = [];
  
  if (site === 'chatgpt') {
    messages = Array.from(document.querySelectorAll('[data-message-author-role="assistant"]'));
    console.log('[Capture] ChatGPT messages found:', messages.length);
    
  } else if (site === 'claude') {
    // CLAUDE FIX: Use the correct class names found in the DOM
    // Assistant responses have class containing 'font-claude-response'
    messages = Array.from(document.querySelectorAll('.font-claude-response'));
    
    // If not found, try broader selector
    if (messages.length === 0) {
      messages = Array.from(document.querySelectorAll('[class*="claude-response"]'));
    }
    
    console.log('[Capture] Claude messages found:', messages.length);
    
  } else if (site === 'gemini') {
    messages = Array.from(document.querySelectorAll('.model-response-text, .response-content'));
    if (messages.length === 0) {
      const allMessages = document.querySelectorAll('.message-content, .content');
      messages = Array.from(allMessages).filter((el) => {
        return el.closest('.model-response') || el.textContent.length > 50;
      });
    }
    console.log('[Capture] Gemini messages found:', messages.length);
    
  } else if (site === 'kimi') {
    messages = Array.from(document.querySelectorAll('.chat-message[data-role="assistant"], [data-role="assistant"]'));
    if (messages.length === 0) {
      const container = document.querySelector('#chat-container, .chat-container, .messages');
      if (container) {
        const children = Array.from(container.children);
        messages = children.filter((el, idx) => idx % 2 === 1 || el.classList.contains('assistant'));
      }
    }
    console.log('[Capture] Kimi messages found:', messages.length);
  }
  
  // Get the last message
  if (messages.length > 0) {
    const lastMessage = messages[messages.length - 1];
    text = lastMessage.innerText || lastMessage.textContent;
    found = true;
    console.log('[Capture] Last message preview:', text.substring(0, 100));
  }
  
  // Fallback
  if (!found) {
    console.log('[Capture] Trying fallback...');
    
    const candidates = Array.from(document.querySelectorAll('div, article, section')).filter(el => {
      const t = el.textContent || '';
      if (t.length < 50 || t.length > 100000) return false;
      if (el.querySelector('input, textarea, button[role="button"]')) return false;
      
      const style = window.getComputedStyle(el);
      if (style.display === 'none' || style.visibility === 'hidden') return false;
      
      const inConversation = el.closest('[data-testid="conversation"], .conversation, main, [role="main"], .chat-ui-core');
      if (!inConversation) return false;
      
      return true;
    });
    
    candidates.sort((a, b) => {
      const position = a.compareDocumentPosition(b);
      return position & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
    });
    
    if (candidates.length > 0) {
      text = candidates[candidates.length - 1].innerText;
      found = true;
      console.log('[Capture] Fallback found:', text.substring(0, 100));
    }
  }
  
  if (found && text) {
    text = text.trim();
    
    // Clean up artifacts
    text = text
      .replace(/Copy code/g, '')
      .replace(/Copy to clipboard/g, '')
      .replace(/\n\s*\n\s*\n/g, '\n\n')
      .substring(0, 50000);
    
    return { success: true, text: text };
  }
  
  return { success: false, text: '' };
}