// background.js — service worker
// Connects to the Go SSE bridge and forwards inject messages to the active tab.

const BRIDGE = "http://localhost:8765/events";
const RETRY_DELAY = 5000;

let es = null;

function connect() {
  if (es) es.close();

  try {
    es = new EventSource(BRIDGE);
  } catch (e) {
    console.error("[AgentBG] EventSource failed:", e);
    setTimeout(connect, RETRY_DELAY);
    return;
  }

  es.onopen = () => {
    console.log("[AgentBG] Connected to bridge");
  };

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "inject" && data.text) {
        injectIntoActiveTab(data.text);
      }
    } catch (e) {
      console.error("[AgentBG] Parse error:", e);
    }
  };

  es.onerror = () => {
    console.warn("[AgentBG] Disconnected, retrying...");
    es.close();
    setTimeout(connect, RETRY_DELAY);
  };
}

async function injectIntoActiveTab(text) {
  // Find an AI site tab, or use the active tab
  const targets = [
    "https://gemini.google.com",
    "https://chat.openai.com",
    "https://chatgpt.com",
    "https://claude.ai",
  ];

  const allTabs = await chrome.tabs.query({});
  let tab = allTabs.find(t => t.url && targets.some(u => t.url.startsWith(u)));

  if (!tab) {
    // Fall back to active tab
    const active = await chrome.tabs.query({ active: true, currentWindow: true });
    tab = active[0];
  }

  if (!tab) {
    console.error("[AgentBG] No suitable tab found");
    return;
  }

  console.log("[AgentBG] Sending to tab:", tab.url);

  try {
    await chrome.tabs.sendMessage(tab.id, { type: "INJECT", text });
  } catch (e) {
    // Content script not loaded yet — inject it first
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"],
    });
    await chrome.tabs.sendMessage(tab.id, { type: "INJECT", text });
  }
}

connect();