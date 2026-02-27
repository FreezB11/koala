const dot = document.getElementById("dot");
const txt = document.getElementById("statusText");

async function check() {
  try {
    const r = await fetch("http://localhost:8765/health", { signal: AbortSignal.timeout(2000) });
    const d = await r.json();
    dot.className = "dot online";
    txt.textContent = `online · ${d.clients} listener(s)`;
  } catch {
    dot.className = "dot offline";
    txt.textContent = "offline";
  }
}

check();
setInterval(check, 3000);

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "STATUS") {
    dot.className = "dot " + (msg.state === "connected" ? "online" : "offline");
    txt.textContent = msg.state;
  }
});