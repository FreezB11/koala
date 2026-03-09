let ws;

connect();

function connect() {

  ws = new WebSocket("ws://localhost:3000");

  ws.onopen = () => {
    console.log("[EXT] Connected to server");
  };

  ws.onmessage = async (event) => {

    const msg = JSON.parse(event.data);

    if (msg.type === "inject") {
      await inject(msg.text);
    }

    if (msg.type === "capture") {
      await capture();
    }

  };

  ws.onclose = () => {
    console.log("[EXT] reconnecting...");
    setTimeout(connect,2000);
  };

  ws.onerror = () => ws.close();

}



async function getTargetTab(){

  const tabs = await chrome.tabs.query({
    url:[
      "https://chatgpt.com/*",
      "https://chat.openai.com/*",
      "https://claude.ai/*",
      "https://gemini.google.com/*",
      "https://kimi.com/*"
    ]
  });

  if(!tabs.length){
    console.log("No AI tab found");
    return [];
  }

  return tabs;
}



// async function inject(text){

//   const tab = await getTargetTab();
//   if(!tab) return;

//   await chrome.scripting.executeScript({

//     target:{tabId:tab.id},

//     func:(text)=>{

//       const el =
//         document.querySelector("#prompt-textarea") ||
//         document.querySelector(".ProseMirror") ||
//         document.querySelector('[contenteditable="true"]') ||
//         document.querySelector("textarea");

//       if(!el){
//         console.log("input not found");
//         return;
//       }

//       el.focus();

//       if(el.tagName==="TEXTAREA"){

//         el.value=text;
//         el.dispatchEvent(new Event("input",{bubbles:true}));

//       }else{

//         el.innerHTML=text.replace(/\n/g,"<br>");

//       }
//       const enterEvent = new KeyboardEvent("keydown", {
//         bubbles: true,
//         cancelable: true,
//         key: "Enter",
//         code: "Enter"
//       });

//       el.dispatchEvent(enterEvent);
//       el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
//       // Try click send button
//       document.querySelector('button[type="submit"], button svg, [data-testid="send-button"]')?.click();

//     },

//     args:[text]

//   });

// }
async function inject(text){

  const tabs = await getTargetTabs();
  if(!tabs.length) return;

  for(const tab of tabs){

    await chrome.scripting.executeScript({

      target:{tabId:tab.id},

      func:(text)=>{

        const el =
          document.querySelector("#prompt-textarea") ||
          document.querySelector(".ProseMirror") ||
          document.querySelector('[contenteditable="true"]') ||
          document.querySelector("textarea");

        if(!el){
          console.log("input not found");
          return;
        }

        el.focus();

        if(el.tagName==="TEXTAREA"){
          el.value=text;
          el.dispatchEvent(new Event("input",{bubbles:true}));
        }else{
          el.innerHTML=text.replace(/\n/g,"<br>");
        }

        el.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true}));

        document.querySelector(
          'button[type="submit"],[data-testid="send-button"]'
        )?.click();

      },

      args:[text]

    });

  }

}


async function capture(){

  const tab = await getTargetTab();
  if(!tab) return;

  const result = await chrome.scripting.executeScript({

    target:{tabId:tab.id},

    func:()=>{

      const msgs=document.querySelectorAll(
        '[data-message-author-role="assistant"], .font-claude-response'
      );

      if(!msgs.length) return "";

      return msgs[msgs.length-1].innerText;

    }

  });

  ws.send(JSON.stringify({
    type:"capture_result",
    text: result[0].result
  }));

}