const WebSocket = require("ws");
const readline = require("readline");

const wss = new WebSocket.Server({ port: 3000 });

let client = null;

wss.on("connection", (ws) => {

  console.log("Extension connected");

  client = ws;

  ws.on("message", (msg) => {
    console.log("FROM EXT:", msg.toString());
  });

});

// const rl = readline.createInterface({
//   input: process.stdin,
//   output: process.stdout
// });

console.log("Commands:");
console.log("inject <text>");
console.log("capture");

// rl.on("line", (line) => {

//   if (!client) {
//     console.log("No extension connected");
//     return;
//   }

//   if (line.startsWith("inject ")) {

//     const text = line.slice(7);

//     client.send(JSON.stringify({
//       type: "inject",
//       text
//     }));

//   }

//   if (line === "capture") {

//     client.send(JSON.stringify({
//       type: "capture"
//     }));

//   }

// });

// const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on("line", (line) => {

  if(!client){
    console.log("Extension not connected");
    return;
  }

  if(line.startsWith("inject ")){

    const text = line.slice(7);

    client.send(JSON.stringify({
      type: "inject",
      text: text
    }));
    // client.send(JSON.stringify({
    //   type: "inject",
    //   text: " "
    // }));
  }

  if(line === "capture"){

    client.send(JSON.stringify({
      type: "capture"
    }));

  }

});