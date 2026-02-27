// agent.go — LLM File Agent + Browser Injector
//
// You type prompts in the terminal as normal.
// Every reply is ALSO pushed to the Chrome extension via SSE,
// which then injects it into whatever AI site tab is open.
//
// Setup .env:
//   GROQ_API_KEY=...
//   MISTRAL_API_KEY=...
//   TOGETHER_API_KEY=...
//
// Run: go run agent.go
// The SSE bridge listens on http://localhost:8765

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// ─── ANSI Colors ─────────────────────────────────────────────────────────────

const (
	Reset   = "\033[0m"
	Bold    = "\033[1m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Cyan    = "\033[36m"
	Grey    = "\033[90m"
	Magenta = "\033[35m"
)

// ─── SSE Broadcaster ─────────────────────────────────────────────────────────
// Runs a tiny HTTP server in the background.
// The extension connects to /events and receives each agent reply as it arrives.

type SSEBroadcaster struct {
	mu      sync.Mutex
	clients map[chan string]struct{}
}

var broadcaster = &SSEBroadcaster{
	clients: make(map[chan string]struct{}),
}

func (b *SSEBroadcaster) addClient(ch chan string) {
	b.mu.Lock()
	b.clients[ch] = struct{}{}
	b.mu.Unlock()
}

func (b *SSEBroadcaster) removeClient(ch chan string) {
	b.mu.Lock()
	delete(b.clients, ch)
	b.mu.Unlock()
}

func (b *SSEBroadcaster) send(text string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for ch := range b.clients {
		select {
		case ch <- text:
		default:
		}
	}
	count := len(b.clients)
	if count > 0 {
		fmt.Printf("  %s↗ pushed to %d extension client(s)%s\n", Green, count, Reset)
	}
}

func (b *SSEBroadcaster) clientCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.clients)
}

func startBridgeServer(port string) {
	mux := http.NewServeMux()

	mux.HandleFunc("/events", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		ch := make(chan string, 4)
		broadcaster.addClient(ch)
		defer broadcaster.removeClient(ch)

		fmt.Fprintf(w, "data: {\"type\":\"connected\"}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}

		for {
			select {
			case msg, ok := <-ch:
				if !ok {
					return
				}
				payload, _ := json.Marshal(map[string]string{
					"type": "inject",
					"text": msg,
				})
				fmt.Fprintf(w, "data: %s\n\n", payload)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			case <-r.Context().Done():
				return
			}
		}
	})

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":  "ok",
			"clients": broadcaster.clientCount(),
		})
	})

	go func() {
		if err := http.ListenAndServe(":"+port, mux); err != nil {
			fmt.Printf("%sBridge server error: %v%s\n", Red, err, Reset)
		}
	}()
}

// ─── Spinner ──────────────────────────────────────────────────────────────────

type Spinner struct {
	chars  []string
	idx    int
	mu     sync.Mutex
	active bool
	stop   chan bool
}

func NewSpinner() *Spinner {
	return &Spinner{
		chars: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		stop:  make(chan bool),
	}
}

func (s *Spinner) Start(msg string) {
	s.active = true
	go func() {
		for {
			select {
			case <-s.stop:
				fmt.Printf("\r%s\r", strings.Repeat(" ", len(msg)+15))
				return
			default:
				s.mu.Lock()
				c := s.chars[s.idx%len(s.chars)]
				s.idx++
				s.mu.Unlock()
				fmt.Printf("\r%s%s%s %s", Cyan, c, Reset, msg)
				time.Sleep(80 * time.Millisecond)
			}
		}
	}()
}

func (s *Spinner) Stop() {
	if s.active {
		s.stop <- true
		s.active = false
		time.Sleep(100 * time.Millisecond)
	}
}

// ─── Models ───────────────────────────────────────────────────────────────────

type Model struct {
	Key      string
	ID       string
	Name     string
	Provider string
	Endpoint string
	APIKey   string
}

var modelDefs = []Model{
	{Key: "groq", ID: "llama-3.3-70b-versatile", Name: "Llama 3.3 70B", Provider: "Groq", Endpoint: "https://api.groq.com/openai/v1/chat/completions"},
	{Key: "mistral", ID: "mistral-small-latest", Name: "Mistral Small 3.1", Provider: "Mistral", Endpoint: "https://api.mistral.ai/v1/chat/completions"},
	{Key: "together", ID: "meta-llama/Llama-3.3-70B-Instruct-Turbo", Name: "Llama 3.3 70B (Together)", Provider: "Together", Endpoint: "https://api.together.xyz/v1/chat/completions"},
}

var envKeys = map[string]string{
	"groq":     "GROQ_API_KEY",
	"mistral":  "MISTRAL_API_KEY",
	"together": "TOGETHER_API_KEY",
}

// ─── OpenAI Tool-Calling Types ────────────────────────────────────────────────

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type ChatRequest struct {
	Model      string    `json:"model"`
	Messages   []Message `json:"messages"`
	Tools      []Tool    `json:"tools,omitempty"`
	ToolChoice string    `json:"tool_choice,omitempty"`
	Stream     bool      `json:"stream"`
}

type ChatResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ─── Tool Definitions ─────────────────────────────────────────────────────────

var agentTools = []Tool{
	{Type: "function", Function: ToolFunction{Name: "create_file", Description: "Create or overwrite a file with given content.", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`)}},
	{Type: "function", Function: ToolFunction{Name: "read_file", Description: "Read full contents of a file.", Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}},
	{Type: "function", Function: ToolFunction{Name: "list_files", Description: "List files in a directory.", Parameters: json.RawMessage(`{"type":"object","properties":{"directory":{"type":"string"}},"required":["directory"]}`)}},
	{Type: "function", Function: ToolFunction{Name: "run_command", Description: "Run a shell command, returns stdout+stderr. Use for compiling/running code. Never create .sh scripts. E.g. 'g++ a.cpp -o a && ./a', 'python3 x.py', 'go run main.go'.", Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"},"timeout_seconds":{"type":"string"}},"required":["command"]}`)}},
}

// ─── Tool Execution ───────────────────────────────────────────────────────────

func executeTool(name, argsJSON string) string {
	var args map[string]string
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf("error parsing args: %v", err)
	}
	switch name {
	case "create_file":
		path, content := args["path"], args["content"]
		if path == "" {
			return "error: path required"
		}
		if dir := filepath.Dir(path); dir != "." {
			os.MkdirAll(dir, 0755)
		}
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		return fmt.Sprintf("✓ Created '%s' (%d lines, %d bytes)", path, len(strings.Split(content, "\n")), len(content))

	case "read_file":
		path := args["path"]
		if path == "" {
			return "error: path required"
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		return string(data)

	case "list_files":
		dir := args["directory"]
		if dir == "" {
			dir = "."
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		var lines []string
		for _, e := range entries {
			info, _ := e.Info()
			kind, size := "file", ""
			if e.IsDir() {
				kind = "dir "
			} else if info != nil {
				size = fmt.Sprintf(" (%d bytes)", info.Size())
			}
			lines = append(lines, fmt.Sprintf("[%s] %s%s", kind, e.Name(), size))
		}
		if len(lines) == 0 {
			return "directory is empty"
		}
		return strings.Join(lines, "\n")

	case "run_command":
		command := args["command"]
		if command == "" {
			return "error: command required"
		}
		timeout := 15 * time.Second
		if ts := args["timeout_seconds"]; ts != "" {
			var secs int
			fmt.Sscanf(ts, "%d", &secs)
			if secs > 0 && secs <= 120 {
				timeout = time.Duration(secs) * time.Second
			}
		}
		shell, flag := "/bin/sh", "-c"
		if runtime.GOOS == "windows" {
			shell, flag = "cmd", "/C"
		}
		var outBuf, errBuf bytes.Buffer
		cmd := exec.Command(shell, flag, command)
		cmd.Stdout = &outBuf
		cmd.Stderr = &errBuf
		startTime := time.Now()
		if err := cmd.Start(); err != nil {
			return fmt.Sprintf("error starting: %v", err)
		}
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		var runErr error
		select {
		case runErr = <-done:
		case <-time.After(timeout):
			cmd.Process.Kill()
			return fmt.Sprintf("⏱ timed out after %v\nstdout:\n%sstderr:\n%s", timeout, outBuf.String(), errBuf.String())
		}
		exitCode := 0
		if runErr != nil {
			if ee, ok := runErr.(*exec.ExitError); ok {
				exitCode = ee.ExitCode()
			}
		}
		var r strings.Builder
		r.WriteString(fmt.Sprintf("$ %s\nexit code: %d  |  time: %.2fs\n", command, exitCode, time.Since(startTime).Seconds()))
		if s := outBuf.String(); s != "" {
			r.WriteString("\n── stdout ──\n")
			r.WriteString(s)
		}
		if s := errBuf.String(); s != "" {
			r.WriteString("\n── stderr ──\n")
			r.WriteString(s)
		}
		if outBuf.Len() == 0 && errBuf.Len() == 0 {
			r.WriteString("\n(no output)")
		}
		return r.String()

	default:
		return fmt.Sprintf("unknown tool: %s", name)
	}
}

// ─── API Call ─────────────────────────────────────────────────────────────────

func callAPI(model Model, messages []Message) (*Message, error) {
	reqBody := ChatRequest{Model: model.ID, Messages: messages, Tools: agentTools, ToolChoice: "auto", Stream: false}
	data, _ := json.Marshal(reqBody)
	req, err := http.NewRequest("POST", model.Endpoint, bytes.NewBuffer(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+model.APIKey)
	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	var result ChatResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("parse error: %v", err)
	}
	if result.Error != nil {
		return nil, fmt.Errorf("API error: %s", result.Error.Message)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty response")
	}
	msg := result.Choices[0].Message
	return &msg, nil
}

// ─── Agent Loop ───────────────────────────────────────────────────────────────

func runAgent(model Model, history []Message, spinner *Spinner) (string, error) {
	messages := make([]Message, len(history))
	copy(messages, history)
	for i := 0; i < 10; i++ {
		spinner.Start("thinking")
		msg, err := callAPI(model, messages)
		spinner.Stop()
		if err != nil {
			return "", err
		}
		if len(msg.ToolCalls) == 0 {
			return msg.Content, nil
		}
		messages = append(messages, *msg)
		for _, tc := range msg.ToolCalls {
			fmt.Printf("\n  %s⚙ tool:%s %s%s%s(%s%s%s)\n", Yellow, Reset, Bold, tc.Function.Name, Reset, Grey, summarizeArgs(tc.Function.Arguments), Reset)
			result := executeTool(tc.Function.Name, tc.Function.Arguments)
			fmt.Printf("  %s→%s %s\n", Grey, Reset, result)
			messages = append(messages, Message{Role: "tool", ToolCallID: tc.ID, Name: tc.Function.Name, Content: result})
		}
	}
	return "", fmt.Errorf("agent exceeded max iterations")
}

func summarizeArgs(argsJSON string) string {
	var args map[string]string
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return argsJSON
	}
	var parts []string
	for k, v := range args {
		if len(v) > 40 {
			v = v[:40] + "..."
		}
		parts = append(parts, fmt.Sprintf("%s=%q", k, v))
	}
	return strings.Join(parts, ", ")
}

// ─── .env Loader ─────────────────────────────────────────────────────────────

func loadEnv(filename string) {
	f, err := os.Open(filename)
	if err != nil {
		return
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		k := strings.TrimSpace(parts[0])
		v := strings.Trim(strings.TrimSpace(parts[1]), `"'`)
		if os.Getenv(k) == "" {
			os.Setenv(k, v)
		}
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	loadEnv(".env")

	available := []Model{}
	for _, m := range modelDefs {
		m.APIKey = os.Getenv(envKeys[m.Key])
		if m.APIKey != "" {
			available = append(available, m)
		}
	}

	fmt.Println()
	fmt.Println(Bold + Cyan + "╔══════════════════════════════╗" + Reset)
	fmt.Println(Bold + Cyan + "║   🤖  LLM File Agent  (Go)   ║" + Reset)
	fmt.Println(Bold + Cyan + "╚══════════════════════════════╝" + Reset)
	fmt.Println()

	fmt.Println("  Models:")
	for i, m := range modelDefs {
		hasKey := false
		for _, a := range available {
			if a.Key == m.Key {
				hasKey = true
			}
		}
		if hasKey {
			fmt.Printf("  %s[%d]%s %s %s(%s)%s\n", Green, i+1, Reset, m.Name, Grey, m.Provider, Reset)
		} else {
			fmt.Printf("  %s[%d] %s (%s) — set %s%s\n", Grey, i+1, m.Name, m.Provider, envKeys[m.Key], Reset)
		}
	}
	fmt.Println()

	if len(available) == 0 {
		fmt.Println(Red + "  No API keys found!" + Reset)
		os.Exit(1)
	}

	reader := bufio.NewReader(os.Stdin)
	var selected Model

	if len(available) == 1 {
		selected = available[0]
		fmt.Printf("  Using: %s%s%s\n\n", Bold, selected.Name, Reset)
	} else {
		fmt.Print("  Select model: ")
		input, _ := reader.ReadString('\n')
		idx := 0
		fmt.Sscanf(strings.TrimSpace(input), "%d", &idx)
		if idx < 1 || idx > len(available) {
			fmt.Println(Red + "  Invalid selection." + Reset)
			os.Exit(1)
		}
		selected = available[idx-1]
		fmt.Printf("\n  Using: %s%s%s (%s)\n\n", Bold, selected.Name, Reset, selected.Provider)
	}

	// ── Start SSE bridge ──
	const bridgePort = "8765"
	startBridgeServer(bridgePort)
	fmt.Printf("  %sBridge:%s http://localhost:%s  %s(extension auto-connects)%s\n", Green, Reset, bridgePort, Grey, Reset)
	fmt.Println(Grey + "  Tools: create_file · read_file · list_files · run_command" + Reset)
	fmt.Println(Grey + "  Commands: 'clear' · 'reset' · 'quit'" + Reset)
	fmt.Println()

	history := []Message{
		{Role: "system", Content: `You are a helpful file agent. Create files, read files, list dirs, run commands.
When asked to run code — use run_command directly, never .sh scripts.
C++: "g++ file.cpp -o out && ./out". Python: "python3 script.py". Go: "go run main.go".
Be concise and action-oriented.`},
	}

	spinner := NewSpinner()

	for {
		clients := broadcaster.clientCount()
		extLabel := fmt.Sprintf("%s[no ext]%s ", Grey, Reset)
		if clients > 0 {
			extLabel = fmt.Sprintf("%s[ext ✓]%s ", Green, Reset)
		}
		fmt.Printf("%s%s>%s ", extLabel, Bold, Reset)

		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "quit", "exit":
			fmt.Println("\n  Goodbye!\n")
			return
		case "clear":
			fmt.Print("\033[H\033[2J")
			continue
		case "reset":
			history = history[:1]
			fmt.Println(Grey + "  Conversation reset.\n" + Reset)
			continue
		case "":
			continue
		}

		history = append(history, Message{Role: "user", Content: input})

		start := time.Now()
		reply, err := runAgent(selected, history, spinner)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("\n  %sError: %v%s\n\n", Red, err, Reset)
			history = history[:len(history)-1]
			continue
		}

		history = append(history, Message{Role: "assistant", Content: reply})

		fmt.Printf("\n%s%s%s\n", Bold+Magenta, selected.Name, Reset)
		fmt.Printf("%s\n", reply)
		fmt.Printf("\n%s[%.2fs]%s\n\n", Grey, elapsed.Seconds(), Reset)

		// Push to extension
		broadcaster.send(reply)
	}
}