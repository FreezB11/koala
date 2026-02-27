// agent.go — LLM File Agent in Go
// Supports: Mistral, Groq, Together (OpenAI-compatible function calling)
// Tools: create_file, read_file, list_files, run_command
//
// Setup: create a .env file with one or more of:
//   MISTRAL_API_KEY=...
//   GROQ_API_KEY=...
//   TOGETHER_API_KEY=...
//
// Run: go run agent.go

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
	{
		Key:      "groq",
		ID:       "llama-3.3-70b-versatile",
		Name:     "Llama 3.3 70B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	{
		Key:      "mistral",
		ID:       "mistral-small-latest",
		Name:     "Mistral Small 3.1",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	{
		Key:      "together",
		ID:       "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		Name:     "Llama 3.3 70B (Together)",
		Provider: "Together",
		Endpoint: "https://api.together.xyz/v1/chat/completions",
	},
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
	{
		Type: "function",
		Function: ToolFunction{
			Name:        "create_file",
			Description: "Create a new file or overwrite an existing file with the given content. Use this to write code, configs, notes, or any text file.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "File path relative to current directory, e.g. 'hello.go' or 'src/utils.go'"
					},
					"content": {
						"type": "string",
						"description": "Full text content to write to the file"
					}
				},
				"required": ["path", "content"]
			}`),
		},
	},
	{
		Type: "function",
		Function: ToolFunction{
			Name:        "read_file",
			Description: "Read and return the full contents of a file. Use this to inspect existing files before editing.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "File path to read, e.g. 'main.go'"
					}
				},
				"required": ["path"]
			}`),
		},
	},
	{
		Type: "function",
		Function: ToolFunction{
			Name:        "list_files",
			Description: "List all files in a directory. Defaults to current directory if no path given.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"directory": {
						"type": "string",
						"description": "Directory path to list. Use '.' for current directory."
					}
				},
				"required": ["directory"]
			}`),
		},
	},
	{
		Type: "function",
		Function: ToolFunction{
			Name: "run_command",
			Description: "Execute a shell command and return its real stdout and stderr output. Use this to compile and run code — do NOT create shell scripts. Examples: compile C++ with 'g++ hello.cpp -o hello', run with './hello', run Python with 'python3 script.py', run Go with 'go run main.go', or do both at once with 'g++ hello.cpp -o hello && ./hello'. Always prefer running directly rather than writing a script.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"command": {
						"type": "string",
						"description": "Shell command to execute, e.g. 'g++ hello.cpp -o hello && ./hello'"
					},
					"timeout_seconds": {
						"type": "string",
						"description": "Max seconds to wait before killing the process. Default 15. Use more for slow builds."
					}
				},
				"required": ["command"]
			}`),
		},
	},
	{
    Type: "function",
    Function: ToolFunction{
        Name:        "create_directory",
        Description: "Create a directory and all parent directories if they don't exist. Use this to set up project structures before creating files.",
        Parameters: json.RawMessage(`{
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to create, e.g. 'src/components' or 'projects/myapp'"
                }
            },
            "required": ["path"]
        }`),
    },
},
}

// ─── Tool Execution ───────────────────────────────────────────────────────────

func executeTool(name, argsJSON string) string {
	var args map[string]string
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return fmt.Sprintf("error parsing args: %v", err)
	}

	switch name {

	case "create_file":
		path := args["path"]
		content := args["content"]
		if path == "" {
			return "error: path is required"
		}
		// Ensure parent dirs exist
		dir := filepath.Dir(path)
		if dir != "." {
			if err := os.MkdirAll(dir, 0755); err != nil {
				return fmt.Sprintf("error creating directories: %v", err)
			}
		}
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return fmt.Sprintf("error writing file: %v", err)
		}
		lines := len(strings.Split(content, "\n"))
		return fmt.Sprintf("✓ Created '%s' (%d lines, %d bytes)", path, lines, len(content))

	case "read_file":
		path := args["path"]
		if path == "" {
			return "error: path is required"
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Sprintf("error reading file: %v", err)
		}
		return string(data)

	case "list_files":
		dir := args["directory"]
		if dir == "" {
			dir = "."
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return fmt.Sprintf("error listing directory: %v", err)
		}
		if len(entries) == 0 {
			return fmt.Sprintf("directory '%s' is empty", dir)
		}
		var lines []string
		for _, e := range entries {
			info, _ := e.Info()
			kind := "file"
			size := ""
			if e.IsDir() {
				kind = "dir "
			} else if info != nil {
				size = fmt.Sprintf(" (%d bytes)", info.Size())
			}
			lines = append(lines, fmt.Sprintf("[%s] %s%s", kind, e.Name(), size))
		}
		return strings.Join(lines, "\n")

	case "create_directory":
		path := args["path"]
		if path == "" {
			return "error: path is required"
		}
		if err := os.MkdirAll(path, 0755); err != nil {
			return fmt.Sprintf("error creating directory: %v", err)
		}
		return fmt.Sprintf("✓ Created directory '%s'", path)

	case "run_command":
		command := args["command"]
		if command == "" {
			return "error: command is required"
		}

		// Parse optional timeout
		timeout := 15 * time.Second
		if ts := args["timeout_seconds"]; ts != "" {
			var secs int
			fmt.Sscanf(ts, "%d", &secs)
			if secs > 0 && secs <= 120 {
				timeout = time.Duration(secs) * time.Second
			}
		}

		// Pick shell based on OS
		var shell, flag string
		if runtime.GOOS == "windows" {
			shell, flag = "cmd", "/C"
		} else {
			shell, flag = "/bin/sh", "-c"
		}

		var stdoutBuf, stderrBuf bytes.Buffer
		cmd := exec.Command(shell, flag, command)
		cmd.Stdout = &stdoutBuf
		cmd.Stderr = &stderrBuf

		startTime := time.Now()
		if err := cmd.Start(); err != nil {
			return fmt.Sprintf("error starting command: %v", err)
		}

		// Wait with timeout
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()

		var runErr error
		select {
		case runErr = <-done:
			// completed normally
		case <-time.After(timeout):
			cmd.Process.Kill()
			return fmt.Sprintf("⏱ timed out after %v\nstdout:\n%s\nstderr:\n%s",
				timeout, stdoutBuf.String(), stderrBuf.String())
		}

		elapsed := time.Since(startTime)
		exitCode := 0
		if runErr != nil {
			if exitErr, ok := runErr.(*exec.ExitError); ok {
				exitCode = exitErr.ExitCode()
			}
		}

		out := stdoutBuf.String()
		errOut := stderrBuf.String()

		var result strings.Builder
		result.WriteString(fmt.Sprintf("$ %s\n", command))
		result.WriteString(fmt.Sprintf("exit code: %d  |  time: %.2fs\n", exitCode, elapsed.Seconds()))
		if out != "" {
			result.WriteString("\n── stdout ──\n")
			result.WriteString(out)
		}
		if errOut != "" {
			result.WriteString("\n── stderr ──\n")
			result.WriteString(errOut)
		}
		if out == "" && errOut == "" {
			result.WriteString("\n(no output)")
		}
		return result.String()

	default:
		return fmt.Sprintf("unknown tool: %s", name)
	}
}

// ─── API Call ─────────────────────────────────────────────────────────────────

func callAPI(model Model, messages []Message) (*Message, error) {
	reqBody := ChatRequest{
		Model:      model.ID,
		Messages:   messages,
		Tools:      agentTools,
		ToolChoice: "auto",
		Stream:     false,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

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
		return nil, fmt.Errorf("parse error: %v\nbody: %s", err, string(body))
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

// runAgent sends the conversation, handles tool calls in a loop, returns final reply.
func runAgent(model Model, history []Message, spinner *Spinner) (string, error) {
	messages := make([]Message, len(history))
	copy(messages, history)

	for iterations := 0; iterations < 10; iterations++ {
		spinner.Start("thinking")
		msg, err := callAPI(model, messages)
		spinner.Stop()

		if err != nil {
			return "", err
		}

		// No tool calls — we have a final answer
		if len(msg.ToolCalls) == 0 {
			return msg.Content, nil
		}

		// Append assistant message with tool calls
		messages = append(messages, *msg)

		// Execute each tool call and collect results
		for _, tc := range msg.ToolCalls {
			fmt.Printf("\n  %s⚙ tool:%s %s%s%s(%s%s%s)\n",
				Yellow, Reset,
				Bold, tc.Function.Name, Reset,
				Grey, summarizeArgs(tc.Function.Arguments), Reset,
			)

			result := executeTool(tc.Function.Name, tc.Function.Arguments)

			// Print a short preview of the result
			preview := result
			// lines := strings.Split(result, "\n")
			// if len(lines) > 4 {
			// 	preview = strings.Join(lines[:4], "\n") + fmt.Sprintf("\n  %s... (%d more lines)%s", Grey, len(lines)-4, Reset)
			// }
			fmt.Printf("  %s→ %s%s\n", Grey, Reset, preview)

			// Append tool result
			messages = append(messages, Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
				Content:    result,
			})
		}
		// Loop: send tool results back to the model
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

	// Hydrate API keys
	available := []Model{}
	for _, m := range modelDefs {
		m.APIKey = os.Getenv(envKeys[m.Key])
		if m.APIKey != "" {
			available = append(available, m)
		}
	}

	// ── Banner ──
	fmt.Println()
	fmt.Println(Bold + Cyan + "╔══════════════════════════════╗" + Reset)
	fmt.Println(Bold + Cyan + "║   🤖  LLM File Agent  (Go)   ║" + Reset)
	fmt.Println(Bold + Cyan + "╚══════════════════════════════╝" + Reset)
	fmt.Println()

	// ── Show models ──
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

	// ── Check keys ──
	if len(available) == 0 {
		fmt.Println(Red + "  No API keys found!" + Reset)
		fmt.Println("\n  Create a " + Bold + ".env" + Reset + " file:")
		for _, m := range modelDefs {
			fmt.Printf("    %s%s=your_key_here%s\n", Grey, envKeys[m.Key], Reset)
		}
		fmt.Println()
		os.Exit(1)
	}

	// ── Select model ──
	reader := bufio.NewReader(os.Stdin)
	var selected Model

	if len(available) == 1 {
		selected = available[0]
		fmt.Printf("  Using: %s%s%s\n\n", Bold, selected.Name, Reset)
	} else {
		fmt.Print("  Select model: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		idx := 0
		fmt.Sscanf(input, "%d", &idx)
		if idx < 1 || idx > len(available) {
			fmt.Println(Red + "  Invalid selection." + Reset)
			os.Exit(1)
		}
		selected = available[idx-1]
		fmt.Printf("\n  Using: %s%s%s (%s)\n\n", Bold, selected.Name, Reset, selected.Provider)
	}

	// ── Tools info ──
	fmt.Println(Grey + "  Tools: create_file · read_file · list_files · run_command" + Reset)
	fmt.Println(Grey + "  Commands: 'clear' clear screen · 'reset' new conversation · 'quit' exit" + Reset)
	fmt.Println()

	// ── System prompt ──
	history := []Message{
		{
			Role: "system",
			Content: `You are a helpful file agent with the ability to create files, read files, list directories, and execute shell commands.

When the user asks you to create code — use create_file to write it.
When the user asks you to run or execute a file — use run_command directly. Do NOT create shell scripts (.sh files) to run things.
When compiling and running (e.g. C/C++), do it in one run_command call: "g++ hello.cpp -o hello && ./hello"
For Python: "python3 script.py", for Go: "go run main.go", for Node: "node app.js"
If a compile step fails (exit code != 0), report the error from stderr and offer to fix the code.
Always show the actual output from the program to the user.
Be concise and action-oriented.`,
		},
	}

	spinner := NewSpinner()

	// ── Chat loop ──
	for {
		fmt.Print(Bold + "> " + Reset)
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
			history = history[:1] // keep system prompt
			fmt.Println(Grey + "  Conversation reset.\n" + Reset)
			continue
		case "":
			continue
		}

		// Append user message
		history = append(history, Message{Role: "user", Content: input})

		start := time.Now()
		reply, err := runAgent(selected, history, spinner)
		elapsed := time.Since(start)

		if err != nil {
			fmt.Printf("\n  %sError: %v%s\n\n", Red, err, Reset)
			// Remove the failed user message to keep history clean
			history = history[:len(history)-1]
			continue
		}

		// Append assistant reply to history
		history = append(history, Message{Role: "assistant", Content: reply})

		fmt.Printf("\n%s%s%s\n", Bold+Magenta, selected.Name, Reset)
		fmt.Printf("%s\n", reply)
		fmt.Printf("\n%s[%.2fs]%s\n\n", Grey, elapsed.Seconds(), Reset)
	}
}