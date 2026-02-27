// server.go — Agent HTTP Bridge
// Receives prompts from the browser extension, runs them through the LLM agent,
// prints the full conversation to the terminal, and returns the result.
//
// Setup: same .env as agent.go
//   GROQ_API_KEY=...
//   MISTRAL_API_KEY=...
//   TOGETHER_API_KEY=...
//
// Run: go run server.go
// Then load the extension in Chrome and send prompts.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// ─── ANSI ─────────────────────────────────────────────────────────────────────

const (
	Reset   = "\033[0m"
	Bold    = "\033[1m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Cyan    = "\033[36m"
	Grey    = "\033[90m"
	Magenta = "\033[35m"
	Blue    = "\033[34m"
)

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

// ─── LLM Types ────────────────────────────────────────────────────────────────

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type ChatResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ─── Request/Response for extension ──────────────────────────────────────────

type PromptRequest struct {
	Prompt string `json:"prompt"`
	Target string `json:"target"` // e.g. "https://gemini.google.com"
}

type PromptResponse struct {
	Result string `json:"result"`
	Model  string `json:"model"`
	Took   string `json:"took"`
}

// ─── Active model ─────────────────────────────────────────────────────────────

var activeModel Model

// ─── LLM call ─────────────────────────────────────────────────────────────────

func callLLM(prompt string) (string, error) {
	messages := []ChatMessage{
		{
			Role:    "system",
			Content: "You are a helpful assistant. Answer clearly and concisely.",
		},
		{
			Role:    "user",
			Content: prompt,
		},
	}

	reqBody := ChatRequest{
		Model:    activeModel.ID,
		Messages: messages,
		Stream:   false,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", activeModel.Endpoint, bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+activeModel.APIKey)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result ChatResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}
	if result.Error != nil {
		return "", fmt.Errorf("API error: %s", result.Error.Message)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("empty response")
	}

	return result.Choices[0].Message.Content, nil
}

// ─── HTTP Handlers ────────────────────────────────────────────────────────────

func withCORS(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h(w, r)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"model":  activeModel.Name,
	})
}

func processHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PromptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	if req.Prompt == "" {
		http.Error(w, "prompt is empty", http.StatusBadRequest)
		return
	}

	// ── Print incoming prompt to terminal ──
	fmt.Printf("\n%s╭─ incoming prompt %s%s\n", Cyan, Grey+"["+targetName(req.Target)+"]"+Reset, "")
	fmt.Printf("%s│%s %s\n", Cyan, Reset, req.Prompt)
	fmt.Printf("%s╰─%s model: %s%s%s\n", Cyan, Grey, Magenta, activeModel.Name, Reset)
	fmt.Println()

	start := time.Now()
	result, err := callLLM(req.Prompt)
	elapsed := time.Since(start)

	if err != nil {
		fmt.Printf("%s✗ error: %v%s\n", Red, err, Reset)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// ── Print agent response to terminal ──
	fmt.Printf("%s┌─ agent response %s[%.2fs]%s\n", Green, Grey, elapsed.Seconds(), Reset)
	lines := strings.Split(result, "\n")
	for _, line := range lines {
		fmt.Printf("%s│%s %s\n", Green, Reset, line)
	}
	fmt.Printf("%s└─%s injecting into %s%s%s\n\n",
		Green, Grey, Yellow, targetName(req.Target), Reset)

	// ── Send back to extension ──
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(PromptResponse{
		Result: result,
		Model:  activeModel.Name,
		Took:   fmt.Sprintf("%.2fs", elapsed.Seconds()),
	})
}

func targetName(url string) string {
	switch {
	case strings.Contains(url, "gemini"):
		return "Gemini"
	case strings.Contains(url, "chatgpt"), strings.Contains(url, "openai"):
		return "ChatGPT"
	case strings.Contains(url, "claude"):
		return "Claude"
	default:
		return url
	}
}

// ─── .env loader ─────────────────────────────────────────────────────────────

func loadEnv(file string) {
	f, err := os.Open(file)
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

	// Find available models
	available := []Model{}
	for _, m := range modelDefs {
		m.APIKey = os.Getenv(envKeys[m.Key])
		if m.APIKey != "" {
			available = append(available, m)
		}
	}

	// ── Banner ──
	fmt.Println()
	fmt.Println(Bold + Blue + "╔═══════════════════════════════════╗" + Reset)
	fmt.Println(Bold + Blue + "║   🌐  Agent HTTP Bridge  :8080    ║" + Reset)
	fmt.Println(Bold + Blue + "╚═══════════════════════════════════╝" + Reset)
	fmt.Println()

	if len(available) == 0 {
		fmt.Println(Red + "  No API keys found! Create a .env file:" + Reset)
		for _, m := range modelDefs {
			fmt.Printf("    %s%s=your_key%s\n", Grey, envKeys[m.Key], Reset)
		}
		os.Exit(1)
	}

	// ── Pick model ──
	fmt.Println("  Models:")
	for i, m := range available {
		fmt.Printf("  %s[%d]%s %s %s(%s)%s\n",
			Green, i+1, Reset, m.Name, Grey, m.Provider, Reset)
	}
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)

	if len(available) == 1 {
		activeModel = available[0]
	} else {
		fmt.Print("  Select model: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		idx := 0
		fmt.Sscanf(input, "%d", &idx)
		if idx < 1 || idx > len(available) {
			fmt.Println(Red + "  Invalid." + Reset)
			os.Exit(1)
		}
		activeModel = available[idx-1]
	}

	fmt.Printf("\n  %sUsing:%s %s%s%s (%s)\n",
		Grey, Reset, Bold+Magenta, activeModel.Name, Reset, activeModel.Provider)
	fmt.Printf("  %sListening on%s %s%s%s\n\n",
		Grey, Reset, Bold+Cyan, "http://localhost:8080", Reset)
	fmt.Println(Grey + "  Load the extension in Chrome, then send prompts from the popup." + Reset)
	fmt.Println(Grey + "  Press Ctrl+C to stop." + Reset)
	fmt.Println()
	fmt.Println(strings.Repeat("─", 50))

	// ── Routes ──
	http.HandleFunc("/health",  withCORS(healthHandler))
	http.HandleFunc("/process", withCORS(processHandler))

	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(Red+"server error:"+Reset, err)
		os.Exit(1)
	}
}