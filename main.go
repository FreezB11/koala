// main.go
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
	"sync"
)

// ANSI color codes
const (
	ColorReset   = "\033[0m"
	ColorRed     = "\033[31m"
	ColorGreen   = "\033[32m"
	ColorYellow  = "\033[33m"
	ColorBlue    = "\033[34m"
	ColorMagenta = "\033[35m"
	ColorCyan    = "\033[36m"
	ColorGrey    = "\033[90m"
	ColorBold    = "\033[1m"
)

// Spinner for loading
type Spinner struct {
	chars  []string
	index  int
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

func (s *Spinner) Start(message string) {
	s.active = true
	go func() {
		for {
			select {
			case <-s.stop:
				fmt.Printf("\r%s\r", strings.Repeat(" ", len(message)+10))
				return
			default:
				s.mu.Lock()
				idx := s.index % len(s.chars)
				s.index++
				s.mu.Unlock()
				
				fmt.Printf("\r%s %s %s%s%s", 
					ColorCyan+s.chars[idx]+ColorReset,
					message,
					ColorGrey, 
					strings.Repeat(".", s.index%4),
					ColorReset)
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
}

func (s *Spinner) Stop() {
	if s.active {
		s.stop <- true
		s.active = false
	}
}

// Add conversation history
type Conversation struct {
	Messages []ChatMessage
	MaxSize  int
}

func NewConversation() *Conversation {
	return &Conversation{
		Messages: make([]ChatMessage, 0),
		MaxSize:  20, // Keep last 20 messages
	}
}

func (c *Conversation) Add(role, content string) {
	c.Messages = append(c.Messages, ChatMessage{Role: role, Content: content})
	
	// Trim if too long (keep system prompt if exists)
	if len(c.Messages) > c.MaxSize {
		// Keep first message if it's system prompt
		start := 0
		if c.Messages[0].Role == "system" {
			start = 1
			c.Messages = append([]ChatMessage{c.Messages[0]}, c.Messages[start+1:]...)
		} else {
			c.Messages = c.Messages[1:]
		}
	}
}

func (c *Conversation) Clear() {
	c.Messages = make([]ChatMessage, 0)
}

func (c *Conversation) Get() []ChatMessage {
	return c.Messages
}

// Load .env file manually (no external dependencies)
func loadEnv(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err // File doesn't exist, that's ok
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse KEY=VALUE
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		
		// Remove quotes if present
		value = strings.Trim(value, `"'`)
		
		// Set if not already set (env vars take precedence)
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}

	return scanner.Err()
}

// Model represents a free model configuration
type Model struct {
	ID       string
	Name     string
	Provider string
	Endpoint string
	APIKey   string
}

// Request/Response structures
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

// Available models (API keys loaded from .env or env vars)
// var models = map[string]Model{
// 	"mistral": {
// 		ID:       "mistral-small-latest",
// 		Name:     "Mistral Small 3.1",
// 		Provider: "Mistral",
// 		Endpoint: "https://api.mistral.ai/v1/chat/completions",
// 	},
// 	"gemma": {
// 		ID:       "gemma-3-4b-it",
// 		Name:     "Gemma 3 4B",
// 		Provider: "Google",
// 		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-4b-it:generateContent",
// 	},
// 	"groq": {
// 		ID:       "llama-3.3-70b-versatile",
// 		Name:     "Llama 3.3 70B (Groq)",
// 		Provider: "Groq",
// 		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
// 	},
// 	"together": {
// 		ID:       "hermes-3-llama-3.1-405b",
// 		Name:     "Hermes 3 405B",
// 		Provider: "Together",
// 		Endpoint: "https://api.together.xyz/v1/chat/completions",
// 	},
// }

var models = map[string]Model{
	// ═══════════════════════════════════════════════════════════════
	// GOOGLE AI (Gemini) - Free tier available
	// Endpoint format: https://generativelanguage.googleapis.com/v1beta/models/{ID}:generateContent
	// ═══════════════════════════════════════════════════════════════
	
	"gemini-2.0-flash": {
		ID:       "gemini-2.0-flash",
		Name:     "Gemini 2.0 Flash",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
	},
	"gemini-2.0-flash-exp": {
		ID:       "gemini-2.0-flash-exp",
		Name:     "Gemini 2.0 Flash Experimental",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
	},
	"gemini-2.0-flash-lite": {
		ID:       "gemini-2.0-flash-lite",
		Name:     "Gemini 2.0 Flash Lite",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent",
	},
	"gemini-1.5-flash": {
		ID:       "gemini-1.5-flash",
		Name:     "Gemini 1.5 Flash",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
	},
	"gemini-1.5-flash-8b": {
		ID:       "gemini-1.5-flash-8b",
		Name:     "Gemini 1.5 Flash 8B",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent",
	},
	"gemini-1.5-pro": {
		ID:       "gemini-1.5-pro",
		Name:     "Gemini 1.5 Pro",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
	},
	"gemini-2.0-flash-thinking": {
		ID:       "gemini-2.0-flash-thinking-exp",
		Name:     "Gemini 2.0 Flash Thinking",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp:generateContent",
	},
	"gemma-3-4b": {
		ID:       "gemma-3-4b-it",
		Name:     "Gemma 3 4B",
		Provider: "Google",
		Endpoint: "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-4b-it:generateContent",
	},
	
	// ═══════════════════════════════════════════════════════════════
	// MISTRAL AI - Free tier available
	// Endpoint: https://api.mistral.ai/v1/chat/completions
	// ═══════════════════════════════════════════════════════════════
	
	"mistral-small": {
		ID:       "mistral-small-latest",
		Name:     "Mistral Small 3.1",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	"mistral-medium": {
		ID:       "mistral-medium-latest",
		Name:     "Mistral Medium",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	"mistral-large": {
		ID:       "mistral-large-latest",
		Name:     "Mistral Large 2",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	"pixtral": {
		ID:       "pixtral-12b-2409",
		Name:     "Pixtral 12B (Vision)",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	"codestral": {
		ID:       "codestral-latest",
		Name:     "Codestral (Code)",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	"ministral-8b": {
		ID:       "ministral-8b-latest",
		Name:     "Ministral 8B",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	"ministral-3b": {
		ID:       "ministral-3b-latest",
		Name:     "Ministral 3B",
		Provider: "Mistral",
		Endpoint: "https://api.mistral.ai/v1/chat/completions",
	},
	
	// ═══════════════════════════════════════════════════════════════
	// GROQ - Free tier: 1,500,000 tokens/day, 20 RPM
	// Endpoint: https://api.groq.com/openai/v1/chat/completions
	// ═══════════════════════════════════════════════════════════════
	
	// Llama 3.3 Models (Recommended)
	"llama-3.3-70b": {
		ID:       "llama-3.3-70b-versatile",
		Name:     "Llama 3.3 70B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"llama-3.3-70b-specdec": {
		ID:       "llama-3.3-70b-specdec",
		Name:     "Llama 3.3 70B SpecDec (Faster)",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// Llama 3.2 Models (Vision + Small)
	"llama-3.2-90b-vision": {
		ID:       "llama-3.2-90b-vision-preview",
		Name:     "Llama 3.2 90B Vision",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"llama-3.2-11b-vision": {
		ID:       "llama-3.2-11b-vision-preview",
		Name:     "Llama 3.2 11B Vision",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"llama-3.2-3b": {
		ID:       "llama-3.2-3b-preview",
		Name:     "Llama 3.2 3B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"llama-3.2-1b": {
		ID:       "llama-3.2-1b-preview",
		Name:     "Llama 3.2 1B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// Llama 3.1 Models (Long Context)
	"llama-3.1-8b": {
		ID:       "llama-3.1-8b-instant",
		Name:     "Llama 3.1 8B Instant",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"llama-3.1-70b": {
		ID:       "llama-3.1-70b-versatile",
		Name:     "Llama 3.1 70B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// Llama 3 Legacy
	"llama3-8b": {
		ID:       "llama3-8b-8192",
		Name:     "Llama 3 8B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"llama3-70b": {
		ID:       "llama3-70b-8192",
		Name:     "Llama 3 70B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// Gemma Models (Google)
	"gemma2-9b": {
		ID:       "gemma2-9b-it",
		Name:     "Gemma 2 9B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// Reasoning Models (DeepSeek, Qwen)
	"deepseek-r1-distill": {
		ID:       "deepseek-r1-distill-llama-70b",
		Name:     "DeepSeek R1 Distill (Llama 70B)",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"qwen-2.5-32b": {
		ID:       "qwen-2.5-32b-instruct",
		Name:     "Qwen 2.5 32B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	"qwen-2.5-coder-32b": {
		ID:       "qwen-2.5-coder-32b-instruct",
		Name:     "Qwen 2.5 Coder 32B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// Mixtral
	"mixtral-8x7b": {
		ID:       "mixtral-8x7b-32768",
		Name:     "Mixtral 8x7B",
		Provider: "Groq",
		Endpoint: "https://api.groq.com/openai/v1/chat/completions",
	},
	
	// ═══════════════════════════════════════════════════════════════
	// TOGETHER AI (Requires credit card - kept for reference)
	// ═══════════════════════════════════════════════════════════════
	
	"together-llama-3.3": {
		ID:       "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		Name:     "Llama 3.3 70B (Together)",
		Provider: "Together",
		Endpoint: "https://api.together.xyz/v1/chat/completions",
	},
	"together-hermes-405b": {
		ID:       "hermes-3-llama-3.1-405b",
		Name:     "Hermes 3 405B",
		Provider: "Together",
		Endpoint: "https://api.together.xyz/v1/chat/completions",
	},
}

var envKeys = map[string]string{
	// Google
	"gemini-2.0-flash":          "GOOGLE_API_KEY",
	"gemini-2.0-flash-exp":      "GOOGLE_API_KEY",
	"gemini-2.0-flash-lite":     "GOOGLE_API_KEY",
	"gemini-1.5-flash":          "GOOGLE_API_KEY",
	"gemini-1.5-flash-8b":       "GOOGLE_API_KEY",
	"gemini-1.5-pro":            "GOOGLE_API_KEY",
	"gemini-2.0-flash-thinking": "GOOGLE_API_KEY",
	"gemma-3-4b":                "GOOGLE_API_KEY",
	
	// Mistral
	"mistral-small":  "MISTRAL_API_KEY",
	"mistral-medium": "MISTRAL_API_KEY",
	"mistral-large":  "MISTRAL_API_KEY",
	"pixtral":        "MISTRAL_API_KEY",
	"codestral":      "MISTRAL_API_KEY",
	"ministral-8b":   "MISTRAL_API_KEY",
	"ministral-3b":   "MISTRAL_API_KEY",
	
	// Groq
	"llama-3.3-70b":           "GROQ_API_KEY",
	"llama-3.3-70b-specdec":   "GROQ_API_KEY",
	"llama-3.2-90b-vision":    "GROQ_API_KEY",
	"llama-3.2-11b-vision":    "GROQ_API_KEY",
	"llama-3.2-3b":            "GROQ_API_KEY",
	"llama-3.2-1b":            "GROQ_API_KEY",
	"llama-3.1-8b":            "GROQ_API_KEY",
	"llama-3.1-70b":           "GROQ_API_KEY",
	"llama3-8b":               "GROQ_API_KEY",
	"llama3-70b":              "GROQ_API_KEY",
	"gemma2-9b":               "GROQ_API_KEY",
	"deepseek-r1-distill":     "GROQ_API_KEY",
	"qwen-2.5-32b":            "GROQ_API_KEY",
	"qwen-2.5-coder-32b":      "GROQ_API_KEY",
	"mixtral-8x7b":            "GROQ_API_KEY",
	
	// Together
	"together-llama-3.3": "TOGETHER_API_KEY",
	"together-hermes-405b": "TOGETHER_API_KEY",
}


func updateModelKey(name, envKey string) {
   if m, ok := models[name]; ok {
   	m.APIKey = os.Getenv(envKey)
   	models[name] = m
   }
}

func main() {
	// Load .env file first
	if err := loadEnv(".env"); err != nil && !os.IsNotExist(err) {
		fmt.Printf("Warning: error loading .env: %v\n", err)
	}

	// Load API keys into models
	// models["mistral"].APIKey = os.Getenv("MISTRAL_API_KEY")
	// models["gemma"].APIKey = os.Getenv("GOOGLE_API_KEY")
	// models["groq"].APIKey = os.Getenv("GROQ_API_KEY")
	// models["together"].APIKey = os.Getenv("TOGETHER_API_KEY")
	for key, envVar := range envKeys {
		updateModelKey(key, envVar)
	}
	// updateModelKey("mistral", "MISTRAL_API_KEY")
	// updateModelKey("gemma", "GOOGLE_API_KEY")
	// updateModelKey("groq", "GROQ_API_KEY")
	// updateModelKey("together", "TOGETHER_API_KEY")
	// Check available models
	// available := getAvailableModels()
	// if len(available) == 0 {
	// 	fmt.Println("No API keys found!")
	// 	fmt.Println("\nCreate a .env file with:")
	// 	fmt.Println("  MISTRAL_API_KEY=your_key_here")
	// 	fmt.Println("  GOOGLE_API_KEY=your_key_here")
	// 	fmt.Println("  GROQ_API_KEY=your_key_here")
	// 	fmt.Println("  TOGETHER_API_KEY=your_key_here")
	// 	fmt.Println("\nOr set environment variables directly.")
	// 	os.Exit(1)
	// }

	// fmt.Println("=== Simple LLM Client ===")
	// fmt.Println("\nAvailable models:")
	// for i, m := range available {
	// 	fmt.Printf("  %d. %s (%s)\n", i+1, m.Name, m.Provider)
	// }
	fmt.Println(ColorBold + ColorCyan + "=== Simple LLM Client ===" + ColorReset)
	fmt.Println()

	// Show ALL models, grey out unavailable
	fmt.Println("Available models:")
	// modelOrder := []string{"mistral", "gemma", "groq", "together"}
	modelOrder := []string{
		// Google/Gemini models
		"gemini-2.0-flash",
		"gemini-2.0-flash-exp",
		"gemini-2.0-flash-lite",
		"gemini-1.5-flash",
		"gemini-1.5-flash-8b",
		"gemini-1.5-pro",
		"gemini-2.0-flash-thinking",
		"gemma-3-4b",
		// Mistral models
		"mistral-small",
		"mistral-medium",
		"mistral-large",
		"pixtral",
		"codestral",
		"ministral-8b",
		"ministral-3b",
		// Groq models (most popular first)
		"llama-3.3-70b",
		"llama-3.3-70b-specdec",
		"llama-3.2-90b-vision",
		"llama-3.2-11b-vision",
		"llama-3.1-8b",
		"llama-3.1-70b",
		"deepseek-r1-distill",
		"qwen-2.5-coder-32b",
		"qwen-3-32b",
		"mixtral-8x7b",
		// Together (optional)
		"together-llama-3.3",
		"together-hermes-405b",
	}

	// for i, key := range modelOrder {
	// 	m := models[key]
	// 	hasKey := m.APIKey != ""
		
	// 	num := fmt.Sprintf("%d.", i+1)
	// 	name := m.Name
	// 	provider := fmt.Sprintf("(%s)", m.Provider)
		
	// 	if hasKey {
	// 		// Available - green
	// 		fmt.Printf("  %s%s%s %s %s%s%s\n", 
	// 			ColorGreen, num, ColorReset,
	// 			name,
	// 			ColorGrey, provider, ColorReset)
	// 	} else {
	// 		// Unavailable - grey
	// 		fmt.Printf("  %s%s. %s %s [set %s]%s\n",
	// 			ColorGrey, num, name, provider, m.APIKey, ColorReset)
	// 	}
	// }
	// Create ordered list of available models (only those with keys)
	var available []Model
	var availableKeys []string // track which keys are available
	
	for _, key := range modelOrder {
		m, ok := models[key]
		if !ok {
			continue
		}
		if m.APIKey != "" {
			available = append(available, m)
			availableKeys = append(availableKeys, key)
		}
	}

	displayIdx := 1
	for _, key := range modelOrder {
		m, ok := models[key]
		if !ok {
			continue // Skip if model doesn't exist in map
		}
		
		hasKey := m.APIKey != ""
		num := fmt.Sprintf("%d.", displayIdx)
		name := m.Name
		provider := fmt.Sprintf("(%s)", m.Provider)
		
		if hasKey {
			fmt.Printf("  %s%s%s %s %s%s%s\n", 
				ColorGreen, num, ColorReset,
				name,
				ColorGrey, provider, ColorReset)
			displayIdx++
		} else {
			// Don't show unavailable models in numbered list, or show greyed out
			fmt.Printf("  %s[ ] %s %s — set %s%s\n",
				ColorGrey, name, provider, envKeys[key], ColorReset)
		}
	}

	// Get available for selection
	// available := getAvailableModels()
	if len(available) == 0 {
		fmt.Println(ColorRed + "\nNo API keys configured!" + ColorReset)
		fmt.Println("\nCreate " + ColorBold + ".env" + ColorReset + " file:")
		fmt.Println("  " + ColorGrey + "MISTRAL_API_KEY=..." + ColorReset)
		fmt.Println("  " + ColorGrey + "GOOGLE_API_KEY=..." + ColorReset)
		fmt.Println("  " + ColorGrey + "GROQ_API_KEY=..." + ColorReset)
		fmt.Println("  " + ColorGrey + "TOGETHER_API_KEY=..." + ColorReset)
		os.Exit(1)
	}
	// Select model
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nSelect model (number): ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	idx := 0
	fmt.Sscanf(input, "%d", &idx)
	if idx < 1 || idx > len(available) {
		fmt.Println("Invalid selection")
		os.Exit(1)
	}
	selected := available[idx-1]

	conv := NewConversation() // Create conversation history for selected model

	fmt.Printf("\nUsing: %s\n", selected.Name)
	fmt.Println("Type your prompts below. Type 'quit' to exit, 'models' to switch.")
	fmt.Println()

	spinner := NewSpinner()
	// Chat loop
	for {
		fmt.Print(ColorBold + "> " + ColorReset)
		prompt, _ := reader.ReadString('\n')
		prompt = strings.TrimSpace(prompt)

		if prompt == "quit" || prompt == "exit" {
			break
		}
		if prompt == "models" {
			main() // Restart
			return
		}
		if prompt == "clear" {
			fmt.Print("\033[H\033[2J")
			continue
		}
		if prompt == "reset" {
			conv.Clear()
			fmt.Println("Conversation history cleared.")
			continue
		}
		if prompt == "" {
			continue
		}

		conv.Add("user", prompt)
		// Send request
		spinner.Start("thinking")
		
		start := time.Now()
		// response, err := sendRequest(selected, prompt)
		response, err := sendRequestWithHistory(selected, conv.Get())
		elapsed := time.Since(start)
		
		spinner.Stop()

		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		conv.Add("assistant", response)

		fmt.Printf("\n%s%s%s\n", ColorBold, selected.Name, ColorReset)
		fmt.Printf("%s\n", response)
		fmt.Printf("\n%s[%.2fs | %s]%s\n\n", 
			ColorGrey, elapsed.Seconds(), selected.Provider, ColorReset)
	}
}

// New function that sends full conversation history
func sendRequestWithHistory(model Model, history []ChatMessage) (string, error) {
	switch model.Provider {
	case "Mistral", "Groq", "Together":
		return sendOpenAIWithHistory(model, history)
	case "Google":
		return sendGoogleWithHistory(model, history)
	default:
		return "", fmt.Errorf("unknown provider: %s", model.Provider)
	}
}

func sendOpenAIWithHistory(model Model, history []ChatMessage) (string, error) {
	reqBody := ChatRequest{
		Model:    model.ID,
		Messages: history, // Send full history
		Stream:   false,
	}

	jsonData, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", model.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+model.APIKey)

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
		return "", fmt.Errorf("no response")
	}

	return result.Choices[0].Message.Content, nil
}

// Google Gemini with history (different format)
// func sendGoogleWithHistory(model Model, history []ChatMessage) (string, error) {
// 	type Part struct {
// 		Text string `json:"text"`lear
// 	}
// 	type Content struct {
// 		Role  string `json:"role,omitempty"`
// 		Parts []Part `json:"parts"`
// 	}
// 	type Request struct {
// 		Contents []Content `json:"contents"`
// 	}

// 	// Convert history to Google format
// 	var contents []Content
// 	for _, msg := range history {
// 		role := msg.Role
// 		if role == "assistant" {
// 			role = "model" // Google uses "model" not "assistant"
// 		}
// 		contents = append(contents, Content{
// 			Role:  role,
// 			Parts: []Part{{Text: msg.Content}},
// 		})
// 	}

// 	reqBody := Request{Contents: contents}
// 	jsonData, _ := json.Marshal(reqBody)

// 	url := fmt.Sprintf("%s?key=%s", model.Endpoint, model.APIKey)
// 	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
// 	if err != nil {
// 		return "", err
// 	}

// 	req.Header.Set("Content-Type", "application/json")

// 	client := &http.Client{Timeout: 60 * time.Second}
// 	resp, err := client.Do(req)
// 	if err != nil {
// 		return "", err
// 	}
// 	defer resp.Body.Close()

// 	body, _ := io.ReadAll(resp.Body)

// 	if resp.StatusCode != 200 {
// 		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
// 	}

// 	var result struct {
// 		Candidates []struct {
// 			Content struct {
// 				Parts []struct {
// 					Text string `json:"text"`
// 				} `json:"parts"`
// 			} `json:"content"`
// 		} `json:"candidates"`
// 		Error *struct {
// 			Message string `json:"message"`
// 		} `json:"error"`
// 	}

// 	if err := json.Unmarshal(body, &result); err != nil {
// 		return "", err
// 	}

// 	if result.Error != nil {
// 		return "", fmt.Errorf("API error: %s", result.Error.Message)
// 	}

// 	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
// 		return "", fmt.Errorf("no response")
// 	}

// 	return result.Candidates[0].Content.Parts[0].Text, nil
// }

func sendGoogleWithHistory(model Model, history []ChatMessage) (string, error) {
	type Part struct {
		Text string `json:"text"`
	}
	type Content struct {
		Role  string `json:"role,omitempty"`
		Parts []Part `json:"parts"`
	}
	type Request struct {
		Contents []Content `json:"contents"`
	}

	var contents []Content
	var systemPrompt string
	
	for _, msg := range history {
		role := msg.Role
		content := msg.Content
		
		// Google only accepts "user" and "model"
		if role == "assistant" {
			role = "model"
		} else if role == "system" {
			// Store system prompt to prepend to first user message
			systemPrompt = content
			continue
		}
		
		// Prepend system prompt to first user message
		if role == "user" && systemPrompt != "" {
			content = systemPrompt + "\n\n" + content
			systemPrompt = ""
		}
		
		contents = append(contents, Content{
			Role:  role,
			Parts: []Part{{Text: content}},
		})
	}
	
	// If system prompt wasn't used, add it as a user message
	if systemPrompt != "" {
		contents = append([]Content{{
			Role:  "user",
			Parts: []Part{{Text: systemPrompt}},
		}}, contents...)
	}

	reqBody := Request{Contents: contents}
	jsonData, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("%s?key=%s", model.Endpoint, model.APIKey)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

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

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}
	if result.Error != nil {
		return "", fmt.Errorf("API error: %s", result.Error.Message)
	}
	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no response")
	}
	return result.Candidates[0].Content.Parts[0].Text, nil
}

func getAvailableModels() []Model {
	var available []Model
	for _, m := range models {
		if m.APIKey != "" {
			available = append(available, m)

		}
	}
	return available
}

func sendRequest(model Model, prompt string) (string, error) {
	switch model.Provider {
	case "Mistral", "Groq", "Together":
		return sendOpenAICompatible(model, prompt)
	case "Google":
		return sendGoogle(model, prompt)
	default:
		return "", fmt.Errorf("unknown provider: %s", model.Provider)
	}
}

func sendOpenAICompatible(model Model, prompt string) (string, error) {
	reqBody := ChatRequest{
		Model: model.ID,
		Messages: []ChatMessage{
			{Role: "user", Content: prompt},
		},
		Stream: false,
	}

	jsonData, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", model.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+model.APIKey)

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
		return "", fmt.Errorf("no response")
	}

	return result.Choices[0].Message.Content, nil
}

func sendGoogle(model Model, prompt string) (string, error) {
	type Part struct {
		Text string `json:"text"`
	}
	type Content struct {
		Parts []Part `json:"parts"`
	}
	type Request struct {
		Contents []Content `json:"contents"`
	}

	reqBody := Request{
		Contents: []Content{
			{Parts: []Part{{Text: prompt}}},
		},
	}

	jsonData, _ := json.Marshal(reqBody)

	url := fmt.Sprintf("%s?key=%s", model.Endpoint, model.APIKey)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")

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

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	if result.Error != nil {
		return "", fmt.Errorf("API error: %s", result.Error.Message)
	}

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no response")
	}

	return result.Candidates[0].Content.Parts[0].Text, nil
}