
# Create the complete working multi-agent system
# I'll write all the files

import os

# 1. router.go - Model Router with fallback
router_go = '''package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

// Model represents a free LLM endpoint
type Model struct {
	Name        string            `json:"name"`
	Provider    string            `json:"provider"`
	Endpoint    string            `json:"endpoint"`
	APIKey      string            `json:"api_key"`
	ModelID     string            `json:"model_id"`
	Format      string            `json:"format"` // "openai", "google"
	Priority    int               `json:"priority"`
	RateLimit   int               `json:"rate_limit"`
	LastUsed    time.Time         `json:"-"`
	SuccessRate float64           `json:"-"`
	TotalCalls  int               `json:"-"`
	FailedCalls int               `json:"-"`
	mu          sync.Mutex        `json:"-"`
}

// Router manages model selection and fallback
type Router struct {
	models []Model
	mu     sync.RWMutex
}

// NewRouter creates a router with available models
func NewRouter() *Router {
	r := &Router{models: make([]Model, 0)}
	
	// Define all free models
	allModels := []Model{
		// Google Gemini - Free tier: 1500 requests/day, 60 RPM
		{
			Name:      "Gemini 2.0 Flash",
			Provider:  "Google",
			Endpoint:  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
			ModelID:   "gemini-2.0-flash",
			Format:    "google",
			Priority:  1,
			RateLimit: 60,
		},
		{
			Name:      "Gemini 2.0 Flash Thinking",
			Provider:  "Google",
			Endpoint:  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp:generateContent",
			ModelID:   "gemini-2.0-flash-thinking-exp",
			Format:    "google",
			Priority:  2,
			RateLimit: 30,
		},
		{
			Name:      "Gemini 1.5 Pro",
			Provider:  "Google",
			Endpoint:  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
			ModelID:   "gemini-1.5-pro",
			Format:    "google",
			Priority:  3,
			RateLimit: 60,
		},
		
		// Groq - Free tier: 1,500,000 tokens/day, 20 RPM
		{
			Name:      "Llama 3.3 70B",
			Provider:  "Groq",
			Endpoint:  "https://api.groq.com/openai/v1/chat/completions",
			ModelID:   "llama-3.3-70b-versatile",
			Format:    "openai",
			Priority:  1,
			RateLimit: 20,
		},
		{
			Name:      "DeepSeek R1 Distill",
			Provider:  "Groq",
			Endpoint:  "https://api.groq.com/openai/v1/chat/completions",
			ModelID:   "deepseek-r1-distill-llama-70b",
			Format:    "openai",
			Priority:  2,
			RateLimit: 20,
		},
		{
			Name:      "Qwen 2.5 Coder 32B",
			Provider:  "Groq",
			Endpoint:  "https://api.groq.com/openai/v1/chat/completions",
			ModelID:   "qwen-2.5-coder-32b-instruct",
			Format:    "openai",
			Priority:  2,
			RateLimit: 20,
		},
		
		// Mistral - Free tier available
		{
			Name:      "Mistral Small 3.1",
			Provider:  "Mistral",
			Endpoint:  "https://api.mistral.ai/v1/chat/completions",
			ModelID:   "mistral-small-latest",
			Format:    "openai",
			Priority:  2,
			RateLimit: 20,
		},
		{
			Name:      "Codestral",
			Provider:  "Mistral",
			Endpoint:  "https://api.mistral.ai/v1/chat/completions",
			ModelID:   "codestral-latest",
			Format:    "openai",
			Priority:  1,
			RateLimit: 20,
		},
		
		// Cerebras - Free tier available (very fast)
		{
			Name:      "Llama 3.3 70B (Cerebras)",
			Provider:  "Cerebras",
			Endpoint:  "https://api.cerebras.ai/v1/chat/completions",
			ModelID:   "llama-3.3-70b",
			Format:    "openai",
			Priority:  1,
			RateLimit: 30,
		},
		
		// SambaNova - Free tier: 40 RPM, 40 RPD, 200K TPD
		{
			Name:      "Llama 3.3 70B (SambaNova)",
			Provider:  "SambaNova",
			Endpoint:  "https://api.sambanova.ai/v1/chat/completions",
			ModelID:   "Meta-Llama-3.3-70B-Instruct",
			Format:    "openai",
			Priority:  2,
			RateLimit: 40,
		},
		{
			Name:      "DeepSeek R1 (SambaNova)",
			Provider:  "SambaNova",
			Endpoint:  "https://api.sambanova.ai/v1/chat/completions",
			ModelID:   "DeepSeek-R1-Distill-Llama-70B",
			Format:    "openai",
			Priority:  3,
			RateLimit: 40,
		},
		
		// AI21 - Free trial: $10 credit, 10 RPS
		{
			Name:      "Jamba 1.5",
			Provider:  "AI21",
			Endpoint:  "https://api.ai21.com/studio/v1/chat/completions",
			ModelID:   "jamba-1.5-large",
			Format:    "openai",
			Priority:  3,
			RateLimit: 20,
		},
	}
	
	// Load API keys from environment
	envKeys := map[string]string{
		"Google":    "GOOGLE_API_KEY",
		"Groq":      "GROQ_API_KEY",
		"Mistral":   "MISTRAL_API_KEY",
		"Cerebras":  "CEREBRAS_API_KEY",
		"SambaNova": "SAMBANOVA_API_KEY",
		"AI21":      "AI21_API_KEY",
	}
	
	for _, m := range allModels {
		if envKey, ok := envKeys[m.Provider]; ok {
			if key := os.Getenv(envKey); key != "" {
				m.APIKey = key
				r.models = append(r.models, m)
			}
		}
	}
	
	// Sort by priority
	sort.Slice(r.models, func(i, j int) bool {
		return r.models[i].Priority < r.models[j].Priority
	})
	
	return r
}

// Route sends request to best model, falls back on failure
func (r *Router) Route(messages []Message, tools []Tool) (*Message, error) {
	r.mu.RLock()
	modelsCopy := make([]Model, len(r.models))
	copy(modelsCopy, r.models)
	r.mu.RUnlock()
	
	var lastErr error
	
	for i := range modelsCopy {
		model := &modelsCopy[i]
		
		// Check rate limit
		model.mu.Lock()
		if time.Since(model.LastUsed) < time.Minute/time.Duration(model.RateLimit) {
			model.mu.Unlock()
			continue
		}
		model.LastUsed = time.Now()
		model.TotalCalls++
		model.mu.Unlock()
		
		resp, err := r.callModel(model, messages, tools)
		if err == nil {
			return resp, nil
		}
		
		lastErr = err
		model.mu.Lock()
		model.FailedCalls++
		model.SuccessRate = float64(model.TotalCalls-model.FailedCalls) / float64(model.TotalCalls)
		model.mu.Unlock()
		
		fmt.Printf("⚠️  Model %s failed: %v, trying next...\\n", model.Name, err)
	}
	
	return nil, fmt.Errorf("all models failed, last error: %v", lastErr)
}

func (r *Router) callModel(model *Model, messages []Message, tools []Tool) (*Message, error) {
	switch model.Format {
	case "openai":
		return r.callOpenAI(model, messages, tools)
	case "google":
		return r.callGoogle(model, messages, tools)
	default:
		return nil, fmt.Errorf("unknown format: %s", model.Format)
	}
}

func (r *Router) callOpenAI(model *Model, messages []Message, tools []Tool) (*Message, error) {
	reqBody := map[string]interface{}{
		"model":       model.ModelID,
		"messages":    messages,
		"temperature": 0.7,
		"max_tokens":  4096,
	}
	
	if len(tools) > 0 {
		reqBody["tools"] = tools
		reqBody["tool_choice"] = "auto"
	}
	
	jsonData, _ := json.Marshal(reqBody)
	
	req, err := http.NewRequest("POST", model.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+model.APIKey)
	
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	body, _ := io.ReadAll(resp.Body)
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}
	
	var result struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}
	
	if result.Error != nil {
		return nil, fmt.Errorf("API error: %s", result.Error.Message)
	}
	
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("no response")
	}
	
	return &result.Choices[0].Message, nil
}

func (r *Router) callGoogle(model *Model, messages []Message, tools []Tool) (*Message, error) {
	type Part struct {
		Text string `json:"text,omitempty"`
	}
	type Content struct {
		Role  string `json:"role,omitempty"`
		Parts []Part `json:"parts"`
	}
	
	var contents []Content
	var systemPrompt string
	
	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
			continue
		}
		
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		
		content := msg.Content
		if role == "user" && systemPrompt != "" {
			content = systemPrompt + "\\n\\n" + content
			systemPrompt = ""
		}
		
		contents = append(contents, Content{
			Role:  role,
			Parts: []Part{{Text: content}},
		})
	}
	
	reqBody := map[string]interface{}{
		"contents": contents,
		"generationConfig": map[string]interface{}{
			"temperature":     0.7,
			"maxOutputTokens": 4096,
		},
	}
	
	jsonData, _ := json.Marshal(reqBody)
	url := fmt.Sprintf("%s?key=%s", model.Endpoint, model.APIKey)
	
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	
	req.Header.Set("Content-Type", "application/json")
	
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	body, _ := io.ReadAll(resp.Body)
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
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
		return nil, err
	}
	
	if result.Error != nil {
		return nil, fmt.Errorf("API error: %s", result.Error.Message)
	}
	
	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no response")
	}
	
	return &Message{
		Role:    "assistant",
		Content: result.Candidates[0].Content.Parts[0].Text,
	}, nil
}

// GetStats returns router statistics
func (r *Router) GetStats() []map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	stats := make([]map[string]interface{}, len(r.models))
	for i, m := range r.models {
		m.mu.Lock()
		stats[i] = map[string]interface{}{
			"name":         m.Name,
			"provider":     m.Provider,
			"priority":     m.Priority,
			"success_rate": m.SuccessRate,
			"total_calls":  m.TotalCalls,
			"failed_calls": m.FailedCalls,
		}
		m.mu.Unlock()
	}
	return stats
}
'''

with open('./agentic/router.go', 'w') as f:
    f.write(router_go)

print("✓ router.go created")
