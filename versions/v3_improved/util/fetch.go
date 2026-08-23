package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type HTTPClient struct {
	client *http.Client
}

func NewHTTPClient() *HTTPClient {
	return &HTTPClient{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (h *HTTPClient) Get(url string, headers map[string]string) ([]byte, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf(
			"HTTP %d: %s",
			resp.StatusCode,
			string(body),
		)
	}

	return body, nil
}

func main() {
	err := godotenv.Load()
	if err != nil {
		fmt.Println("failed to load .env")
		return
	}

	httpClient := NewHTTPClient()

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("GROQ MODELS")
	fmt.Println(strings.Repeat("=", 60))
	fetchGroq(httpClient)

	fmt.Println()

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("GOOGLE GEMINI MODELS")
	fmt.Println(strings.Repeat("=", 60))
	fetchGoogle(httpClient)

	fmt.Println()

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("MISTRAL MODELS")
	fmt.Println(strings.Repeat("=", 60))
	fetchMistral(httpClient)
}

func fetchGroq(client *HTTPClient) {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		fmt.Println("missing GROQ_API_KEY")
		return
	}

	body, err := client.Get(
		"https://api.groq.com/openai/v1/models",
		map[string]string{
			"Authorization": "Bearer " + apiKey,
		},
	)

	if err != nil {
		fmt.Println(err)
		return
	}

	var result struct {
		Data []struct {
			ID                  string `json:"id"`
			OwnedBy             string `json:"owned_by"`
			ContextWindow       int    `json:"context_window"`
			MaxCompletionTokens int    `json:"max_completion_tokens"`
			Active              bool   `json:"active"`
		} `json:"data"`
	}

	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, m := range result.Data {
		fmt.Printf(
			"• %-45s ctx=%-8d max_out=%-8d owner=%s\n",
			m.ID,
			m.ContextWindow,
			m.MaxCompletionTokens,
			m.OwnedBy,
		)
	}
}

func fetchGoogle(client *HTTPClient) {
	apiKey := os.Getenv("GOOGLE_API_KEY_1")
	if apiKey == "" {
		fmt.Println("missing GOOGLE_API_KEY")
		return
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models?key=%s",
		apiKey,
	)

	body, err := client.Get(url, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	var result struct {
		Models []struct {
			Name                       string   `json:"name"`
			DisplayName                string   `json:"displayName"`
			Description                string   `json:"description"`
			InputTokenLimit            int      `json:"inputTokenLimit"`
			OutputTokenLimit           int      `json:"outputTokenLimit"`
			SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
		} `json:"models"`
	}

	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, m := range result.Models {
		supportsGenerate := false

		for _, method := range m.SupportedGenerationMethods {
			if method == "generateContent" {
				supportsGenerate = true
				break
			}
		}

		if !supportsGenerate {
			continue
		}

		fmt.Printf(
			"• %-45s ctx=%-8d out=%-8d\n",
			m.Name,
			m.InputTokenLimit,
			m.OutputTokenLimit,
		)
	}
}

func fetchMistral(client *HTTPClient) {
	apiKey := os.Getenv("MISTRAL_API_KEY_1")
	if apiKey == "" {
		fmt.Println("missing MISTRAL_API_KEY")
		return
	}

	body, err := client.Get(
		"https://api.mistral.ai/v1/models",
		map[string]string{
			"Authorization": "Bearer " + apiKey,
		},
	)

	if err != nil {
		fmt.Println(err)
		return
	}

	var result struct {
		Data []struct {
			ID          string `json:"id"`
			Object      string `json:"object"`
			Created     int64  `json:"created"`
			OwnedBy     string `json:"owned_by"`
			MaxContext  int    `json:"max_context_length"`
		} `json:"data"`
	}

	err = json.Unmarshal(body, &result)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, m := range result.Data {
		fmt.Printf(
			"• %-45s owner=%s\n",
			m.ID,
			m.OwnedBy,
		)
	}
}