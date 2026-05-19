package asr

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Tencent/WeKnora/internal/logger"
	secutils "github.com/Tencent/WeKnora/internal/utils"
)

const (
	// AliyunASRBaseURL 阿里云 DashScope ASR 的默认 BaseURL（OpenAI 兼容模式）
	AliyunASRBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

	// aliyunASRMaxBase64Size Qwen ASR 同步接口 Base64 编码后音频的最大大小（10MB）
	aliyunASRMaxBase64Size = 10 * 1024 * 1024
)

// AliyunASR implements ASR via Aliyun DashScope OpenAI-Compatible chat/completions API
// with input_audio content type (Qwen ASR models: qwen3-asr-flash, etc.)
type AliyunASR struct {
	modelName     string
	modelID       string
	apiKey        string
	baseURL       string
	language      string
	client        *http.Client
	customHeaders map[string]string
}

// SetCustomHeaders 设置用户自定义 HTTP 请求头（类似 OpenAI Python SDK 的 extra_headers）。
func (a *AliyunASR) SetCustomHeaders(headers map[string]string) {
	a.customHeaders = headers
}

// NewAliyunASR creates an Aliyun DashScope ASR instance.
func NewAliyunASR(config *Config) (*AliyunASR, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = AliyunASRBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")

	httpClient := &http.Client{Timeout: asrDefaultTimeout}

	return &AliyunASR{
		modelName: config.ModelName,
		modelID:   config.ModelID,
		apiKey:    config.APIKey,
		baseURL:   baseURL,
		language:  config.Language,
		client:    httpClient,
	}, nil
}

// Transcribe sends audio bytes to the Aliyun DashScope Qwen ASR API.
func (a *AliyunASR) Transcribe(ctx context.Context, audioBytes []byte, fileName string) (*TranscriptionResult, error) {
	if len(audioBytes) == 0 {
		return nil, fmt.Errorf("audio bytes are empty")
	}

	// Detect audio format and build Base64 Data URI
	ext := DetectAudioFormat(audioBytes, fileName)
	mimeType := extToMIME(ext)
	encoded := base64.StdEncoding.EncodeToString(audioBytes)
	dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, encoded)

	if len(dataURI) > aliyunASRMaxBase64Size {
		return nil, fmt.Errorf("base64-encoded audio exceeds 10MB limit (%d bytes); use a shorter audio clip", len(dataURI))
	}

	// Build request
	asrOpts := aliyunASROptions{EnableITN: true}
	if a.language != "" {
		asrOpts.Language = a.language
	}

	reqBody := aliyunASRRequest{
		Model: a.modelName,
		Messages: []aliyunASRMessage{
			{
				Role: "user",
				Content: []aliyunASRContentItem{
					{
						Type: "input_audio",
						InputAudio: &aliyunASRInputAudio{
							Data: dataURI,
						},
					},
				},
			},
		},
		AsrOptions: &asrOpts,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal ASR request: %w", err)
	}

	url := a.baseURL + "/chat/completions"

	logger.Infof(ctx, "[ASR] Calling Aliyun DashScope ASR API, model=%s, baseURL=%s, audioSize=%d, file=%s",
		a.modelName, a.baseURL, len(audioBytes), fileName)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create ASR request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+a.apiKey)
	secutils.ApplyCustomHeaders(req, a.customHeaders)

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ASR request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read ASR response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errResp aliyunASRErrorResponse
		if json.Unmarshal(body, &errResp) == nil && errResp.Error.Message != "" {
			return nil, fmt.Errorf("Aliyun ASR API error: %s (code: %s)", errResp.Error.Message, errResp.Error.Code)
		}
		return nil, fmt.Errorf("Aliyun ASR API error: HTTP %d, body: %s", resp.StatusCode, string(body))
	}

	var response aliyunASRResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("unmarshal ASR response: %w", err)
	}

	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("Aliyun ASR API returned no choices")
	}

	text := strings.TrimSpace(response.Choices[0].Message.Content)
	logger.Infof(ctx, "[ASR] Transcription completed, text length=%d", len(text))

	return &TranscriptionResult{
		Text: text,
	}, nil
}

func (a *AliyunASR) GetModelName() string { return a.modelName }
func (a *AliyunASR) GetModelID() string   { return a.modelID }

// extToMIME maps a file extension to a MIME type for Base64 Data URI construction.
func extToMIME(ext string) string {
	switch strings.ToLower(ext) {
	case ".mp3":
		return "audio/mpeg"
	case ".wav":
		return "audio/wav"
	case ".flac":
		return "audio/flac"
	case ".ogg":
		return "audio/ogg"
	case ".m4a":
		return "audio/mp4"
	case ".aac":
		return "audio/aac"
	default:
		return "audio/mpeg"
	}
}

// --- Request/Response types for Aliyun DashScope OpenAI-Compatible ASR ---

type aliyunASRRequest struct {
	Model      string             `json:"model"`
	Messages   []aliyunASRMessage `json:"messages"`
	AsrOptions *aliyunASROptions  `json:"asr_options,omitempty"`
}

type aliyunASRMessage struct {
	Role    string               `json:"role"`
	Content []aliyunASRContentItem `json:"content"`
}

type aliyunASRContentItem struct {
	Type       string               `json:"type"`
	InputAudio *aliyunASRInputAudio `json:"input_audio,omitempty"`
	Text       string               `json:"text,omitempty"`
}

type aliyunASRInputAudio struct {
	Data string `json:"data"`
}

type aliyunASROptions struct {
	Language  string `json:"language,omitempty"`
	EnableITN bool   `json:"enable_itn"`
}

type aliyunASRResponse struct {
	Choices []aliyunASRChoice `json:"choices"`
}

type aliyunASRChoice struct {
	Message aliyunASRMessageResponse `json:"message"`
}

type aliyunASRMessageResponse struct {
	Content string `json:"content"`
}

type aliyunASRErrorResponse struct {
	Error struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}
