package chatpipeline

import (
	"context"
	"strings"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/models/chat"
	"github.com/Tencent/WeKnora/internal/types"
	"github.com/Tencent/WeKnora/internal/types/interfaces"
)

// RewriteQuery calls an LLM to rewrite a search query for better retrieval.
// It reuses the same prompt templates and JSON parsing logic as
// PluginQueryUnderstand but without the full chat pipeline dependency.
//
// If chatModelID is empty, the query is returned unchanged.
// If the model call fails, the original query is returned as a fallback.
func RewriteQuery(
	ctx context.Context,
	modelService interfaces.ModelService,
	chatModelID string,
	systemPrompt string,
	userPrompt string,
	query string,
) string {
	if chatModelID == "" || query == "" {
		return query
	}

	rewriteModel, err := modelService.GetChatModel(ctx, chatModelID)
	if err != nil {
		logger.Warnf(ctx, "RewriteQuery: failed to get chat model %s, using original query: %v", chatModelID, err)
		return query
	}

	// Build prompt with placeholder replacement (same as PluginQueryUnderstand.buildPrompts)
	queryContent := query + "\n\n<no_image_attached />\n<no_document_attached />"
	vals := types.PlaceholderValues{
		"conversation": "",
		"query":        queryContent,
	}
	renderedSystem := types.RenderPromptPlaceholders(systemPrompt, vals)
	renderedUser := types.RenderPromptPlaceholders(userPrompt, vals)

	thinking := false
	response, err := rewriteModel.Chat(ctx, []chat.Message{
		{Role: "system", Content: renderedSystem},
		{Role: "user", Content: renderedUser},
	}, &chat.ChatOptions{
		Temperature:         0.3,
		MaxCompletionTokens: 150,
		Thinking:            &thinking,
	})
	if err != nil {
		logger.Warnf(ctx, "RewriteQuery: model call failed, using original query: %v", err)
		return query
	}

	// Parse structured JSON output (reuse existing parser)
	if output, ok := parseStructuredQueryOutput(response.Content); ok {
		if rewrite := strings.TrimSpace(output.RewriteQuery); rewrite != "" {
			logger.Infof(ctx, "RewriteQuery: %q → %q", query, rewrite)
			return rewrite
		}
	}

	// Fallback: treat raw text as rewritten query
	content := strings.TrimSpace(response.Content)
	if content != "" {
		logger.Infof(ctx, "RewriteQuery (raw fallback): %q → %q", query, content)
		return content
	}

	return query
}
