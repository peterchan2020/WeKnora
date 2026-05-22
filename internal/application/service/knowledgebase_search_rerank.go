package service

import (
	"context"
	"strings"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/types"
)

// rerankSearchCandidates applies a rerank model to over-retrieved candidates
// and returns the top-k results filtered by threshold. If no rerank model is
// configured (rerankModelID == "") the input is returned unchanged.
func (s *knowledgeBaseService) rerankSearchCandidates(
	ctx context.Context,
	candidates []*types.IndexWithScore,
	query string,
	retrievalCfg *types.RetrievalConfig,
	matchCount int,
) []*types.IndexWithScore {
	if retrievalCfg == nil || retrievalCfg.RerankModelID == "" {
		logger.Info(ctx, "Rerank skipped: no rerank model configured in retrieval config")
		return candidates
	}
	if len(candidates) == 0 {
		return candidates
	}

	rerankModelID := retrievalCfg.RerankModelID
	threshold := retrievalCfg.GetEffectiveRerankThreshold()

	reranker, err := s.modelService.GetRerankModel(ctx, rerankModelID)
	if err != nil {
		logger.Warnf(ctx, "Rerank: failed to get rerank model %s, skipping rerank: %v", rerankModelID, err)
		return candidates
	}

	// Build passages from candidate content
	var passages []string
	var rerankable []*types.IndexWithScore
	for _, c := range candidates {
		passage := strings.TrimSpace(c.Content)
		if passage == "" {
			continue
		}
		passages = append(passages, passage)
		rerankable = append(rerankable, c)
	}
	if len(passages) == 0 {
		return candidates
	}

	logger.Infof(ctx, "Rerank: calling model %s with %d candidates, threshold=%.4f", rerankModelID, len(passages), threshold)

	rerankResp, err := reranker.Rerank(ctx, query, passages)
	if err != nil {
		logger.Warnf(ctx, "Rerank: model call failed, skipping rerank: %v", err)
		return candidates
	}

	// Rerank score blending weights (same formula as chat pipeline).
	// Weights sum to 1.0 with no additive bias so the blended score stays
	// in the same [0, 1] range as the rerank model's raw output.
	const rerankWeight = 0.7
	const baseScoreWeight = 0.3
	// Fallback threshold: keep top-1 if its relevance exceeds this value.
	const fallbackMinRelevance = 0.15

	// Filter by threshold and build reranked slice (immutable: create new
	// IndexWithScore copies instead of mutating the original pointers).
	var reranked []*types.IndexWithScore
	for _, rr := range rerankResp {
		if rr.Index >= len(rerankable) {
			continue
		}
		if rr.RelevanceScore < threshold {
			continue
		}
		c := rerankable[rr.Index]
		blended := rerankWeight*rr.RelevanceScore + baseScoreWeight*c.Score
		if blended > 1 {
			blended = 1
		}
		reranked = append(reranked, &types.IndexWithScore{
			ID:              c.ID,
			Content:         c.Content,
			SourceID:        c.SourceID,
			SourceType:      c.SourceType,
			ChunkID:         c.ChunkID,
			KnowledgeID:     c.KnowledgeID,
			KnowledgeBaseID: c.KnowledgeBaseID,
			TagID:           c.TagID,
			Score:           blended,
			MatchType:       c.MatchType,
			IsEnabled:       c.IsEnabled,
		})
	}

	// Fallback: if threshold removed all results but top-1 has a reasonable score, keep it
	if len(reranked) == 0 && len(rerankResp) > 0 && rerankResp[0].RelevanceScore >= fallbackMinRelevance {
		rr := rerankResp[0]
		if rr.Index < len(rerankable) {
			c := rerankable[rr.Index]
			blended := rerankWeight*rr.RelevanceScore + baseScoreWeight*c.Score
			if blended > 1 {
				blended = 1
			}
			reranked = []*types.IndexWithScore{{
				ID:              c.ID,
				Content:         c.Content,
				SourceID:        c.SourceID,
				SourceType:      c.SourceType,
				ChunkID:         c.ChunkID,
				KnowledgeID:     c.KnowledgeID,
				KnowledgeBaseID: c.KnowledgeBaseID,
				TagID:           c.TagID,
				Score:           blended,
				MatchType:       c.MatchType,
				IsEnabled:       c.IsEnabled,
			}}
			logger.Infof(ctx, "Rerank: fallback to top-1 (score=%.4f)", rr.RelevanceScore)
		}
	}

	// Limit to matchCount
	if len(reranked) > matchCount {
		reranked = reranked[:matchCount]
	}

	logger.Infof(ctx, "Rerank: %d candidates → %d results after threshold filtering", len(candidates), len(reranked))
	return reranked
}