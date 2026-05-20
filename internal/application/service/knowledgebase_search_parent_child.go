package service

import (
	"context"
	"slices"

	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/types"
)

// expandToParentChunks replaces child chunk hits with their parent chunks
// when the KB has parent-child chunking enabled. This is critical for
// answer_in_one_chunk: child chunks are too small to contain full answers,
// but parent chunks (which are larger) can. The expansion:
//  1. Fetches child chunk records from DB to resolve ParentChunkID
//  2. Fetches parent chunk records from DB to get parent Content/ChunkID
//  3. Replaces each child IndexWithScore with its parent (keeping the
//     child's retrieval score as the parent's score)
//  4. Deduplicates parents that were hit by multiple children (keeps
//     the highest score)
//
// When EnableParentChild is false, the input is returned unchanged.
func (s *knowledgeBaseService) expandToParentChunks(
	ctx context.Context,
	candidates []*types.IndexWithScore,
	kb *types.KnowledgeBase,
) []*types.IndexWithScore {
	if !kb.ChunkingConfig.EnableParentChild {
		return candidates
	}
	if len(candidates) == 0 {
		return candidates
	}

	tenantID := types.MustTenantIDFromContext(ctx)

	// Collect child chunk IDs to look up
	childIDs := make([]string, 0, len(candidates))
	for _, c := range candidates {
		childIDs = append(childIDs, c.ChunkID)
	}

	// Fetch child chunk records to resolve ParentChunkID
	childChunks, err := s.listChunksByIDWithShared(ctx, tenantID, childIDs)
	if err != nil {
		logger.Warnf(ctx, "expandToParentChunks: failed to fetch child chunks, returning originals: %v", err)
		return candidates
	}

	// Build child chunk map
	childMap := make(map[string]*types.Chunk, len(childChunks))
	for _, c := range childChunks {
		childMap[c.ID] = c
	}

	// Collect parent IDs
	parentIDSet := make(map[string]struct{})
	for _, c := range childChunks {
		if c.ParentChunkID != "" {
			parentIDSet[c.ParentChunkID] = struct{}{}
		}
	}

	// If no parents found, some chunks may be standalone (not parent-child)
	// or the parent-child feature isn't active for this data. Return originals.
	if len(parentIDSet) == 0 {
		logger.Info(ctx, "expandToParentChunks: no parent chunks found, returning original candidates")
		return candidates
	}

	parentIDs := make([]string, 0, len(parentIDSet))
	for id := range parentIDSet {
		parentIDs = append(parentIDs, id)
	}

	// Fetch parent chunk records
	parentChunks, err := s.listChunksByIDWithShared(ctx, tenantID, parentIDs)
	if err != nil {
		logger.Warnf(ctx, "expandToParentChunks: failed to fetch parent chunks, returning originals: %v", err)
		return candidates
	}

	parentMap := make(map[string]*types.Chunk, len(parentChunks))
	for _, p := range parentChunks {
		parentMap[p.ID] = p
	}

	// Build expanded list: replace child with parent, deduplicate
	type parentHit struct {
		score     float64
		matchType types.MatchType
		content   string
		sourceID  string
		kbID      string
		knowID    string
		tagID     string
	}
	parentHits := make(map[string]*parentHit)
	var standalone []*types.IndexWithScore // chunks without parents

	for _, candidate := range candidates {
		child, ok := childMap[candidate.ChunkID]
		if !ok || child.ParentChunkID == "" {
			// No parent — keep as-is (standalone chunk or parent itself)
			standalone = append(standalone, candidate)
			continue
		}

		parent, ok := parentMap[child.ParentChunkID]
		if !ok {
			// Parent not found — keep child as-is
			standalone = append(standalone, candidate)
			continue
		}

		// Replace child with parent, keeping child's score
		existing, exists := parentHits[parent.ID]
		if !exists || candidate.Score > existing.score {
			parentHits[parent.ID] = &parentHit{
				score:     candidate.Score,
				matchType: candidate.MatchType,
				content:   parent.Content,
				sourceID:  candidate.SourceID,
				kbID:      candidate.KnowledgeBaseID,
				knowID:    candidate.KnowledgeID,
				tagID:     candidate.TagID,
			}
		}
	}

	// Build result: parent hits first, then standalone
	var expanded []*types.IndexWithScore
	for parentID, hit := range parentHits {
		parent := parentMap[parentID]
		expanded = append(expanded, &types.IndexWithScore{
			ID:              parent.ID,
			Content:         hit.content,
			SourceID:        hit.sourceID,
			SourceType:      types.ChunkSourceType,
			ChunkID:         parent.ID,
			KnowledgeID:     hit.knowID,
			KnowledgeBaseID: hit.kbID,
			TagID:           hit.tagID,
			Score:           hit.score,
			MatchType:       hit.matchType,
			IsEnabled:       true,
		})
	}
	expanded = append(expanded, standalone...)

	// Sort by score descending
	slices.SortFunc(expanded, sortByScoreDesc)

	logger.Infof(ctx, "expandToParentChunks: %d child candidates → %d parent + %d standalone = %d total",
		len(candidates), len(parentHits), len(standalone), len(expanded))

	return expanded
}