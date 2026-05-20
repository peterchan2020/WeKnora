package service

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Tencent/WeKnora/internal/infrastructure/chunker"
	"github.com/Tencent/WeKnora/internal/types"
)

func TestBuildSplitterConfigPreservesSemanticZeroBuffer(t *testing.T) {
	kb := &types.KnowledgeBase{
		ChunkingConfig: types.ChunkingConfig{
			Strategy:                     chunker.StrategyStructureAware,
			SemanticBufferSize:           0,
			SemanticBreakpointPercentile: 0,
		},
	}

	cfg := buildSplitterConfig(kb)
	if cfg.SemanticBufferSize != 0 {
		t.Fatalf("SemanticBufferSize = %d, want explicit zero preserved", cfg.SemanticBufferSize)
	}
	if cfg.SemanticBreakpointPercentile != 0 {
		t.Fatalf("SemanticBreakpointPercentile = %d, want zero left for chunker defaulting", cfg.SemanticBreakpointPercentile)
	}
}

func TestParsedChunksFromChunksCarriesStructureMetadata(t *testing.T) {
	chunks := []chunker.Chunk{
		{
			Content:  "| A | B |\n| --- | --- |\n| 1 | 2 |\n",
			Seq:      0,
			Start:    0,
			End:      36,
			Metadata: map[string]any{"table_count": 1, "table_summary": "A | B"},
		},
	}

	parsed := parsedChunksFromChunks(chunks)
	if len(parsed) != 1 {
		t.Fatalf("parsed chunks = %d, want 1", len(parsed))
	}
	if got := parsed[0].Metadata["table_count"]; got != 1 {
		t.Fatalf("table_count = %v, want 1", got)
	}

	raw := parsedChunkMetadataToJSON(context.Background(), parsed[0].Metadata)
	var stored map[string]any
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("metadata JSON should unmarshal: %v", err)
	}
	if got := stored["table_summary"]; got != "A | B" {
		t.Fatalf("table_summary = %v, want A | B", got)
	}
}

func TestParsedChunkMetadataToJSONDropsUnsupportedMetadata(t *testing.T) {
	raw := parsedChunkMetadataToJSON(context.Background(), map[string]any{
		"table_count": 1,
		"bad":         func() {},
	})
	if raw != nil {
		t.Fatalf("raw metadata = %s, want nil for unsupported metadata value", string(raw))
	}
}

func TestSetDocumentMetadataPreservesStructureMetadata(t *testing.T) {
	chunk := &types.Chunk{
		Metadata: types.JSON(`{"table_count":1,"table_summary":"A | B"}`),
	}
	err := chunk.SetDocumentMetadata(&types.DocumentChunkMetadata{
		GeneratedQuestions: []types.GeneratedQuestion{{ID: "q1", Question: "What is A?"}},
	})
	if err != nil {
		t.Fatalf("SetDocumentMetadata failed: %v", err)
	}

	var stored map[string]any
	if err := json.Unmarshal(chunk.Metadata, &stored); err != nil {
		t.Fatalf("metadata JSON should unmarshal: %v", err)
	}
	if got := stored["table_count"]; got != float64(1) {
		t.Fatalf("table_count = %v, want 1", got)
	}
	if _, ok := stored["generated_questions"]; !ok {
		t.Fatalf("generated_questions missing from metadata: %v", stored)
	}
}

func TestNormalizeSemanticAlias(t *testing.T) {
	tests := []struct {
		name     string
		strategy string
		want     string
	}{
		{name: "semantic maps to structure_aware", strategy: "semantic", want: chunker.StrategyStructureAware},
		{name: "empty is preserved", strategy: "", want: ""},
		{name: "explicit heading is preserved", strategy: chunker.StrategyHeading, want: chunker.StrategyHeading},
		{name: "explicit heuristic is preserved", strategy: chunker.StrategyHeuristic, want: chunker.StrategyHeuristic},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeSemanticAlias(tt.strategy); got != tt.want {
				t.Fatalf("normalizeSemanticAlias(%q) = %q, want %q", tt.strategy, got, tt.want)
			}
		})
	}
}

func TestShouldIndexTextChunkFiltersOnlyWeakTinyChunks(t *testing.T) {
	tests := []struct {
		name string
		text string
		want bool
	}{
		{name: "punctuation", text: ";", want: false},
		{name: "chinese transition", text: "\u4e8e\u662f", want: false},
		{name: "chinese transition with punctuation", text: "\u56e0\u6b64\u3002", want: false},
		{name: "more chinese transition", text: "\u6240\u4ee5", want: false},
		{name: "contrastive chinese transition", text: "\u4e0d\u8fc7", want: false},
		{name: "short acronym", text: "AI", want: true},
		{name: "normal sentence", text: "Key point for retrieval.", want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldIndexTextChunk(&types.Chunk{Content: tt.text})
			if got != tt.want {
				t.Fatalf("shouldIndexTextChunk(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}

func TestBuildParentChildConfigs_ChildOverlap(t *testing.T) {
	tests := []struct {
		name              string
		childSize         int
		baseOverlap       int
		wantChildOverlap  int
	}{
		{name: "default: childSize=384, baseOverlap=128", childSize: 384, baseOverlap: 128, wantChildOverlap: 128},
		{name: "small base overlap: childSize=384, baseOverlap=30", childSize: 384, baseOverlap: 30, wantChildOverlap: 30},
		{name: "large base overlap capped by childSize/3: childSize=200, baseOverlap=128", childSize: 200, baseOverlap: 128, wantChildOverlap: 66},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cc := types.ChunkingConfig{
				ChildChunkSize: tt.childSize,
				ChunkOverlap:   tt.baseOverlap,
			}
			base := chunker.SplitterConfig{ChunkOverlap: tt.baseOverlap}
			_, child := buildParentChildConfigs(cc, base)
			if child.ChunkOverlap != tt.wantChildOverlap {
				t.Errorf("child.ChunkOverlap = %d, want %d", child.ChunkOverlap, tt.wantChildOverlap)
			}
		})
	}
}
