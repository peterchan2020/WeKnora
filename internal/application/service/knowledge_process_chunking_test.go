package service

import (
	"testing"

	"github.com/Tencent/WeKnora/internal/infrastructure/chunker"
	"github.com/Tencent/WeKnora/internal/types"
)

func TestBuildSplitterConfigPreservesSemanticZeroBuffer(t *testing.T) {
	kb := &types.KnowledgeBase{
		ChunkingConfig: types.ChunkingConfig{
			Strategy:                     chunker.StrategySemantic,
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

func TestParentStrategyForSemanticChild(t *testing.T) {
	tests := []struct {
		name     string
		strategy string
		want     string
	}{
		{name: "semantic parent falls back to auto", strategy: chunker.StrategySemantic, want: chunker.StrategyAuto},
		{name: "empty parent falls back to auto", strategy: "", want: chunker.StrategyAuto},
		{name: "explicit heading is preserved", strategy: chunker.StrategyHeading, want: chunker.StrategyHeading},
		{name: "explicit heuristic is preserved", strategy: chunker.StrategyHeuristic, want: chunker.StrategyHeuristic},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parentStrategyForSemanticChild(tt.strategy); got != tt.want {
				t.Fatalf("parentStrategyForSemanticChild(%q) = %q, want %q", tt.strategy, got, tt.want)
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
