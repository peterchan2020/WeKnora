package service

import (
	"testing"

	"github.com/Tencent/WeKnora/internal/types"
)

func TestShouldIndexTextChunkFiltersOnlyWeakTinyChunks(t *testing.T) {
	tests := []struct {
		name string
		text string
		want bool
	}{
		{name: "punctuation", text: ";", want: false},
		{name: "chinese transition", text: "\u4e8e\u662f", want: false},
		{name: "chinese transition with punctuation", text: "\u56e0\u6b64\u3002", want: false},
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
