package chunker

import (
	"context"
	"strings"
	"testing"
)

type fakeSemanticEmbedder struct{}

func (fakeSemanticEmbedder) BatchEmbed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		switch {
		case strings.Contains(text, "beta"):
			out[i] = []float32{0, 1}
		default:
			out[i] = []float32{1, 0}
		}
	}
	return out, nil
}

func TestSplitSemantic_BreaksAtEmbeddingDistance(t *testing.T) {
	text := "alpha starts here. alpha keeps going. beta begins now. beta keeps going."
	cfg := SplitterConfig{
		ChunkSize:                    200,
		ChunkOverlap:                 0,
		Separators:                   []string{"\n\n", "\n", ". "},
		SemanticBufferSize:           0,
		SemanticBreakpointPercentile: 80,
	}

	chunks, err := SplitSemantic(context.Background(), text, cfg, fakeSemanticEmbedder{})
	if err != nil {
		t.Fatalf("SplitSemantic returned error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d: %#v", len(chunks), chunks)
	}
	if !strings.Contains(chunks[0].Content, "alpha keeps going") || strings.Contains(chunks[0].Content, "beta begins") {
		t.Errorf("first chunk should contain only alpha topic, got %q", chunks[0].Content)
	}
	if !strings.Contains(chunks[1].Content, "beta begins now") {
		t.Errorf("second chunk should contain beta topic, got %q", chunks[1].Content)
	}
}

func TestSplitSemantic_PreservesRuneOffsets(t *testing.T) {
	text := "第一句关于苹果。第二句仍然关于苹果。第三句换到火箭。第四句继续火箭。"
	cfg := SplitterConfig{
		ChunkSize:                    200,
		ChunkOverlap:                 0,
		SemanticBufferSize:           0,
		SemanticBreakpointPercentile: 80,
	}

	chunks, err := SplitSemantic(context.Background(), text, cfg, fakeSemanticEmbedder{})
	if err != nil {
		t.Fatalf("SplitSemantic returned error: %v", err)
	}
	textRunes := []rune(text)
	for i, c := range chunks {
		if c.Start < 0 || c.End > len(textRunes) || c.End < c.Start {
			t.Fatalf("chunk[%d] invalid span [%d,%d)", i, c.Start, c.End)
		}
		if got := string(textRunes[c.Start:c.End]); got != c.Content {
			t.Fatalf("chunk[%d] content does not match original span: got %q want %q", i, c.Content, got)
		}
	}
}
