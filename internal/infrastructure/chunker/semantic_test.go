package chunker

import (
	"context"
	"fmt"
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

type recordingSemanticEmbedder struct {
	batchSizes []int
}

func (e *recordingSemanticEmbedder) BatchEmbed(_ context.Context, texts []string) ([][]float32, error) {
	e.batchSizes = append(e.batchSizes, len(texts))
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{1, 0}
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

func TestSplitSemantic_BatchesEmbeddingWindows(t *testing.T) {
	var sentences []string
	for i := 0; i < semanticEmbeddingBatchSize+7; i++ {
		sentences = append(sentences, fmt.Sprintf("sentence %d stays on topic.", i))
	}
	embedder := &recordingSemanticEmbedder{}
	cfg := SplitterConfig{
		ChunkSize:                    1000,
		ChunkOverlap:                 0,
		SemanticBufferSize:           0,
		SemanticBreakpointPercentile: 80,
	}

	if _, err := SplitSemantic(context.Background(), strings.Join(sentences, " "), cfg, embedder); err != nil {
		t.Fatalf("SplitSemantic returned error: %v", err)
	}
	if len(embedder.batchSizes) < 2 {
		t.Fatalf("expected embedding windows to be split into multiple batches, got %v", embedder.batchSizes)
	}
	for _, size := range embedder.batchSizes {
		if size > semanticEmbeddingBatchSize {
			t.Fatalf("embedding batch size exceeded cap %d: %v", semanticEmbeddingBatchSize, embedder.batchSizes)
		}
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

func TestRefineSemantic_AdjustsOffsetsWithinStructuredSegments(t *testing.T) {
	prefix := "preface. "
	segmentText := "alpha starts here. alpha keeps going. beta begins now. beta keeps going."
	cfg := SplitterConfig{
		ChunkSize:                    55,
		ChunkOverlap:                 0,
		Separators:                   []string{"\n\n", "\n", ". "},
		SemanticBufferSize:           0,
		SemanticBreakpointPercentile: 80,
	}
	segments := []Chunk{{
		Content: segmentText,
		Seq:     0,
		Start:   len([]rune(prefix)),
		End:     len([]rune(prefix)) + len([]rune(segmentText)),
	}}

	chunks, err := RefineSemantic(context.Background(), segments, cfg, fakeSemanticEmbedder{})
	if err != nil {
		t.Fatalf("RefineSemantic returned error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 refined chunks, got %d: %#v", len(chunks), chunks)
	}
	full := []rune(prefix + segmentText)
	for i, c := range chunks {
		if got := string(full[c.Start:c.End]); got != c.Content {
			t.Fatalf("chunk[%d] content does not match adjusted full-doc span: got %q want %q", i, c.Content, got)
		}
	}
}
