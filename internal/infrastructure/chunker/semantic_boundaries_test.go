package chunker

import (
	"strings"
	"testing"
)

func TestSemanticSentences_DoesNotSplitOnColonOrSemicolon(t *testing.T) {
	text := "Plan: keep this clause together; it is not a new sentence. Next sentence."
	sentences := semanticSentences(text)
	if len(sentences) != 2 {
		t.Fatalf("expected 2 sentences, got %d: %#v", len(sentences), sentences)
	}
	first := text[sentences[0].start:sentences[0].end]
	if !strings.Contains(first, "together; it") {
		t.Fatalf("first sentence should retain semicolon clause, got %q", first)
	}
	if !strings.Contains(first, "Plan: keep") {
		t.Fatalf("first sentence should retain colon clause, got %q", first)
	}
}

func TestCoalesceTinySemanticChunks_DoesNotMutateInput(t *testing.T) {
	chunks := []Chunk{
		{Content: strings.Repeat("a", 55), Seq: 0, Start: 0, End: 55},
		{Content: " tiny", Seq: 1, Start: 55, End: 60},
		{Content: "bbbbb", Seq: 2, Start: 60, End: 65},
	}
	originalNext := chunks[2]

	_ = coalesceTinySemanticChunks(chunks, nil, SplitterConfig{ChunkSize: 59}, nil)

	if chunks[2].Content != originalNext.Content || chunks[2].Seq != originalNext.Seq || chunks[2].Start != originalNext.Start || chunks[2].End != originalNext.End {
		t.Fatalf("coalescing should not mutate input slice: got Content=%q Seq=%d Start=%d End=%d want Content=%q Seq=%d Start=%d End=%d",
			chunks[2].Content, chunks[2].Seq, chunks[2].Start, chunks[2].End,
			originalNext.Content, originalNext.Seq, originalNext.Start, originalNext.End)
	}
}

func TestMergeSemanticChunks_ReturnsNewChunk(t *testing.T) {
	left := Chunk{Content: "left", Seq: 7, Start: 3, End: 7}
	right := Chunk{Content: "right", Seq: 8, Start: 7, End: 12}

	merged := mergeSemanticChunks(left, right)

	if left.Content != "left" || left.End != 7 {
		t.Fatalf("merge should not mutate left input, got %#v", left)
	}
	if merged.Content != "leftright" || merged.Seq != 7 || merged.Start != 3 || merged.End != 12 {
		t.Fatalf("merged chunk mismatch: %#v", merged)
	}
}

func TestSemanticSentences_DoesNotSplitPunctuationEnumeration(t *testing.T) {
	text := "分隔符里务必同时包含中文标点（。！？）和英文标点（. ! ?），否则只对一种语言生效，效果会很差。"
	sentences := semanticSentences(text)
	if len(sentences) != 1 {
		t.Fatalf("expected punctuation enumeration to stay in one sentence, got %d: %#v", len(sentences), sentences)
	}
	got := string([]rune(text)[sentences[0].start:sentences[0].end])
	if got != text {
		t.Fatalf("sentence = %q, want original text", got)
	}
}

func TestBuildSemanticChunks_CoalescesTinyInteriorFragments(t *testing.T) {
	text := "第一个主题说明检索质量受切分策略影响，并且需要保留完整上下文来避免召回时语义被稀释。短句。第二个主题说明嵌入模型和重排会影响最终召回，也需要避免边界碎片污染向量索引。"
	runes := []rune(text)
	sentences := semanticSentences(text)
	if len(sentences) != 3 {
		t.Fatalf("need three sentences for this regression, got %d: %#v", len(sentences), sentences)
	}

	chunks := buildSemanticChunks(runes, sentences, map[int]bool{0: true, 1: true}, SplitterConfig{
		ChunkSize: 200,
	})
	if len(chunks) != 2 {
		t.Fatalf("expected tiny middle sentence to merge into a neighbor, got %d chunks: %#v", len(chunks), chunks)
	}
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk.Content) == "短句。" {
			t.Fatalf("tiny fragment should not remain standalone: %#v", chunks)
		}
	}
	if !strings.Contains(chunks[0].Content, "短句。") && !strings.Contains(chunks[1].Content, "短句。") {
		t.Fatalf("tiny fragment should be preserved after merge: %#v", chunks)
	}
}
