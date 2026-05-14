package chunker

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSplitByHeuristics_FormFeedBoundary(t *testing.T) {
	doc := strings.Repeat("page one body text. ", 30) + "\f" + strings.Repeat("page two body. ", 30)
	cfg := SplitterConfig{ChunkSize: 400, ChunkOverlap: 20, Separators: []string{". "}}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) < 2 {
		t.Fatalf("form feed should produce ≥2 chunks, got %d", len(chunks))
	}
}

func TestSplitByHeuristics_NumberedSections(t *testing.T) {
	body := strings.Repeat("body sentence. ", 8)
	doc := "1. Introduction\n" + body + "\n\n2. Methods\n" + body + "\n\n3. Results\n" + body
	cfg := SplitterConfig{ChunkSize: 200, ChunkOverlap: 20, Separators: []string{". "}}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) < 2 {
		t.Fatalf("numbered sections should split: got %d chunks", len(chunks))
	}
}

func TestSplitByHeuristics_GermanChapterMarkers(t *testing.T) {
	body := strings.Repeat("Beispieltext. ", 10)
	doc := "Kapitel 1: Einführung\n" + body + "\n\nKapitel 2: Hauptteil\n" + body
	cfg := SplitterConfig{ChunkSize: 200, ChunkOverlap: 20, Separators: []string{". "}}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) < 2 {
		t.Fatalf("German chapter markers should split: got %d", len(chunks))
	}
}

func TestSplitByHeuristics_ChineseChapterMarkers(t *testing.T) {
	body := strings.Repeat("内容内容内容。", 60)
	doc := "第一章 引言\n" + body + "\n\n第二章 方法\n" + body
	cfg := SplitterConfig{ChunkSize: 200, ChunkOverlap: 20, Separators: []string{"。"}, Languages: []string{LangChinese}}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) < 2 {
		t.Fatalf("Chinese chapter markers should split: got %d", len(chunks))
	}
}

func TestSplitByHeuristics_FallsThroughForUnstructuredDoc(t *testing.T) {
	doc := strings.Repeat("plain prose without structure. ", 5)
	cfg := SplitterConfig{ChunkSize: 1000, ChunkOverlap: 20}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) != 1 {
		t.Errorf("unstructured short doc should be one chunk, got %d", len(chunks))
	}
}

func TestSplitByHeuristics_OversizeBlockRecursesIntoLegacy(t *testing.T) {
	huge := strings.Repeat("This is a long sentence. ", 200) // ~5000 chars
	doc := "1. Intro\n" + huge
	cfg := SplitterConfig{ChunkSize: 500, ChunkOverlap: 50, Separators: []string{". "}}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) < 5 {
		t.Errorf("oversize block should produce many sub-chunks, got %d", len(chunks))
	}
	// No single chunk should massively exceed the budget.
	for i, c := range chunks {
		if len([]rune(c.Content)) > 2*cfg.ChunkSize {
			t.Errorf("chunk %d exceeds 2x size: %d runes", i, len([]rune(c.Content)))
		}
	}
}

func TestSplitByHeuristics_BoundariesAreOrdered(t *testing.T) {
	doc := "Kapitel 1: A\nbody\n\n---\n\n2. Section B\nbody\n\nPage 3 of 10\n\n第三章 C\nbody"
	bounds := findHeuristicBoundaries(doc, nil)
	if len(bounds) < 2 {
		t.Fatalf("expected multiple boundaries, got %d", len(bounds))
	}
	for i := 1; i < len(bounds); i++ {
		if bounds[i].runeStart < bounds[i-1].runeStart {
			t.Errorf("bounds not sorted: %d before %d", bounds[i].runeStart, bounds[i-1].runeStart)
		}
	}
}

func TestSplitByHeuristics_EmptyText(t *testing.T) {
	if got := splitByHeuristicsImpl("", DefaultConfig(), nil); got != nil {
		t.Errorf("empty doc should be nil, got %v", got)
	}
}

// Regression: applyOverlapAligned previously included curEnd itself in its
// boundary search, and curEnd is always one of the bounds (the bin-packer
// flushes at boundary positions). The function therefore always returned
// curEnd, producing zero overlap regardless of cfg.ChunkOverlap.
func TestSplitByHeuristics_OverlapActuallyOverlaps(t *testing.T) {
	// Build many small numbered sections so the bin-packer flushes mid-doc
	// with at least one earlier boundary inside the overlap window.
	var sb strings.Builder
	for i := 1; i <= 12; i++ {
		sb.WriteString("\n\n")
		sb.WriteByte(byte('0' + i%10))
		sb.WriteString(". ")
		sb.WriteString(strings.Repeat("alpha beta gamma. ", 4)) // ~72 chars / section
	}
	doc := sb.String()

	cfg := SplitterConfig{ChunkSize: 200, ChunkOverlap: 80, Separators: []string{". "}}
	chunks := splitByHeuristicsImpl(doc, cfg, nil)
	if len(chunks) < 2 {
		t.Fatalf("need >=2 chunks to test overlap, got %d", len(chunks))
	}

	// At least one consecutive chunk pair must share a non-trivial suffix /
	// prefix. We don't require *every* pair to overlap (oversize blocks
	// short-circuit through legacy and reset chunkStart), but at least one
	// regular flush boundary should produce real overlap.
	saw := false
	for i := 1; i < len(chunks); i++ {
		prev := strings.TrimSpace(chunks[i-1].Content)
		cur := strings.TrimSpace(chunks[i].Content)
		// Walk back the longest suffix of prev that prefixes cur.
		match := 0
		maxScan := len(prev)
		if len(cur) < maxScan {
			maxScan = len(cur)
		}
		for n := 1; n <= maxScan; n++ {
			if strings.HasPrefix(cur, prev[len(prev)-n:]) {
				match = n
			}
		}
		if match >= 20 {
			saw = true
			break
		}
	}
	if !saw {
		t.Fatalf("expected at least one chunk pair to overlap by >=20 chars, none did. chunk sizes: %v",
			chunkLengths(chunks))
	}
}

// Heuristic boundaries that fall inside protected regions (LaTeX block,
// table, link, etc.) must be dropped so the bin-packer doesn't break
// atomic content. Without the protected-span filter, a numbered-section
// looking line inside a $$...$$ math block would be picked as a boundary.
func TestSplitByHeuristics_DropsBoundariesInsideProtectedSpans(t *testing.T) {
	body := strings.Repeat("filler. ", 30)
	// LaTeX block whose middle line matches NumberedSectionPattern. The
	// filter should drop that boundary so the math block stays intact.
	doc := body + "\n\n$$\nx = 1\n1. equation step one\ny = 2\n$$\n\n" + body

	bounds := findHeuristicBoundaries(doc, nil)
	prot := protectedSpansRune(doc, protectedSpans(doc))
	if len(prot) == 0 {
		t.Fatalf("expected protected spans for doc, got none")
	}
	filtered := dropBoundsInsideSpans(bounds, prot)
	for _, b := range filtered {
		for _, s := range prot {
			if b.runeStart > s.start && b.runeStart < s.end {
				t.Errorf("boundary %d still inside protected span [%d,%d)", b.runeStart, s.start, s.end)
			}
		}
	}
	// And it should actually have removed at least one boundary.
	if len(filtered) >= len(bounds) {
		t.Errorf("filter removed nothing: before=%d after=%d", len(bounds), len(filtered))
	}
}

func chunkLengths(chunks []Chunk) []int {
	out := make([]int, len(chunks))
	for i, c := range chunks {
		out[i] = len([]rune(c.Content))
	}
	return out
}

// ---------------------------------------------------------------------------
// coalesceTinyHeuristicChunks tests
// ---------------------------------------------------------------------------

func TestCoalesceTinyHeuristicChunks_MergesInteriorTinyChunks(t *testing.T) {
	runes := []rune("AAAAAA(2)BBBBBB")
	// Simulate chunks where chunk 1 is tiny: "(2)" = 4 runes
	chunks := []Chunk{
		{Content: "AAAAAA", Seq: 0, Start: 0, End: 6},
		{Content: "(2)", Seq: 1, Start: 6, End: 9},
		{Content: "BBBBBB", Seq: 2, Start: 9, End: 15},
	}
	result := coalesceTinyHeuristicChunks(chunks, runes, 512)
	if len(result) != 2 {
		t.Fatalf("expected 2 chunks after coalescing, got %d", len(result))
	}
	// The tiny chunk should be merged into the previous chunk.
	if result[0].Content != "AAAAAA(2)" {
		t.Errorf("first chunk content = %q, want %q", result[0].Content, "AAAAAA(2)")
	}
	if result[0].Start != 0 || result[0].End != 9 {
		t.Errorf("first chunk offsets = [%d,%d], want [0,9]", result[0].Start, result[0].End)
	}
	if result[1].Content != "BBBBBB" {
		t.Errorf("second chunk content = %q, want %q", result[1].Content, "BBBBBB")
	}
}

func TestCoalesceTinyHeuristicChunks_MergesForwardWhenPrevTooLarge(t *testing.T) {
	runes := []rune(strings.Repeat("A", 200) + "(2)" + strings.Repeat("B", 10))
	chunks := []Chunk{
		{Content: strings.Repeat("A", 200), Seq: 0, Start: 0, End: 200},
		{Content: "(2)", Seq: 1, Start: 200, End: 203},
		{Content: strings.Repeat("B", 10), Seq: 2, Start: 203, End: 213},
	}
	result := coalesceTinyHeuristicChunks(chunks, runes, 200)
	if len(result) != 2 {
		t.Fatalf("expected 2 chunks after coalescing, got %d", len(result))
	}
	// Tiny chunk can't merge backward (200+3 > 200), so it merges forward.
	if !strings.Contains(result[1].Content, "(2)") {
		t.Errorf("second chunk should contain the tiny fragment, got %q", result[1].Content)
	}
	if result[1].Start != 200 {
		t.Errorf("second chunk Start = %d, want 200 (extended backward)", result[1].Start)
	}
}

func TestCoalesceTinyHeuristicChunks_PreservesFirstAndLast(t *testing.T) {
	runes := []rune("abBBBBBBcd")
	chunks := []Chunk{
		{Content: "ab", Seq: 0, Start: 0, End: 2},
		{Content: "BBBBBB", Seq: 1, Start: 2, End: 8},
		{Content: "cd", Seq: 2, Start: 8, End: 10},
	}
	result := coalesceTinyHeuristicChunks(chunks, runes, 512)
	// First and last chunks should not be merged away even if tiny.
	if len(result) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(result))
	}
}

func TestCoalesceTinyHeuristicChunks_OverlappingChunks(t *testing.T) {
	// Simulate overlapping chunks where a tiny chunk overlaps with neighbors.
	runes := []rune("AAAAAA_overlap_(2)_overlap_BBBBBB")
	// chunk 0: runes[0:12], chunk 1 (tiny): runes[8:12], chunk 2: runes[8:30]
	// Overlapping: chunk 1 starts inside chunk 0, chunk 2 starts inside chunk 1
	chunks := []Chunk{
		{Content: string(runes[0:12]), Seq: 0, Start: 0, End: 12},
		{Content: string(runes[8:12]), Seq: 1, Start: 8, End: 12},
		{Content: string(runes[8:30]), Seq: 2, Start: 8, End: 30},
	}
	result := coalesceTinyHeuristicChunks(chunks, runes, 512)
	// Tiny chunk should be merged into previous.
	if len(result) != 2 {
		t.Fatalf("expected 2 chunks after coalescing, got %d", len(result))
	}
	// Verify rune-count invariant: End-Start == runeCount(Content)
	for i, c := range result {
		runeCount := utf8.RuneCountInString(c.Content)
		if c.End-c.Start != runeCount {
			t.Errorf("chunk %d: End-Start=%d but runeCount(Content)=%d", i, c.End-c.Start, runeCount)
		}
	}
}

func TestCoalesceTinyHeuristicChunks_DoesNotMutateInput(t *testing.T) {
	runes := []rune("AAAAAA(2)BBBBBB")
	chunks := []Chunk{
		{Content: "AAAAAA", Seq: 0, Start: 0, End: 6},
		{Content: "(2)", Seq: 1, Start: 6, End: 9},
		{Content: "BBBBBB", Seq: 2, Start: 9, End: 15},
	}
	original := make([]Chunk, len(chunks))
	copy(original, chunks)

	_ = coalesceTinyHeuristicChunks(chunks, runes, 512)

	for i := range chunks {
		if chunks[i].Content != original[i].Content || chunks[i].Seq != original[i].Seq ||
			chunks[i].Start != original[i].Start || chunks[i].End != original[i].End {
			t.Fatalf("coalescing should not mutate input slice at index %d", i)
		}
	}
}

func TestCoalesceTinyHeuristicChunks_ReSequences(t *testing.T) {
	runes := []rune("AAAAAA(2)BBBBBB(3)CCCCCC")
	chunks := []Chunk{
		{Content: "AAAAAA", Seq: 0, Start: 0, End: 6},
		{Content: "(2)", Seq: 1, Start: 6, End: 9},
		{Content: "BBBBBB", Seq: 2, Start: 9, End: 15},
		{Content: "(3)", Seq: 3, Start: 15, End: 18},
		{Content: "CCCCCC", Seq: 4, Start: 18, End: 24},
	}
	result := coalesceTinyHeuristicChunks(chunks, runes, 512)
	for i, c := range result {
		if c.Seq != i {
			t.Errorf("chunk %d Seq = %d, want %d", i, c.Seq, i)
		}
	}
}

// Regression: consecutive tiny chunks that can't merge backward must all be
// preserved by forward-merging into the next non-tiny chunk. The old code
// silently dropped intermediate tiny chunks when more than one appeared in
// a row.
func TestCoalesceTinyHeuristicChunks_ConsecutiveTinyChunks(t *testing.T) {
	// Layout: BIG | tiny | tiny | tiny | BIG
	// The prev chunk is at chunkSize limit so tiny chunks can't merge backward.
	big := strings.Repeat("A", 200)
	runes := []rune(big + "x" + "y" + "z" + strings.Repeat("B", 200))
	chunks := []Chunk{
		{Content: big, Seq: 0, Start: 0, End: 200},
		{Content: "x", Seq: 1, Start: 200, End: 201},
		{Content: "y", Seq: 2, Start: 201, End: 202},
		{Content: "z", Seq: 3, Start: 202, End: 203},
		{Content: strings.Repeat("B", 200), Seq: 4, Start: 203, End: 403},
	}
	result := coalesceTinyHeuristicChunks(chunks, runes, 200)
	if len(result) != 2 {
		t.Fatalf("expected 2 chunks after coalescing, got %d", len(result))
	}
	// The second chunk must contain "xyz" + "BBB..." — no content lost.
	if !strings.Contains(result[1].Content, "xyz") {
		t.Errorf("second chunk lost tiny group content, got %q", result[1].Content)
	}
	// Verify rune-count invariant.
	for i, c := range result {
		runeCount := utf8.RuneCountInString(c.Content)
		if c.End-c.Start != runeCount {
			t.Errorf("chunk %d: End-Start=%d but runeCount=%d", i, c.End-c.Start, runeCount)
		}
	}
}
