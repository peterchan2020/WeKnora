package chunker

import (
	"context"
	"errors"
	"math"
	"sort"
	"strings"
)

// SemanticEmbedder is the minimal embedding capability needed by the
// LlamaIndex-style semantic splitter. It intentionally matches
// embedding.Embedder.BatchEmbed without importing the embedding package into
// chunker.
type SemanticEmbedder interface {
	BatchEmbed(ctx context.Context, texts []string) ([][]float32, error)
}

const semanticEmbeddingBatchSize = 10

type sentenceSpan struct {
	start int
	end   int
}

// SplitSemantic implements a LlamaIndex-style semantic breakpoint splitter:
// sentence windows are embedded, adjacent window distances are measured, and
// distances above the configured percentile become chunk boundaries.
//
// It preserves WeKnora's native chunk contract: Content is always a literal
// slice of the original text and Start/End are rune offsets into that text.
func SplitSemantic(ctx context.Context, text string, cfg SplitterConfig, embedder SemanticEmbedder) ([]Chunk, error) {
	if text == "" {
		return nil, nil
	}
	if embedder == nil {
		return nil, errors.New("semantic chunking requires an embedding model")
	}

	cfg = ensureDefaults(cfg)
	sentences := semanticSentences(text)
	if len(sentences) < 2 {
		return SplitText(text, cfg), nil
	}

	// For short documents, the default buffer=1 causes "mixed windows" that
	// straddle topic boundaries — both W[i] and W[i+1] contain sentences from
	// both topics, so dist(W[i], W[i+1]) is small and the real boundary is
	// invisible. When sentences < 10, use buffer=0 so each window is exactly
	// one sentence and adjacent-window distance directly measures sentence-pair
	// semantic shift (LlamaIndex buffer_size=0 is explicitly documented for
	// per-sentence evaluation).
	buffer := cfg.SemanticBufferSize
	if buffer < 0 {
		buffer = 0
	}
	if len(sentences) < 10 {
		buffer = 0
	}

	windows := semanticWindows([]rune(text), sentences, buffer)
	embeddings, err := batchSemanticEmbeddings(ctx, embedder, windows)
	if err != nil {
		return nil, err
	}
	if len(embeddings) != len(windows) {
		return nil, errors.New("semantic chunking embedding count mismatch")
	}

	distances := make([]float64, 0, len(embeddings)-1)
	for i := 0; i+1 < len(embeddings); i++ {
		distances = append(distances, cosineDistance(embeddings[i], embeddings[i+1]))
	}
	if len(distances) == 0 {
		return SplitText(text, cfg), nil
	}

	threshold := percentile(distances, cfg.SemanticBreakpointPercentile)
	breakAfter := make(map[int]bool)
	for i, d := range distances {
		if d > threshold {
			breakAfter[i] = true
		}
	}

	chunks := buildSemanticChunks([]rune(text), sentences, breakAfter, cfg)
	if v := ValidateChunks(chunks, len([]rune(text)), cfg.ChunkSize); !v.OK {
		return SplitText(text, cfg), nil
	}
	return chunks, nil
}

func batchSemanticEmbeddings(ctx context.Context, embedder SemanticEmbedder, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += semanticEmbeddingBatchSize {
		end := start + semanticEmbeddingBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch, err := embedder.BatchEmbed(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		embeddings = append(embeddings, batch...)
	}
	return embeddings, nil
}

func semanticSentences(text string) []sentenceSpan {
	runes := []rune(text)
	protected := protectedSpansRune(text, protectedSpans(text))
	var out []sentenceSpan
	pos := 0
	for _, p := range protected {
		if p.start > pos {
			out = append(out, splitPlainSentences(runes[pos:p.start], pos)...)
		}
		if p.end > p.start {
			out = append(out, sentenceSpan{start: p.start, end: p.end})
		}
		pos = p.end
	}
	if pos < len(runes) {
		out = append(out, splitPlainSentences(runes[pos:], pos)...)
	}
	return out
}

func splitPlainSentences(runes []rune, base int) []sentenceSpan {
	var out []sentenceSpan
	insidePaired := insidePairedPunctuation(runes)
	start := 0
	for i := 0; i < len(runes); i++ {
		if isSentenceBoundary(runes, i, insidePaired) {
			end := i + 1
			if end > start {
				out = append(out, sentenceSpan{start: base + start, end: base + end})
			}
			start = end
		}
	}
	if start < len(runes) {
		out = append(out, sentenceSpan{start: base + start, end: base + len(runes)})
	}
	return out
}

func isSentenceBoundary(runes []rune, i int, insidePaired []bool) bool {
	r := runes[i]
	switch r {
	case '.', '!', '?', '。', '！', '？':
		if insidePaired[i] || hasAdjacentSentencePunctuation(runes, i) {
			return false
		}
		return true
	case '\n':
		return i > 0 && runes[i-1] == '\n'
	}
	return false
}

func insidePairedPunctuation(runes []rune) []bool {
	inside := make([]bool, len(runes))
	var stack []rune
	for pos, r := range runes {
		switch r {
		case '(', '（', '[', '【', '{', '「', '『', '《':
			stack = append(stack, r)
		case ')', '）', ']', '】', '}', '」', '』', '》':
			if len(stack) > 0 && isMatchingClose(stack[len(stack)-1], r) {
				stack = stack[:len(stack)-1]
			}
		}
		inside[pos] = len(stack) > 0
	}
	return inside
}

func isMatchingClose(open, close rune) bool {
	switch open {
	case '(':
		return close == ')'
	case '（':
		return close == '）'
	case '[':
		return close == ']'
	case '【':
		return close == '】'
	case '{':
		return close == '}'
	case '「':
		return close == '」'
	case '『':
		return close == '』'
	case '《':
		return close == '》'
	default:
		return false
	}
}

func hasAdjacentSentencePunctuation(runes []rune, i int) bool {
	for right := i + 1; right < len(runes); right++ {
		if runes[right] == ' ' || runes[right] == '\t' {
			continue
		}
		if isSentencePunctuation(runes[right]) {
			return true
		}
		break
	}
	return false
}

func isSentencePunctuation(r rune) bool {
	switch r {
	case '.', '!', '?', '。', '！', '？':
		return true
	default:
		return false
	}
}

func semanticWindows(runes []rune, sentences []sentenceSpan, buffer int) []string {
	if buffer < 0 {
		buffer = 0
	}
	windows := make([]string, len(sentences))
	for i := range sentences {
		start := i - buffer
		if start < 0 {
			start = 0
		}
		end := i + buffer + 1
		if end > len(sentences) {
			end = len(sentences)
		}
		windows[i] = strings.TrimSpace(string(runes[sentences[start].start:sentences[end-1].end]))
	}
	return windows
}

func buildSemanticChunks(runes []rune, sentences []sentenceSpan, breakAfter map[int]bool, cfg SplitterConfig) []Chunk {
	var chunks []Chunk
	seq := 0
	groupStart := 0
	// Track which chunk boundaries correspond to semantic breakpoints.
	// semanticBreakAfterChunk[j] == true means the boundary between chunks[j]
	// and chunks[j+1] was determined by a semantic embedding distance above
	// threshold — the coalescer must not merge across it.
	semanticBreakAfterChunk := make(map[int]bool)

	for i := range sentences {
		if i == len(sentences)-1 || breakAfter[i] {
			chunks = appendSemanticGroup(chunks, runes, sentences[groupStart].start, sentences[i].end, cfg, &seq)
			if breakAfter[i] {
				semanticBreakAfterChunk[len(chunks)-1] = true
			}
			groupStart = i + 1
		}
	}
	return coalesceTinySemanticChunks(chunks, semanticBreakAfterChunk, cfg)
}

// coalesceTinySemanticChunks merges chunks whose trimmed rune length is below
// the threshold into their nearest neighbor. Unlike the heuristic coalescer,
// this function respects semantic breakpoints: it will never merge two chunks
// across a boundary that was created by a semantic embedding distance above the
// configured percentile threshold.
func coalesceTinySemanticChunks(chunks []Chunk, semanticBreakAfter map[int]bool, cfg SplitterConfig) []Chunk {
	if len(chunks) < 2 {
		return chunks
	}

	chunks = append([]Chunk(nil), chunks...)
	threshold := tinySemanticChunkThreshold(cfg)
	out := make([]Chunk, 0, len(chunks))
	for i := 0; i < len(chunks); i++ {
		chunk := chunks[i]
		if i == 0 || i == len(chunks)-1 || len([]rune(strings.TrimSpace(chunk.Content))) >= threshold {
			out = append(out, chunk)
			continue
		}

		// If there is a semantic breakpoint immediately before this chunk
		// (between chunks[i-1] and chunks[i]), do not merge backwards across it.
		if i > 0 && semanticBreakAfter[i-1] {
			// Try merging forward into the next chunk instead, but only if
			// there is no semantic break after the current chunk either.
			if i+1 < len(chunks) && !semanticBreakAfter[i] {
				chunks[i+1] = mergeSemanticChunks(chunk, chunks[i+1])
				continue
			}
			out = append(out, chunk)
			continue
		}

		// If there is a semantic breakpoint immediately after this chunk,
		// do not merge forward across it.
		if semanticBreakAfter[i] {
			// Try merging backward instead.
			if len(out) > 0 && canMergeSemanticChunks(out[len(out)-1], chunk, cfg) {
				out[len(out)-1] = mergeSemanticChunks(out[len(out)-1], chunk)
				continue
			}
			out = append(out, chunk)
			continue
		}

		// No semantic break on either side — use the default merge logic.
		if len(out) > 0 && canMergeSemanticChunks(out[len(out)-1], chunk, cfg) {
			out[len(out)-1] = mergeSemanticChunks(out[len(out)-1], chunk)
			continue
		}
		if i+1 < len(chunks) {
			chunks[i+1] = mergeSemanticChunks(chunk, chunks[i+1])
			continue
		}
		out = append(out, chunk)
	}

	for i := range out {
		out[i].Seq = i
	}
	return out
}

func tinySemanticChunkThreshold(cfg SplitterConfig) int {
	threshold := 50
	if cfg.ChunkSize > 0 && cfg.ChunkSize/8 > threshold {
		threshold = cfg.ChunkSize / 8
	}
	return threshold
}

func canMergeSemanticChunks(left, right Chunk, cfg SplitterConfig) bool {
	if cfg.ChunkSize <= 0 {
		return true
	}
	return len([]rune(left.Content))+len([]rune(right.Content)) <= cfg.ChunkSize
}

func mergeSemanticChunks(left, right Chunk) Chunk {
	merged := Chunk{
		Content: left.Content + right.Content,
		Seq:     left.Seq,
		Start:   left.Start,
		End:     right.End,
	}
	populateStructuralMetadata(&merged)
	return merged
}

func appendSemanticGroup(out []Chunk, runes []rune, start, end int, cfg SplitterConfig, seq *int) []Chunk {
	if end <= start {
		return out
	}
	raw := string(runes[start:end])
	if strings.TrimSpace(raw) == "" {
		return out
	}
	if len([]rune(raw)) <= cfg.ChunkSize {
		chunk := Chunk{Content: raw, Seq: *seq, Start: start, End: end}
		populateStructuralMetadata(&chunk)
		out = append(out, chunk)
		*seq++
		return out
	}

	subCfg := cfg
	subCfg.Strategy = StrategyLegacy
	for _, sub := range SplitText(raw, subCfg) {
		out = append(out, Chunk{
			Content:   sub.Content,
			Seq:       *seq,
			Start:     start + sub.Start,
			End:       start + sub.End,
			Tables:    sub.Tables,
			Formulas:  sub.Formulas,
			CodeBlocks: sub.CodeBlocks,
			Images:    sub.Images,
			Metadata:  sub.Metadata,
		})
		*seq++
	}
	return out
}

func cosineDistance(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 1
	}
	var dot, na, nb float64
	for i := range a {
		av := float64(a[i])
		bv := float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 1
	}
	sim := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if sim > 1 {
		sim = 1
	} else if sim < -1 {
		sim = -1
	}
	return 1 - sim
}

func percentile(values []float64, pct int) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	if pct <= 0 {
		return sorted[0]
	}
	if pct >= 100 || len(sorted) == 1 {
		return sorted[len(sorted)-1]
	}
	rank := (float64(pct) / 100) * float64(len(sorted)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}
