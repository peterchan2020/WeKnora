// Package chunker implements text splitting for document chunking.
//
// Ported from the Python docreader/splitter/splitter.py recursive text splitter.
package chunker

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/Tencent/WeKnora/internal/infrastructure/docparser"
)

// Chunk represents a piece of split text with position tracking.
//
// Content holds exactly the text from the original document between Start
// and End (rune offsets), so End-Start == utf8.RuneCountInString(Content).
// This invariant is relied on by document-reconstruction code paths
// (knowledge.go:2278+ for summary generation, UI highlighting, etc.).
//
// ContextHeader is a separately-tracked context string (e.g. a Markdown
// heading breadcrumb) that should be prepended at embedding/retrieval time
// but is NOT part of Content. Keeping the two apart preserves the
// position invariant while still letting embedding pipelines see the
// section context.
type Chunk struct {
	Content       string
	ContextHeader string
	Seq           int
	Start         int
	End           int

	// Structural metadata extracted from Content after chunk assembly.
	Tables     []TableElement
	Formulas   []FormulaRef
	CodeBlocks []CodeBlock
	Images     []ImageRef

	// Metadata holds backward-compatible summary counts and key fields
	// for DB persistence (table_count, formula_count, etc.).
	Metadata map[string]any
}

// EmbeddingContent returns the text that should be fed to the embedding
// model — the ContextHeader prepended (when set) plus the chunk content.
// Use this where Content alone would lose semantic context (Tier-1 chunks).
//
// Content is returned verbatim from the source document (the End-Start
// rune-count invariant requires that), but for embedding we trim the
// surrounding whitespace so leading/trailing newlines from boundary slices
// don't dilute the embedded vector or waste tokens. Inner whitespace is
// preserved.
func (c Chunk) EmbeddingContent() string {
	body := strings.TrimSpace(c.Content)
	if c.ContextHeader == "" {
		return body
	}
	return c.ContextHeader + "\n\n" + body
}

// ImageRef is an image reference found within a chunk's content.
type ImageRef struct {
	OriginalRef string
	AltText     string
	Start       int // byte offset within the chunk content
	End         int
}

// SplitterConfig configures the text splitter. Strategy and TokenLimit are
// honored by the strategy entry point in strategy.go; the legacy SplitText
// path uses only ChunkSize/Overlap/Separators.
type SplitterConfig struct {
	ChunkSize    int
	ChunkOverlap int
	Separators   []string

	// Strategy selects an adaptive tier. Empty = legacy (backwards-compatible).
	// See strategy.go for valid values.
	Strategy string
	// TokenLimit caps chunk size in approximate tokens. 0 = use ChunkSize chars.
	TokenLimit int
	// Languages hints multilingual heuristic patterns. Empty = auto-detect.
	Languages []string
	// SemanticBufferSize is the sentence-window radius used by semantic
	// chunking. 1 matches LlamaIndex's SemanticSplitterNodeParser default.
	SemanticBufferSize int
	// SemanticBreakpointPercentile is the percentile of adjacent embedding
	// distances used as the semantic breakpoint threshold.
	SemanticBreakpointPercentile int
}

// Default chunk sizing constants. Single source of truth for the entire
// chunker package and (via knowledge.go::buildSplitterConfig) the
// knowledge service. The frontend KnowledgeBaseEditorModal mirrors these
// numbers in its initial form state — keep them in sync if you change
// either value here.
//
// DefaultChunkSize = 512 chars: ~100–130 English tokens / ~300 Chinese
// tokens. Validated as a strong baseline by the Vecta Feb-2026 benchmark
// across 50 academic papers. Use 200–400 for FAQ-style atomic content,
// 1000–2000 for narrative / argumentative documents.
//
// DefaultChunkOverlap = 80 chars (≈15% of DefaultChunkSize): community-
// recommended sweet spot between recall (an answer split across a
// boundary needs overlap to be retrievable) and storage cost. Use 0 for
// strictly atomic data (FAQ, JSON records), 150–200 for long narratives
// where reasoning crosses chunks.
//
// MIGRATION NOTE: Prior versions had three different overlap defaults
// (Go DefaultConfig: 64, knowledge.go buildSplitterConfig: 50, Python
// docreader: 100). All consolidated to 80 here.
//
// Existing knowledge bases that stored ChunkOverlap=0 in the DB pick
// this 80 up on next re-index; their previously-indexed embeddings will
// not match new ones bit-for-bit. Recall stays similar but search
// ranking can shift slightly. To freeze the old behavior on a per-KB
// basis, explicitly set ChunkingConfig.ChunkOverlap to 64 before
// re-indexing.
const (
	DefaultChunkSize    = 512
	DefaultChunkOverlap = 80
)

// DefaultConfig returns sensible defaults.
func DefaultConfig() SplitterConfig {
	return SplitterConfig{
		ChunkSize:    DefaultChunkSize,
		ChunkOverlap: DefaultChunkOverlap,
		Separators:   []string{"\n\n", "\n", "。"},
	}
}

// protectedPatterns are regex patterns for content that must not be split.
var protectedPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?s)\$\$.*?\$\$`),                                                               // LaTeX block math
	regexp.MustCompile(`(?s)\\\[.*?\\\]`),                                                               // LaTeX display math
	regexp.MustCompile(`(?s)\\\(.*?\\\)`),                                                               // LaTeX inline math
	regexp.MustCompile(`(?s)(^|[^\\$])\$[^$\n]+?\$`),                                                    // LaTeX inline math
	regexp.MustCompile(`(?s)\\begin\{[a-zA-Z]+\*?\}.*?\\end\{[a-zA-Z]+\*?\}`),                           // LaTeX environments
	regexp.MustCompile(`!\[[^\]]*\]\([^)]+\)`),                                                          // Markdown images
	regexp.MustCompile(`\[[^\]]*\]\([^)]+\)`),                                                           // Markdown links
	regexp.MustCompile(`(?is)<table\b[^>]*>.*?</table>`),                                                // HTML tables
	regexp.MustCompile("(?m)[ ]*(?:\\|[^|\\n]*)+\\|[\\r\\n]+\\s*(?:\\|\\s*:?-{3,}:?\\s*)+\\|[\\r\\n]+"), // Table header+separator
	regexp.MustCompile("(?m)[ ]*(?:\\|[^|\\n]*)+\\|[\\r\\n]+"),                                          // Table rows
	regexp.MustCompile("(?s)```(?:\\w+)?[\\r\\n].*?```"),                                                // Fenced code blocks
}

type span struct {
	start, end int
}

// protectedSpansRune converts byte-offset protected spans to rune offsets
// in a single forward pass over text. Used by callers that work in rune
// space (e.g. the heuristic splitter) to avoid choosing chunk boundaries
// that cut through protected content. byteSpans must be sorted by start
// (protectedSpans guarantees this).
func protectedSpansRune(text string, byteSpans []span) []span {
	if len(byteSpans) == 0 {
		return nil
	}
	out := make([]span, 0, len(byteSpans))
	runeIdx := 0
	byteIdx := 0
	for _, s := range byteSpans {
		for byteIdx < s.start && byteIdx < len(text) {
			_, size := utf8.DecodeRuneInString(text[byteIdx:])
			byteIdx += size
			runeIdx++
		}
		startRune := runeIdx
		for byteIdx < s.end && byteIdx < len(text) {
			_, size := utf8.DecodeRuneInString(text[byteIdx:])
			byteIdx += size
			runeIdx++
		}
		out = append(out, span{start: startRune, end: runeIdx})
	}
	return out
}

// protectedSpans finds all non-overlapping protected regions in text.
func protectedSpans(text string) []span {
	type match struct {
		start, end int
	}
	var all []match
	for _, pat := range protectedPatterns {
		locs := pat.FindAllStringIndex(text, -1)
		for _, loc := range locs {
			if loc[1]-loc[0] > 0 {
				all = append(all, match{loc[0], loc[1]})
			}
		}
	}
	if len(all) == 0 {
		return nil
	}

	// Sort by start, then by length descending
	for i := 1; i < len(all); i++ {
		for j := i; j > 0; j-- {
			if all[j].start < all[j-1].start ||
				(all[j].start == all[j-1].start && (all[j].end-all[j].start) > (all[j-1].end-all[j-1].start)) {
				all[j], all[j-1] = all[j-1], all[j]
			} else {
				break
			}
		}
	}

	// Remove overlaps
	var result []span
	lastEnd := 0
	for _, m := range all {
		if m.start >= lastEnd {
			result = append(result, span{m.start, m.end})
			lastEnd = m.end
		}
	}
	return result
}

// splitUnit is a piece of text with its original position.
type splitUnit struct {
	text       string
	start, end int
}

// splitBySeparators splits text by separators in priority order, recursively
// applying the next separator to any piece that is still larger than
// chunkSize. Mirrors the recursive priority semantics of the Python
// reference splitter (docreader/splitter/splitter.py:_split): if `\n\n`
// produces a piece that's still too big, `\n` (and subsequent separators)
// are applied within that piece — not to the whole text.
//
// chunkSize == 0 disables the recursion guard; callers that don't care
// about size budget (e.g. a final mergeUnits-style pass) pass 0.
func splitBySeparators(text string, separators []string, chunkSize int) []string {
	if text == "" || len(separators) == 0 {
		return []string{text}
	}
	if chunkSize > 0 && runeLen(text) <= chunkSize {
		return []string{text}
	}

	for i, sep := range separators {
		if sep == "" {
			continue
		}
		re := regexp.MustCompile("(" + regexp.QuoteMeta(sep) + ")")
		splits := re.Split(text, -1)
		matches := re.FindAllString(text, -1)
		if len(matches) == 0 {
			continue
		}

		var pieces []string
		for j, s := range splits {
			if s != "" {
				pieces = append(pieces, s)
			}
			if j < len(matches) && matches[j] != "" {
				pieces = append(pieces, matches[j])
			}
		}
		if len(pieces) <= 1 {
			continue
		}

		// Recursively split any piece that is still too large with the
		// remaining (lower-priority) separators.
		var out []string
		remaining := separators[i+1:]
		for _, p := range pieces {
			if chunkSize > 0 && runeLen(p) > chunkSize && len(remaining) > 0 {
				out = append(out, splitBySeparators(p, remaining, chunkSize)...)
			} else {
				out = append(out, p)
			}
		}
		return out
	}
	return []string{text}
}

// runeLen returns the number of runes in s.
func runeLen(s string) int {
	return utf8.RuneCountInString(s)
}

func maxChunkSizeForBudget(chunkSize int) int {
	const hardMaxChunkSize = 7500
	if chunkSize <= 0 {
		return hardMaxChunkSize
	}
	budgetMax := chunkSize * 2
	if budgetMax < chunkSize {
		return hardMaxChunkSize
	}
	if budgetMax < hardMaxChunkSize {
		return budgetMax
	}
	return hardMaxChunkSize
}

func forcedSplitEnd(runes []rune, offset, maxSize int) int {
	chunkEnd := offset + maxSize
	if chunkEnd >= len(runes) {
		return len(runes)
	}

	tagSearchStart := chunkEnd - 500
	if tagSearchStart < offset+1 {
		tagSearchStart = offset + 1
	}
	// Prefer splitting at </table> over </tr> so HTML tables stay
	// structurally complete. Search the full window for </table> first.
	for i := chunkEnd; i >= tagSearchStart; i-- {
		if hasASCIITagSuffix(runes, offset, i, "</table>") {
			return i
		}
	}
	for i := chunkEnd; i >= tagSearchStart; i-- {
		if hasASCIITagSuffix(runes, offset, i, "</tr>") ||
			hasASCIITagSuffix(runes, offset, i, "</li>") ||
			hasASCIITagSuffix(runes, offset, i, "</p>") {
			return i
		}
	}

	for i := chunkEnd - 1; i > offset && i > chunkEnd-200; i-- {
		if runes[i] == '\n' || runes[i] == ' ' {
			return i + 1
		}
	}
	return chunkEnd
}

func hasASCIITagSuffix(runes []rune, start, end int, tag string) bool {
	j := end - 1
	for j >= start && unicode.IsSpace(runes[j]) {
		j--
	}
	if j-start+1 < len(tag) {
		return false
	}
	for k := len(tag) - 1; k >= 0; k-- {
		if lowerASCII(runes[j]) != rune(tag[k]) {
			return false
		}
		j--
	}
	return true
}

func lowerASCII(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + ('a' - 'A')
	}
	return r
}

// SplitText splits text into chunks with overlap, respecting protected patterns.
func SplitText(text string, cfg SplitterConfig) []Chunk {
	if text == "" {
		return nil
	}

	chunkSize := cfg.ChunkSize
	chunkOverlap := cfg.ChunkOverlap
	separators := cfg.Separators

	if chunkSize <= 0 {
		chunkSize = 512
	}
	if chunkOverlap < 0 {
		chunkOverlap = 0
	}

	// Step 1: Find protected spans
	protected := protectedSpans(text)

	// Step 2: Split non-protected regions by separators, keep protected as atomic units.
	// chunkSize is forwarded so splitBySeparators can recursively apply lower-priority
	// separators to oversize pieces (Python-parity recursive split).
	units := buildUnitsWithProtection(text, protected, separators, chunkSize)

	// Step 2b: Attach short preceding text to formula units so formulas are never
	// isolated from their explanatory context.
	units = attachFormulaContext(units, chunkSize)

	// Step 3: Merge units into chunks with overlap
	chunks := mergeUnits(units, chunkSize, chunkOverlap)

	// Step 3b: Merge heading-only chunks into their next neighbor so titles
	// are never orphaned as standalone fragments.
	chunks = coalesceOrphanHeadings(chunks, chunkSize)

	// Step 3c: Re-split any chunks that still exceed 1.5x chunkSize at
	// paragraph/sentence boundaries. This catches oversized chunks produced
	// by coalescing or protected-region preservation.
	chunks = resplitOversizedChunks(chunks, chunkSize)

	return chunks
}

// resplitOversizedChunks re-splits any chunk whose Content exceeds 1.5×
// chunkSize at paragraph or sentence boundaries.
func resplitOversizedChunks(chunks []Chunk, chunkSize int) []Chunk {
	if len(chunks) == 0 || chunkSize <= 0 {
		return chunks
	}
	threshold := chunkSize + chunkSize/2 // 1.5x

	out := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		runes := []rune(c.Content)
		if len(runes) <= threshold {
			out = append(out, c)
			continue
		}

		parts := splitOversizedRunes(runes, chunkSize)
		offset := c.Start
		for _, part := range parts {
			partLen := len(part)
			sub := Chunk{
				Content:       string(part),
				ContextHeader: c.ContextHeader,
				Start:         offset,
				End:           offset + partLen,
			}
			populateStructuralMetadata(&sub)
			out = append(out, sub)
			offset += partLen
		}
	}

	for i := range out {
		out[i].Seq = i
	}
	return out
}

// splitOversizedRunes splits an oversized rune slice into pieces that are
// each ≤ chunkSize runes, preferring paragraph then sentence boundaries.
func splitOversizedRunes(runes []rune, chunkSize int) [][]rune {
	if len(runes) <= chunkSize {
		return [][]rune{runes}
	}

	var parts [][]rune
	start := 0
	for start < len(runes) {
		end := findBestSplitPoint(runes, start, chunkSize)
		parts = append(parts, runes[start:end])
		start = end
	}
	return parts
}

// findBestSplitPoint finds the best position to split runes[start:] into a
// chunk of at most chunkSize runes. Prefers \n\n, then \n, then sentence
// boundaries, then a hard cut.
func findBestSplitPoint(runes []rune, start, chunkSize int) int {
	maxEnd := start + chunkSize
	if maxEnd >= len(runes) {
		return len(runes)
	}

	// Paragraph break (\n\n).
	for i := maxEnd; i > start+1; i-- {
		if runes[i-1] == '\n' && runes[i] == '\n' {
			return i + 1
		}
	}

	// Single newline.
	for i := maxEnd; i > start; i-- {
		if runes[i] == '\n' {
			return i + 1
		}
	}

	// Sentence boundary.
	for i := maxEnd; i > start+1; i-- {
		if sentenceEndRunes[runes[i-1]] {
			after := i
			if after < len(runes) && unicode.IsSpace(runes[after]) {
				return after
			}
		}
	}

	// Hard cut.
	return maxEnd
}

// buildUnitsWithProtection splits text into units, preserving protected spans as atomic.
// Start/End positions in the returned units are rune offsets (not byte offsets),
// because downstream merge logic indexes content via []rune slicing.
// If a protected span exceeds maxProtectedSize, it will be forcibly split to prevent
// creating chunks that are too large for downstream processing (e.g., embedding APIs).
// chunkSize is forwarded to splitBySeparators so recursive splitting can keep pieces
// under the budget when one separator alone leaves a piece oversize.
func buildUnitsWithProtection(text string, protected []span, separators []string, chunkSize int) []splitUnit {
	maxProtectedSize := maxChunkSizeForBudget(chunkSize)

	var units []splitUnit
	bytePos := 0
	runePos := 0

	for _, p := range protected {
		if p.start > bytePos {
			pre := text[bytePos:p.start]
			parts := splitBySeparators(pre, separators, chunkSize)
			runeOffset := runePos
			for _, part := range parts {
				partRuneLen := runeLen(part)
				units = append(units, splitUnit{
					text:  part,
					start: runeOffset,
					end:   runeOffset + partRuneLen,
				})
				runeOffset += partRuneLen
			}
			runePos += runeLen(pre)
		}

		protText := text[p.start:p.end]
		protRuneLen := runeLen(protText)

		// If protected content is too large, forcibly split it
		if protRuneLen > maxProtectedSize {
			if codeUnits := splitFencedCodeUnitWithTreeSitter(protText, runePos, maxProtectedSize); len(codeUnits) > 0 {
				units = append(units, codeUnits...)
				runePos += protRuneLen
				bytePos = p.end
				continue
			}

			// Split into smaller chunks at line breaks or spaces
			runes := []rune(protText)
			offset := 0
			for offset < len(runes) {
				chunkEnd := forcedSplitEnd(runes, offset, maxProtectedSize)

				chunkText := string(runes[offset:chunkEnd])
				chunkLen := chunkEnd - offset
				units = append(units, splitUnit{
					text:  chunkText,
					start: runePos + offset,
					end:   runePos + offset + chunkLen,
				})
				offset = chunkEnd
			}
		} else {
			// Normal case: keep protected content as a single unit
			units = append(units, splitUnit{
				text:  protText,
				start: runePos,
				end:   runePos + protRuneLen,
			})
		}
		runePos += protRuneLen
		bytePos = p.end
	}

	if bytePos < len(text) {
		remaining := text[bytePos:]
		parts := splitBySeparators(remaining, separators, chunkSize)
		runeOffset := runePos
		for _, part := range parts {
			partRuneLen := runeLen(part)
			units = append(units, splitUnit{
				text:  part,
				start: runeOffset,
				end:   runeOffset + partRuneLen,
			})
			runeOffset += partRuneLen
		}
	}

	return units
}

// formulaPattern matches text that looks like a LaTeX math formula (block or
// display math). Used by attachFormulaContext to identify formula units.
var formulaPattern = regexp.MustCompile(`(?s)^\s*\$\$.*\$\$\s*$|^\s*\\\[.*\\\]\s*$|^\s*\\begin\{[a-zA-Z]+\*?\}.*\\end\{[a-zA-Z]+\*?\}\s*$`)

// attachFormulaContext merges a short preceding text unit into a formula unit
// so that formulas are never isolated from their explanatory context. When a
// formula unit is preceded by a text unit shorter than 200 runes and the
// combined length fits within chunkSize, the two units are merged into one.
func attachFormulaContext(units []splitUnit, chunkSize int) []splitUnit {
	if len(units) <= 1 || chunkSize <= 0 {
		return units
	}
	maxContext := 200
	out := make([]splitUnit, 0, len(units))
	out = append(out, units[0])
	for i := 1; i < len(units); i++ {
		prev := out[len(out)-1]
		cur := units[i]
		prevLen := runeLen(prev.text)
		curLen := runeLen(cur.text)
		if formulaPattern.MatchString(cur.text) && prevLen <= maxContext && prevLen+curLen <= chunkSize && prev.end == cur.start {
			// Merge: the preceding short text becomes part of the formula unit.
			merged := splitUnit{
				text:  prev.text + cur.text,
				start: prev.start,
				end:   cur.end,
			}
			out[len(out)-1] = merged
			continue
		}
		out = append(out, cur)
	}
	return out
}

// coalesceOrphanHeadings merges heading-only chunks with their next neighbor
// so that standalone title fragments (e.g. "# 4.4Boosting Models with top-$k$
// Sampling" produced by PDF extraction) are never orphaned. When a chunk
// contains only a heading line (plus optional blank lines), the heading text
// is promoted to the next chunk's ContextHeader and the orphan chunk is
// dropped. This runs after mergeUnits so it handles overlap-based chunks
// where coalesceTinyChunks (heading splitter) cannot merge due to
// cur.End != next.Start.
func coalesceOrphanHeadings(chunks []Chunk, chunkSize int) []Chunk {
	if len(chunks) <= 1 || chunkSize <= 0 {
		return chunks
	}

	out := make([]Chunk, 0, len(chunks))
	for i := 0; i < len(chunks); i++ {
		c := chunks[i]
		header, isHeading := pureHeadingChunkHeader(c)
		if !isHeading {
			out = append(out, c)
			continue
		}
		// Heading-only chunk: try to merge into next chunk.
		if i < len(chunks)-1 {
			next := chunks[i+1]
			// Use untrimmed lengths for budget check to avoid the
			// trimmed-length vs actual-content-length mismatch.
			mergedContent := strings.TrimSpace(c.Content) + "\n\n" + strings.TrimSpace(next.Content)
			combinedLen := utf8.RuneCountInString(mergedContent)
			next.ContextHeader = mergeBreadcrumbs(header, next.ContextHeader)
			if combinedLen <= chunkSize {
				// Small enough to also merge Content — absorb heading text into
				// next chunk so the title is visible in the chunk body.
				next.Content = mergedContent
				populateStructuralMetadata(&next)
				out = append(out, next)
				i++ // skip next since we already emitted it
			} else {
				// Too large to merge content — keep heading as ContextHeader only.
				out = append(out, next)
				i++ // skip next since we already emitted it
			}
			continue
		}
		// Last chunk is a heading: merge backward into the previous chunk
		// in the output so the heading isn't orphaned at document end.
		if len(out) > 0 {
			prev := &out[len(out)-1]
			prevLen := utf8.RuneCountInString(prev.Content)
			headingLen := utf8.RuneCountInString(c.Content)
			if prevLen+headingLen+2 <= chunkSize {
				prev.Content = strings.TrimSpace(prev.Content) + "\n\n" + strings.TrimSpace(c.Content)
				prev.End = c.End
				populateStructuralMetadata(prev)
			}
			// If too large to merge, promote heading to prev's ContextHeader.
			prev.ContextHeader = mergeBreadcrumbs(prev.ContextHeader, header)
		} else {
			// No previous chunk to merge into — keep as-is.
			out = append(out, c)
		}
	}

	// Re-sequence.
	for i := range out {
		out[i].Seq = i
	}
	return out
}

// sentenceEndRunes are the punctuation runes that mark a sentence boundary
// when followed by whitespace or end-of-text. Used by snapFlushToSentenceBoundary
// to avoid cutting chunks mid-sentence.
var sentenceEndRunes = map[rune]bool{
	'.':  true,
	'!':  true,
	'?':  true,
	'。': true,
	'！': true,
	'？': true,
	'；': true,
}

// snapFlushToSentenceBoundary tries to align the flush point of the current
// unit accumulation to the last complete sentence, so chunks don't end
// mid-sentence. It walks backward through the accumulated units, looking for
// a sentence-ending rune followed by whitespace or a newline. When found, it
// splits the containing unit at that point: the prefix stays in the current
// chunk, the suffix is returned as carry-over for the next chunk.
//
// If no sentence boundary is found within a reasonable lookback window
// (chunkSize/2 runes), returns the units unchanged (no carry-over) — this
// avoids pathological cases where the entire chunk is one long sentence.
// snapRuneToSentenceEnd searches backward from `end` within runes[start:end]
// for the last sentence-ending punctuation followed by whitespace. Returns the
// snapped end position (inclusive of the boundary rune + trailing spaces), or
// the original `end` if no suitable boundary is found that keeps the chunk at
// least `minSize` runes long. Used by the heuristic splitter's bin-packing to
// avoid flushing mid-sentence.
func snapRuneToSentenceEnd(runes []rune, start, end, minSize int) int {
	if end <= start || end-start < minSize {
		return end
	}

	// Search backward from end-1 for a sentence-ending rune.
	for i := end - 1; i > start; i-- {
		if !sentenceEndRunes[runes[i]] {
			continue
		}
		// Check that the sentence-ending rune is followed by whitespace,
		// a closing quote/paren, or end-of-text.
		after := i + 1
		if after < end {
			r := runes[after]
			if !unicode.IsSpace(r) && r != '"' && r != '\'' && r != ')' && r != ']' && r != '}' && r != '’' && r != '”' {
				continue
			}
		}
		// Advance past the sentence-ending rune and any trailing spaces
		// on the same line. Stop before a newline (it belongs to the
		// next sentence for clean separation).
		boundary := after
		for boundary < end && (runes[boundary] == ' ' || runes[boundary] == '\t') {
			boundary++
		}
		// Ensure the snapped chunk is still at least minSize.
		if boundary-start >= minSize {
			return boundary
		}
		// Snapped position would be too small — don't snap, keep searching.
		// (In practice this means the sentence is very long and we accept
		// cutting it rather than producing a tiny chunk.)
		break
	}
	return end
}

func snapFlushToSentenceBoundary(units []splitUnit, chunkSize int) ([]splitUnit, []splitUnit) {
	if len(units) <= 1 || chunkSize <= 0 {
		return units, nil
	}

	// Compute total rune length to decide lookback budget.
	totalLen := 0
	for _, u := range units {
		totalLen += runeLen(u.text)
	}
	maxLookback := chunkSize / 2
	if maxLookback < 100 {
		maxLookback = 100
	}

	// Walk backward through units accumulating text. When we find a sentence
	// boundary, record the split position. We keep the *last* (closest to end)
	// sentence boundary within the lookback window.
	accumulated := 0
	type splitPoint struct {
		unitIdx int   // index of the unit containing the boundary
		runeOff int   // rune offset within that unit's text (after the boundary)
	}
	var best splitPoint
	bestSet := false

	for i := len(units) - 1; i >= 0 && accumulated < maxLookback; i-- {
		runes := []rune(units[i].text)
		accumulated += len(runes)

		// Scan backward within this unit for a sentence-ending rune.
		for j := len(runes) - 1; j >= 1; j-- {
			if !sentenceEndRunes[runes[j]] {
				continue
			}
			// Sentence boundary: the end-rune must be followed by whitespace
			// or be the last rune in the unit (next unit starts with whitespace
			// or uppercase), or followed by a closing quote/paren.
			after := j + 1
			if after < len(runes) {
				r := runes[after]
				if !unicode.IsSpace(r) && r != '"' && r != '\'' && r != ')' && r != ']' && r != '}' && r != '’' && r != '”' {
					continue
				}
			}
			// Boundary found: split after the sentence-ending rune (and any
			// trailing whitespace on the same line, to keep the newline at the
			// start of the next unit for clean separation).
			splitAt := after
			// Consume trailing spaces on the same line — but not the newline
			// itself, which belongs to the next sentence.
			for splitAt < len(runes) && (runes[splitAt] == ' ' || runes[splitAt] == '\t') {
				splitAt++
			}
			if splitAt < len(runes) && runes[splitAt] == '\n' {
				// Keep the newline with the next chunk for cleaner separation.
				// Don't advance splitAt past the newline.
			}

			if splitAt > 0 && splitAt < len(runes) {
				best = splitPoint{unitIdx: i, runeOff: splitAt}
				bestSet = true
			}
			// We found a boundary in this unit; no need to scan further back
			// within it. But we continue to other units to find a closer one.
			break
		}
	}

	if !bestSet {
		return units, nil
	}

	// Split at the recorded position: units[:best.unitIdx] + prefix of
	// units[best.unitIdx] stay in the current chunk; suffix of
	// units[best.unitIdx] + units[best.unitIdx+1:] carry over.
	su := units[best.unitIdx]
	runes := []rune(su.text)
	prefix := string(runes[:best.runeOff])
	suffix := string(runes[best.runeOff:])

	flushed := make([]splitUnit, best.unitIdx, best.unitIdx+1)
	copy(flushed, units[:best.unitIdx])

	if strings.TrimSpace(prefix) != "" {
		flushed = append(flushed, splitUnit{
			text:  prefix,
			start: su.start,
			end:   su.start + best.runeOff,
		})
	}

	var carry []splitUnit
	if strings.TrimSpace(suffix) != "" {
		carry = append(carry, splitUnit{
			text:  suffix,
			start: su.start + best.runeOff,
			end:   su.end,
		})
	}
	carry = append(carry, units[best.unitIdx+1:]...)

	return flushed, carry
}

// mergeUnits combines split units into chunks with overlap tracking.
// Enforces an absolute maximum chunk size to prevent exceeding downstream limits (e.g., embedding APIs).
// Active contextual headers (e.g., Markdown table headers) are prepended to new
// chunks so that every chunk carries its own header context.
func mergeUnits(units []splitUnit, chunkSize, chunkOverlap int) []Chunk {
	if len(units) == 0 {
		return nil
	}

	absoluteMaxSize := maxChunkSizeForBudget(chunkSize)

	ht := newHeaderTracker()

	var chunks []Chunk
	var current []splitUnit
	curLen := 0

	for _, u := range units {
		uLen := runeLen(u.text)

		// If this single unit exceeds absolute max, force split it further
		if uLen > absoluteMaxSize {
			// Flush current chunk if any
			if len(current) > 0 {
				chunks = append(chunks, buildChunk(current, len(chunks)))
				current = nil
				curLen = 0
			}

			// Update header state even for oversized units
			ht.update(u.text)

			// Split this oversized unit into smaller chunks
			runes := []rune(u.text)
			offset := 0
			for offset < len(runes) {
				chunkEnd := forcedSplitEnd(runes, offset, absoluteMaxSize)

				chunkText := string(runes[offset:chunkEnd])
				chunk := Chunk{
					Content: chunkText,
					Seq:     len(chunks),
					Start:   u.start + offset,
					End:     u.start + chunkEnd,
				}
				populateStructuralMetadata(&chunk)
				chunks = append(chunks, chunk)
				offset = chunkEnd
			}
			continue
		}

		// Update header tracking
		ht.update(u.text)
		headers := ht.getHeaders()
		headersLen := runeLen(headers)
		if headersLen > chunkSize {
			headers = ""
			headersLen = 0
		}

		// If adding this unit (plus reserving space for headers in a potential
		// next chunk) would exceed chunk size, flush the current chunk.
		if curLen+uLen+headersLen > chunkSize && len(current) > 0 {
			// Try to snap the flush boundary to a sentence boundary so we
			// don't cut mid-sentence. Any carry-over units replace the
			// overlap for the next chunk.
			flushed, carry := snapFlushToSentenceBoundary(current, chunkSize)
			if len(flushed) > 0 {
				chunks = append(chunks, buildChunk(flushed, len(chunks)))
			}
			if len(carry) > 0 {
				// Carry-over from sentence snap: use as the next chunk's base.
				current = carry
				curLen = 0
				for _, cu := range carry {
					curLen += runeLen(cu.text)
				}
			} else {
				// No sentence boundary found — fall back to overlap-based continuation.
				current, curLen = computeOverlap(current, chunkOverlap, chunkSize, uLen)
			}

			// Shrink overlap/carry further if needed to fit headers + next unit
			if headers != "" && headersLen+uLen <= chunkSize {
				for len(current) > 0 && curLen+uLen+headersLen > chunkSize {
					curLen -= runeLen(current[0].text)
					current = current[1:]
				}

				// Prepend headers if the column-name context is not already present
				// in the overlap or the next unit being added.
				overlapText := unitsText(current)
				if !headerAlreadyPresent(headers, overlapText, u.text) {
					startPos := u.start
					if len(current) > 0 {
						startPos = current[0].start
					}
					hUnit := splitUnit{text: headers, start: startPos, end: startPos}
					current = append([]splitUnit{hUnit}, current...)
					curLen += headersLen
				}
			}
		}

		// Check if adding this unit would exceed absolute max
		if curLen+uLen > absoluteMaxSize {
			if len(current) > 0 {
				chunks = append(chunks, buildChunk(current, len(chunks)))
				current = nil
				curLen = 0
			}
		}

		current = append(current, u)
		curLen += uLen
	}

	// Flush remaining
	if len(current) > 0 {
		chunks = append(chunks, buildChunk(current, len(chunks)))
	}

	return chunks
}

// unitsText concatenates the text of all units.
func unitsText(units []splitUnit) string {
	var sb strings.Builder
	for _, u := range units {
		sb.WriteString(u.text)
	}
	return sb.String()
}

// headerAlreadyPresent returns true if the column-name row from the header
// is already present in the overlap or the next unit, preventing duplication.
func headerAlreadyPresent(headers, overlapText, unitText string) bool {
	// Fast path: full header already in overlap or unit
	if strings.Contains(overlapText, headers) || strings.Contains(unitText, headers) {
		return true
	}

	// Extract the column-name row (first meaningful non-separator line).
	// For a rewritten header like "| col1 | col2 |\n| --- | --- |\n",
	// the first line is the column names.
	colRow := headerColumnRow(headers)
	if colRow == "" {
		return false
	}

	return strings.Contains(overlapText, colRow) || strings.Contains(unitText, colRow)
}

// headerColumnRow extracts the column-name line from a header string.
// Returns empty string if the header has no meaningful column names.
func headerColumnRow(header string) string {
	for _, line := range strings.Split(header, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.Contains(line, "---") {
			continue
		}
		// Skip lines that are only pipes/whitespace (empty header rows)
		onlyPipes := true
		for _, r := range line {
			if r != '|' && r != ' ' && r != '\t' {
				onlyPipes = false
				break
			}
		}
		if !onlyPipes {
			return line
		}
	}
	return ""
}

func buildChunk(units []splitUnit, seq int) Chunk {
	var sb strings.Builder
	for _, u := range units {
		sb.WriteString(u.text)
	}
	content := NormalizeStickyHeadingsInText(sb.String())
	chunk := Chunk{
		Content: content,
		Seq:     seq,
		Start:   units[0].start,
		End:     units[len(units)-1].end,
	}
	populateStructuralMetadata(&chunk)
	return chunk
}

// populateStructuralMetadata extracts tables, formulas, code blocks, and
// images from the chunk content and populates both the typed fields and the
// Metadata map (for backward-compatible DB persistence).
func populateStructuralMetadata(c *Chunk) {
	tables := ExtractTableElements(c.Content)
	formulas := ExtractFormulaRefs(c.Content)
	codeBlocks := ExtractCodeBlocks(c.Content)
	images := ExtractImageRefs(c.Content)

	c.Tables = tables
	c.Formulas = formulas
	c.CodeBlocks = codeBlocks
	c.Images = images

	meta := make(map[string]any, 6)
	meta["table_count"] = len(tables)
	if len(tables) > 0 {
		meta["table_summary"] = tableHeaderSummary(tables[0])
		if len(tables[0].Columns) > 0 {
			meta["table_columns"] = tables[0].Columns
		}
	}
	meta["formula_count"] = len(formulas)
	meta["code_block_count"] = len(codeBlocks)
	meta["image_count"] = len(images)
	c.Metadata = meta
}

// tableHeaderSummary returns the column names of a TableElement joined as a
// pipe-delimited string. Used as a compact summary for the Metadata map.
func tableHeaderSummary(t TableElement) string {
	if len(t.Columns) == 0 {
		return ""
	}
	return strings.Join(t.Columns, " | ")
}

// computeOverlap returns the units to keep for overlap and their total rune length.
func computeOverlap(current []splitUnit, chunkOverlap, chunkSize, nextLen int) ([]splitUnit, int) {
	if chunkOverlap <= 0 {
		return nil, 0
	}

	// Walk backward from end, accumulating overlap
	overlapLen := 0
	startIdx := len(current)
	for i := len(current) - 1; i >= 0; i-- {
		uLen := runeLen(current[i].text)
		if overlapLen+uLen > chunkOverlap {
			break
		}
		// Check that overlap + next unit fits in chunk
		if overlapLen+uLen+nextLen > chunkSize {
			break
		}
		overlapLen += uLen
		startIdx = i
	}

	// Skip leading separator-only and header-marker units in the overlap
	for startIdx < len(current) {
		u := current[startIdx]
		isHeaderMarker := u.start == u.end
		trimmed := strings.TrimSpace(u.text)
		if isHeaderMarker || trimmed == "" || isSeparatorOnly(u.text) {
			overlapLen -= runeLen(u.text)
			startIdx++
		} else {
			break
		}
	}

	if startIdx >= len(current) {
		return nil, 0
	}

	overlap := make([]splitUnit, len(current)-startIdx)
	copy(overlap, current[startIdx:])
	return overlap, overlapLen
}

func isSeparatorOnly(s string) bool {
	for _, r := range s {
		if r != '\n' && r != '\r' && r != ' ' && r != '\t' && r != '。' {
			return false
		}
	}
	return true
}

// ParentChildResult holds the two-level chunking output.
// Parent chunks provide context (large window), child chunks are used for
// embedding/retrieval (small window). Each child carries its ParentIndex so
// the caller can wire up ParentChunkID after DB insertion.
type ParentChildResult struct {
	Parents  []Chunk
	Children []ChildChunk
}

// ChildChunk extends Chunk with a reference to its parent.
type ChildChunk struct {
	Chunk
	ParentIndex int // index into ParentChildResult.Parents
}

// SplitTextParentChild performs two-level chunking:
//  1. Split text into large parent chunks (parentCfg).
//  2. Split each parent into smaller child chunks (childCfg) for embedding.
//
// The child Seq is globally unique across the entire document.
func SplitTextParentChild(text string, parentCfg, childCfg SplitterConfig) ParentChildResult {
	parents := SplitText(text, parentCfg)
	if len(parents) == 0 {
		return ParentChildResult{}
	}

	var newParents []Chunk
	var children []ChildChunk
	childSeq := 0
	for _, parent := range parents {
		subs := SplitText(parent.Content, childCfg)

		parentIndex := -1
		if len(subs) > 1 || (len(subs) == 1 && subs[0].Content != parent.Content) {
			parentIndex = len(newParents)
			newParents = append(newParents, parent)
		}

		for _, sub := range subs {
			// Adjust offsets: sub positions are relative to parent content,
			// shift to document-level offsets.
			// Use additive shift (not Content-length based) so that chunks with
			// prepended context headers keep correct positional tracking.
			sub.Seq = childSeq
			sub.Start += parent.Start
			sub.End += parent.Start
			children = append(children, ChildChunk{
				Chunk:       sub,
				ParentIndex: parentIndex,
			})
			childSeq++
		}
	}
	return ParentChildResult{Parents: newParents, Children: children}
}

// ExtractImageRefs extracts markdown image references from text.
// The URL group supports one level of balanced parentheses so that URLs
// like https://example.com/item_(abc)/123 are captured in full.
var imageRefPattern = regexp.MustCompile(`!\[([^\]]*)\]\(([^()\s]*(?:\([^)]*\)[^()\s]*)*)\)`)

func ExtractImageRefs(text string) []ImageRef {
	text = docparser.UnwrapLinkedImages(text)
	matches := imageRefPattern.FindAllStringSubmatchIndex(text, -1)
	var refs []ImageRef
	for _, m := range matches {
		refs = append(refs, ImageRef{
			OriginalRef: text[m[4]:m[5]], // group 2: URL
			AltText:     text[m[2]:m[3]], // group 1: alt text
			Start:       m[0],
			End:         m[1],
		})
	}
	return refs
}
