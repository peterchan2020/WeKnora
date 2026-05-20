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
// tokens. Use 200–400 for FAQ-style atomic content, 1024+ for narrative
// documents where answer_in_one_chunk matters.
//
// DefaultChunkOverlap = 80 chars (~15% of DefaultChunkSize): provides
// cross-boundary recall without excessive duplication.
const (
	DefaultChunkSize    = 512
	DefaultChunkOverlap = 80
)

// DefaultConfig returns sensible defaults.
func DefaultConfig() SplitterConfig {
	return SplitterConfig{
		ChunkSize:    DefaultChunkSize,
		ChunkOverlap: DefaultChunkOverlap,
		Separators:   []string{"\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", "; ", "；"},
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
	regexp.MustCompile("(?s)```(?:\\w+)?[ \\t]*(?:\\r?\\n)?.*?```"),                                      // Fenced code blocks
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

// protectedSpansRuneFor computes rune-offset protected spans for text in a
// single call. This is the canonical way to get protected spans for merge
// decisions — callers should use this instead of calling protectedSpansRune
// and protectedSpans separately.
func protectedSpansRuneFor(text string) []span {
	return protectedSpansRune(text, protectedSpans(text))
}

// mergeCrossesProtectedSpan checks whether merging chunks that cover the
// rune range [mergeStart, mergeEnd) would split any protected span (LaTeX
// block, fenced code, etc.) in half. A span is split if it starts inside
// the merge range but ends outside, or starts before the range but ends
// inside. Spans fully contained within or fully containing the range are
// fine. The spans must be sorted by start (protectedSpans guarantees this).
func mergeCrossesProtectedSpan(mergeStart, mergeEnd int, spans []span) bool {
	for _, s := range spans {
		if s.start >= mergeEnd {
			break
		}
		// Span starts inside merge range but ends after it → split.
		if s.start >= mergeStart && s.end > mergeEnd {
			return true
		}
		// Span starts before merge range but ends inside it → split.
		if s.start < mergeStart && s.end > mergeStart {
			return true
		}
	}
	return false
}

// adjustOutOfSpan returns pos adjusted so it does not fall inside any
// protected span. If pos is inside a span, it returns span.end (push
// forward past the span). If pos is not inside any span, it returns pos
// unchanged. When spans is nil or empty, returns pos unchanged.
// Spans must be sorted by start.
func adjustOutOfSpan(pos int, spans []span) int {
	for _, s := range spans {
		if s.start > pos {
			break
		}
		if pos >= s.start && pos < s.end {
			return s.end
		}
	}
	return pos
}

// splitUnit is a piece of text with its original position.
type splitUnit struct {
	text       string
	start, end int
	isSentence bool // true = produced by sentence-level separator, must not be split across chunks
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
var sentenceLevelSeparators = map[string]bool{
	". ": true, "! ": true, "? ": true,
	"。": true, "！": true, "？": true,
}

func isSentenceLevelSeparator(sep string) bool {
	return sentenceLevelSeparators[sep]
}

func dropSentenceLevelSeparators(seps []string) []string {
	out := make([]string, 0, len(seps))
	for _, s := range seps {
		if !isSentenceLevelSeparator(s) {
			out = append(out, s)
		}
	}
	return out
}

func splitBySeparators(text string, separators []string, chunkSize int) []splitUnit {
	mkUnit := func(s string) splitUnit {
		return splitUnit{text: s, isSentence: true}
	}
	if text == "" || len(separators) == 0 {
		return []splitUnit{mkUnit(text)}
	}
	if chunkSize > 0 && runeLen(text) <= chunkSize {
		return []splitUnit{mkUnit(text)}
	}

	for i, sep := range separators {
		if sep == "" {
			continue
		}

		if isSentenceLevelSeparator(sep) {
			sentUnits := splitBySentencesPunkt(text)
			if len(sentUnits) > 1 {
				remaining := dropSentenceLevelSeparators(separators[i+1:])
				var out []splitUnit
				for _, s := range sentUnits {
					if chunkSize > 0 && runeLen(s.text) > chunkSize && len(remaining) > 0 {
						out = append(out, splitBySeparators(s.text, remaining, chunkSize)...)
					} else {
						out = append(out, s)
					}
				}
				return out
			}
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

		var out []splitUnit
		remaining := separators[i+1:]
		for _, p := range pieces {
			if chunkSize > 0 && runeLen(p) > chunkSize && len(remaining) > 0 {
				out = append(out, splitBySeparators(p, remaining, chunkSize)...)
			} else {
				out = append(out, mkUnit(p))
			}
		}
		return out
	}
	return []splitUnit{mkUnit(text)}
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

	// When the region is much larger than maxSize (e.g. a 3000-char
	// HTML table with maxSize=512), prefer splitting at </tr> so
	// oversized tables are split by row rather than kept as one
	// monolithic chunk. Only prefer </table> when the region is
	// reasonably close to maxSize.
	regionLen := chunkEnd - offset
	preferTableRowSplit := regionLen > maxSize*3/2

	if !preferTableRowSplit {
		// Normal case: prefer </table> for structural completeness.
		for i := chunkEnd; i >= tagSearchStart; i-- {
			if hasASCIITagSuffix(runes, offset, i, "</table>") {
				return i
			}
		}
	}

	// Split at row/list/paragraph boundaries.
	for i := chunkEnd; i >= tagSearchStart; i-- {
		if hasASCIITagSuffix(runes, offset, i, "</tr>") ||
			hasASCIITagSuffix(runes, offset, i, "</li>") ||
			hasASCIITagSuffix(runes, offset, i, "</p>") {
			return i
		}
	}

	// For oversized tables where row split was preferred, try </table>
	// as a fallback if no </tr> was found.
	if preferTableRowSplit {
		for i := chunkEnd; i >= tagSearchStart; i-- {
			if hasASCIITagSuffix(runes, offset, i, "</table>") {
				return i
			}
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

	// Step 1: Find protected spans (byte offsets) and precompute rune-offset
	// spans for downstream merge decisions. This avoids redundant regex scans
	// in coalesceOrphanHeadings and other coalesce functions.
	protected := protectedSpans(text)
	protSpans := protectedSpansRune(text, protected)

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
	chunks = coalesceOrphanHeadings(chunks, chunkSize, protSpans)

	// Step 3c: Re-split any chunks that still exceed 1.5x chunkSize at
	// paragraph/sentence boundaries. This catches oversized chunks produced
	// by coalescing or protected-region preservation.
	chunks = resplitOversizedChunks(chunks, chunkSize)

	// Step 3d: Re-run orphan heading coalescence after resplitting. The
	// resplit pass can produce new heading-only fragments when it cuts an
	// oversized chunk at a paragraph break that follows a heading line.
	chunks = coalesceOrphanHeadings(chunks, chunkSize, protSpans)

	return chunks
}

// resplitOversizedChunks re-splits any chunk whose Content exceeds 1.5×
// chunkSize at paragraph or sentence boundaries. Protected spans (math
// formulas, code blocks) are computed per chunk and respected so they are
// not split across sub-chunks.
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

		// Compute protected spans for this chunk's content (offsets are
		// local to the chunk, not the original document).
		chunkSpans := protectedSpansRune(c.Content, protectedSpans(c.Content))

		parts := splitOversizedRunes(runes, chunkSize, chunkSpans)
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
// When spans is non-nil, split points that fall inside a protected span are
// adjusted past the span's end (or before its start) so formulas and code
// blocks are not split across chunks.
func splitOversizedRunes(runes []rune, chunkSize int, spans []span) [][]rune {
	if len(runes) <= chunkSize {
		return [][]rune{runes}
	}

	var parts [][]rune
	start := 0
	for start < len(runes) {
		end := findBestSplitPoint(runes, start, chunkSize, spans)
		parts = append(parts, runes[start:end])
		start = end
	}
	return parts
}

// findBestSplitPoint finds the best position to split runes[start:] into a
// chunk of at most chunkSize runes. Prefers \n\n, then \n, then CJK sentence
// boundaries (no space required), then English sentence boundaries (with
// abbreviation awareness), then clause boundaries, then a hard cut as last resort.
//
// When spans is non-nil, the chosen split point is adjusted out of any protected
// span it falls inside. If the split point is inside a span, it is pushed to
// span.end (keeping the span in the left chunk). If span.end exceeds the search
// range, it retreats to span.start. If neither works, it returns the end of the
// range to keep the span intact as one oversized chunk.
func findBestSplitPoint(runes []rune, start, chunkSize int, spans []span) int {
	maxEnd := start + chunkSize
	if maxEnd >= len(runes) {
		return len(runes)
	}

	// Paragraph break (\n\n).
	for i := maxEnd; i > start+1; i-- {
		if runes[i-1] == '\n' && runes[i] == '\n' {
			return adjustSplitOutOfSpans(i+1, start, len(runes), spans)
		}
	}

	// Single newline.
	for i := maxEnd; i > start; i-- {
		if runes[i] == '\n' {
			return adjustSplitOutOfSpans(i+1, start, len(runes), spans)
		}
	}

	// CJK sentence boundary: 。！？ don't require trailing whitespace.
	// These are common in academic PDFs and were the primary source of
	// mid-sentence cuts (84% boundary violations in the academic KB eval).
	for i := maxEnd; i > start+1; i-- {
		r := runes[i-1]
		if r == '。' || r == '！' || r == '？' {
			return adjustSplitOutOfSpans(i, start, len(runes), spans)
		}
	}

	// English sentence boundary (primary: period/exclamation/question), excluding abbreviations.
	for i := maxEnd; i > start+1; i-- {
		if primarySentenceEnd[runes[i-1]] {
			if runes[i-1] == '.' && isAbbreviationDot(runes, i-1) {
				continue
			}
			after := i
			if after < len(runes) && unicode.IsSpace(runes[after]) {
				return adjustSplitOutOfSpans(after, start, len(runes), spans)
			}
		}
	}

	// Clause boundary: ,;: — weaker than sentence but better than hard cut.
	for i := maxEnd; i > start+1; i-- {
		if clauseEndRunes[runes[i-1]] {
			after := i
			if after < len(runes) && (unicode.IsSpace(runes[after]) || runes[after] == '\n') {
				return adjustSplitOutOfSpans(after, start, len(runes), spans)
			}
		}
	}

	// If remaining text after maxEnd is <30% of chunkSize, extend to end
	// rather than creating a tiny trailing fragment with a mid-sentence cut.
	remaining := len(runes) - maxEnd
	if remaining > 0 && remaining <= chunkSize*3/10 {
		return len(runes)
	}

	// Hard cut (last resort).
	return adjustSplitOutOfSpans(maxEnd, start, len(runes), spans)
}

// adjustSplitOutOfSpans adjusts a split point so it does not fall inside a
// protected span. If pos is inside a span, it pushes to span.end (keeping
// the span in the left chunk). If span.end > totalLen (unrealistic but
// defensive), it retreats to span.start. If neither works, returns
// totalLen to keep the span intact.
func adjustSplitOutOfSpans(pos, start, totalLen int, spans []span) int {
	if len(spans) == 0 {
		return pos
	}
	for _, s := range spans {
		if s.start > pos {
			break
		}
		if pos >= s.start && pos < s.end {
			if s.end <= totalLen {
				return s.end
			}
			if s.start > start {
				return s.start
			}
			return totalLen
		}
	}
	return pos
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
				partRuneLen := runeLen(part.text)
				units = append(units, splitUnit{
					text:       part.text,
					start:      runeOffset,
					end:        runeOffset + partRuneLen,
					isSentence: part.isSentence,
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
			partRuneLen := runeLen(part.text)
			units = append(units, splitUnit{
				text:       part.text,
				start:      runeOffset,
				end:        runeOffset + partRuneLen,
				isSentence: part.isSentence,
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
//
// The function uses an accumulator for consecutive heading-only / near-empty /
// very-small chunks so that chains like "# 7.4." → "# 7.6.4." → "7.6.4.2."
// are all absorbed into the next real-content chunk rather than being merged
// pair-wise and emitting tiny intermediate chunks.
func coalesceOrphanHeadings(chunks []Chunk, chunkSize int, protSpans []span) []Chunk {
	if len(chunks) <= 1 || chunkSize <= 0 {
		return chunks
	}

	verySmallThreshold := chunkSize / 6

	out := make([]Chunk, 0, len(chunks))
	// pendingHeadings accumulates consecutive heading-only, near-empty, or
	// very-small chunks until a normal-size chunk is encountered.
	var pendingHeadings []Chunk

	isTiny := func(c Chunk) bool {
		if _, ok := pureHeadingChunkHeader(c); ok {
			return true
		}
		trimmed := strings.TrimSpace(c.Content)
		trimmedLen := utf8.RuneCountInString(trimmed)
		if trimmedLen < 10 {
			return true
		}
		if utf8.RuneCountInString(c.Content) < verySmallThreshold {
			return true
		}
		return false
	}

	// pendingRunesCovered returns the rune range [start, end) covered by
	// pendingHeadings. Used to check protected-span crossings.
	pendingRunesCovered := func() (start, end int) {
		if len(pendingHeadings) == 0 {
			return 0, 0
		}
		start = pendingHeadings[0].Start
		end = pendingHeadings[len(pendingHeadings)-1].End
		return
	}

	flushPending := func(target *Chunk) {
		if len(pendingHeadings) == 0 {
			return
		}
		// Check if merging pending headings into target would cross a
		// protected span boundary (e.g. split a $$..$$ block).
		pStart, _ := pendingRunesCovered()
		mergeStart := pStart
		mergeEnd := target.End
		if mergeStart > target.Start {
			mergeStart = target.Start
		}
		crossesProtected := mergeCrossesProtectedSpan(mergeStart, mergeEnd, protSpans)

		// Separate pure headings from non-heading tiny chunks.
		// Only pure headings go into ContextHeader; all tiny chunks go into Content.
		var headingParts []string // pure headings → ContextHeader breadcrumb
		var contentParts []string // all tiny chunks → Content merge
		for _, ph := range pendingHeadings {
			header, ok := pureHeadingChunkHeader(ph)
			if ok && header != "" {
				headingParts = append(headingParts, header)
				contentParts = append(contentParts, header)
			} else {
				contentParts = append(contentParts, strings.TrimSpace(ph.Content))
			}
		}
		mergedHeader := strings.Join(headingParts, "\n")
		mergedContent := strings.Join(contentParts, "\n\n") + "\n\n" + strings.TrimSpace(target.Content)
		combinedLen := utf8.RuneCountInString(mergedContent)

		if combinedLen <= chunkSize && !crossesProtected {
			// Budget allows: prepend all tiny chunk text into Content so it's
			// retrievable by search.
			target.Content = mergedContent
			populateStructuralMetadata(target)
			pendingHeadings = nil
			return
		}
		// Budget exceeded or protected span would be crossed: promote heading
		// context but emit non-heading tiny chunks as standalone output rather
		// than silently dropping them.
		if mergedHeader != "" {
			target.ContextHeader = mergeBreadcrumbs(mergedHeader, target.ContextHeader)
		}
		for _, ph := range pendingHeadings {
			if _, ok := pureHeadingChunkHeader(ph); !ok {
				out = append(out, ph)
			}
		}
		pendingHeadings = nil
	}

	for i := 0; i < len(chunks); i++ {
		c := chunks[i]
		if !isTiny(c) {
			// Normal-size chunk: flush all accumulated headings into it.
			flushPending(&c)
			out = append(out, c)
			continue
		}
		// Tiny chunk: accumulate.
		pendingHeadings = append(pendingHeadings, c)
	}

	// Flush remaining pending headings backward into last output chunk.
	if len(pendingHeadings) > 0 {
		if len(out) > 0 {
			last := &out[len(out)-1]
			// Same separation logic as flushPending: only pure headings
			// go into ContextHeader, all tiny chunks go into Content.
			var headingParts []string
			var contentParts []string
			for _, ph := range pendingHeadings {
				header, ok := pureHeadingChunkHeader(ph)
				if ok && header != "" {
					headingParts = append(headingParts, header)
					contentParts = append(contentParts, header)
				} else {
					contentParts = append(contentParts, strings.TrimSpace(ph.Content))
				}
			}
			mergedHeader := strings.Join(headingParts, "\n")
			mergedContent := strings.TrimSpace(last.Content) + "\n\n" + strings.Join(contentParts, "\n\n")
			combinedLen := utf8.RuneCountInString(mergedContent)
			if combinedLen <= chunkSize {
				last.Content = mergedContent
				populateStructuralMetadata(last)
			}
			// Always promote pure heading context even if Content budget
			// exceeded — avoids emitting orphan heading-only chunks.
			if mergedHeader != "" {
				last.ContextHeader = mergeBreadcrumbs(last.ContextHeader, mergedHeader)
			}
		} else {
			// No previous chunk to merge into — emit as-is.
			out = append(out, pendingHeadings...)
		}
	}

	// Re-sequence.
	for i := range out {
		out[i].Seq = i
	}
	return out
}

// sentenceEndRunes are the punctuation runes that mark a sentence boundary
// when followed by whitespace or end-of-text. Kept for backward compatibility
// with callers that don't need the primary/secondary distinction.
var sentenceEndRunes = map[rune]bool{
	'.': true,
	'!': true,
	'?': true,
	'。': true,
	'！': true,
	'？': true,
	'；': true,
}

// primarySentenceEnd marks strong sentence-ending punctuation that almost
// always terminates a sentence (period, exclamation, question mark in both
// scripts). Used by findBestSplitPoint and snapRuneToSentenceEnd to prefer
// these over weaker clause boundaries.
var primarySentenceEnd = map[rune]bool{
	'.': true,
	'!': true,
	'?': true,
	'。': true,
	'！': true,
	'？': true,
}

// clauseEndRunes marks clause-level punctuation where a split is acceptable
// (though not ideal) when no primary sentence boundary is found. These are
// used as a fallback tier in findBestSplitPoint and snapRuneToSentenceEnd.
var clauseEndRunes = map[rune]bool{
	',': true,
	';': true,
	':': true,
	'，': true,
	'、': true,
	'；': true,
	'：': true,
	'—': true, // em dash (clause break)
	'…': true, // ellipsis (clause break)
}

// isAbbreviationDot returns true when the '.' at position i in runes is part
// of a known abbreviation (e.g. "Fig.", "et al.", "vs.") and should NOT be
// treated as a sentence-ending period. It looks backward from i to collect the
// preceding word and checks against a common abbreviation set.
func isAbbreviationDot(runes []rune, i int) bool {
	if i <= 0 || runes[i] != '.' {
		return false
	}
	// Collect the word preceding the dot (up to 10 runes back).
	j := i - 1
	for j >= 0 && j > i-11 {
		r := runes[j]
		if r == ' ' || r == '\n' || r == '\t' || r == ',' || r == ';' || r == ':' {
			break
		}
		j--
	}
	word := strings.ToLower(string(runes[j+1 : i]))
	return abbreviationSet[word]
}

// abbreviationSet contains common English abbreviations whose trailing period
// should not be treated as a sentence boundary. Curated from academic paper
// patterns observed in the WeKnora academic knowledge base.
var abbreviationSet = map[string]bool{
	"fig": true, "figs": true, "eq": true, "eqs": true,
	"et al": true, "vs": true, "etc": true,
	"i.e": true, "e.g": true, "cf": true, "al": true,
	"dr": true, "prof": true, "mr": true, "mrs": true, "ms": true,
	"no": true, "nos": true, "vol": true, "pp": true,
	"ref": true, "refs": true, "sec": true, "sect": true,
	"approx": true, "max": true, "min": true, "avg": true,
	"dept": true, "univ": true, "inc": true, "ltd": true,
	"ed": true, "eds": true, "rev": true, "est": true,
	"jan": true, "feb": true, "mar": true, "apr": true,
	"jun": true, "jul": true, "aug": true, "sep": true,
	"sept": true, "oct": true, "nov": true, "dec": true,
	"st": true, "nd": true, "rd": true, "th": true,
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
//
// When spans is non-nil, the snapped position is adjusted out of any protected
// span it falls inside. If adjustment pushes past end, the function tries
// retreating before the span (re-snapping within [start, spanStart)); if that
// also fails, it returns end to keep the protected span intact as one chunk.
func snapRuneToSentenceEnd(runes []rune, start, end, minSize int, spans []span) int {
	if end <= start || end-start < minSize {
		return end
	}

	snapped := snapRuneToSentenceEndInner(runes, start, end, minSize)

	if len(spans) == 0 {
		return snapped
	}

	adjusted := adjustOutOfSpan(snapped, spans)
	if adjusted <= end {
		return adjusted
	}

	// Adjustment pushed past end — try retreating before the span instead.
	for _, s := range spans {
		if snapped >= s.start && snapped < s.end && s.start >= start+minSize {
			retreat := snapRuneToSentenceEndInner(runes, start, s.start, minSize)
			if retreat >= start+minSize {
				return retreat
			}
			break
		}
	}
	// Cannot retreat without violating minSize. Keep the protected span intact.
	return end
}

// snapRuneToSentenceEndInner is the original sentence-snap logic without
// protected-span awareness. Extracted so the outer function can re-call it
// with a narrower range when retreating before a protected span.
func snapRuneToSentenceEndInner(runes []rune, start, end, minSize int) int {
	if end <= start || end-start < minSize {
		return end
	}

	// Pass 0: CJK sentence boundaries (。！？) — no trailing space required.
	// These must be checked first because the English pass below requires
	// whitespace after the punctuation, which CJK never has.
	for i := end - 1; i > start; i-- {
		r := runes[i]
		if r == '。' || r == '！' || r == '？' {
			boundary := i + 1
			if boundary-start >= minSize {
				return boundary
			}
			break
		}
	}

	// Pass 1: Search backward for a primary sentence-ending rune (.!?).
	for i := end - 1; i > start; i-- {
		if !primarySentenceEnd[runes[i]] {
			continue
		}
		if runes[i] == '.' && isAbbreviationDot(runes, i) {
			continue
		}
		after := i + 1
		if after < end {
			r := runes[after]
			if !unicode.IsSpace(r) && r != '"' && r != '\'' && r != ')' && r != ']' && r != '}' && r != '’' && r != '”' {
				continue
			}
		}
		boundary := after
		for boundary < end && (runes[boundary] == ' ' || runes[boundary] == '\t') {
			boundary++
		}
		if boundary-start >= minSize {
			return boundary
		}
		break
	}

	// Pass 2: Fallback to clause boundaries (,;: etc.) when no primary found.
	for i := end - 1; i > start; i-- {
		if !clauseEndRunes[runes[i]] {
			continue
		}
		after := i + 1
		if after < end {
			r := runes[after]
			if !unicode.IsSpace(r) && r != '"' && r != '\'' && r != ')' && r != ']' && r != '}' && r != '’' && r != '”' {
				continue
			}
		}
		boundary := after
		for boundary < end && (runes[boundary] == ' ' || runes[boundary] == '\t') {
			boundary++
		}
		if boundary-start >= minSize {
			return boundary
		}
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

	accumulated := 0
	type splitPoint struct {
		unitIdx int
		runeOff int
	}
	var best splitPoint
	bestSet := false

	// Pass 0: CJK sentence boundaries (。！？) — no trailing space required.
	for i := len(units) - 1; i >= 0 && accumulated < maxLookback; i-- {
		runes := []rune(units[i].text)
		accumulated += len(runes)

		for j := len(runes) - 1; j >= 0; j-- {
			r := runes[j]
			if r == '。' || r == '！' || r == '？' {
				splitAt := j + 1
				if splitAt > 0 && splitAt <= len(runes) {
					best = splitPoint{unitIdx: i, runeOff: splitAt}
					bestSet = true
				}
				break
			}
		}
		if bestSet {
			break
		}
	}

	// Pass 1: primary sentence endings (.!?), excluding abbreviations.
	if !bestSet {
		accumulated = 0
		for i := len(units) - 1; i >= 0 && accumulated < maxLookback; i-- {
			runes := []rune(units[i].text)
			accumulated += len(runes)

			for j := len(runes) - 1; j >= 1; j-- {
				if !primarySentenceEnd[runes[j]] {
					continue
				}
				if runes[j] == '.' && isAbbreviationDot(runes, j) {
					continue
				}
				after := j + 1
				if after < len(runes) {
					r := runes[after]
					if !unicode.IsSpace(r) && r != '"' && r != '\'' && r != ')' && r != ']' && r != '}' && r != '’' && r != '”' {
						continue
					}
				}
				splitAt := after
				for splitAt < len(runes) && (runes[splitAt] == ' ' || runes[splitAt] == '\t') {
					splitAt++
				}
				if splitAt > 0 && splitAt <= len(runes) {
					best = splitPoint{unitIdx: i, runeOff: splitAt}
					bestSet = true
				}
				break
			}
		}
	}

	// Pass 2: clause-level fallback (,;: etc.) when no primary found.
	if !bestSet {
		accumulated = 0
		for i := len(units) - 1; i >= 0 && accumulated < maxLookback; i-- {
			runes := []rune(units[i].text)
			accumulated += len(runes)

			for j := len(runes) - 1; j >= 1; j-- {
				if !clauseEndRunes[runes[j]] {
					continue
				}
				after := j + 1
				if after < len(runes) {
					r := runes[after]
					if !unicode.IsSpace(r) && r != '"' && r != '\'' && r != ')' && r != ']' && r != '}' && r != '’' && r != '”' {
						continue
					}
				}
				splitAt := after
				for splitAt < len(runes) && (runes[splitAt] == ' ' || runes[splitAt] == '\t') {
					splitAt++
				}
				if splitAt > 0 && splitAt <= len(runes) {
					best = splitPoint{unitIdx: i, runeOff: splitAt}
					bestSet = true
				}
				break
			}
		}
	}

	if !bestSet {
		return units, nil
	}

	// Split at the recorded position.
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
			// If the next unit is a complete sentence and the overflow is
			// modest (≤10% of chunkSize), allow slight overflow for
			// sentence integrity instead of cutting mid-sentence.
			if u.isSentence && curLen+uLen+headersLen <= chunkSize+chunkSize/5 {
				// Fall through to append the unit below
			} else {
				// Try to snap the flush boundary to a sentence boundary so we
				// don't cut mid-sentence. Any carry-over units replace the
				// overlap for the next chunk.
				flushed, carry := snapFlushToSentenceBoundary(current, chunkSize)
				if len(flushed) > 0 {
					chunks = append(chunks, buildChunk(flushed, len(chunks)))
				}
				// Always prepend overlap from the flushed portion so the
				// next chunk carries context across the boundary.
				overlapUnits, overlapLen := computeOverlap(flushed, chunkOverlap, chunkSize, uLen)
				if len(carry) > 0 {
					if overlapLen > 0 {
						current = append(overlapUnits, carry...)
						curLen = overlapLen
						for _, cu := range carry {
							curLen += runeLen(cu.text)
						}
					} else {
						current = carry
						curLen = 0
						for _, cu := range carry {
							curLen += runeLen(cu.text)
						}
					}
				} else {
					// No carry from sentence snap — use overlap-based continuation.
					current = overlapUnits
					curLen = overlapLen
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

	meta := make(map[string]any, 7)
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
	if c.ContextHeader != "" {
		meta["context_header"] = c.ContextHeader
	}
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
// When choosing which overlap units to keep, prefer isSentence=true units at the
// boundary so that overlap preserves complete sentence boundaries.
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

	// Prefer starting overlap at a sentence boundary: if the unit just before
	// startIdx is a sentence unit and the one at startIdx is not, try to
	// include the preceding sentence unit (shifting startIdx back) as long as
	// we stay within the overlap budget.
	if startIdx > 0 && startIdx < len(current) && !current[startIdx].isSentence && current[startIdx-1].isSentence {
		prevLen := runeLen(current[startIdx-1].text)
		if overlapLen+prevLen <= chunkOverlap && overlapLen+prevLen+nextLen <= chunkSize {
			overlapLen += prevLen
			startIdx--
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
