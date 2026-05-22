// Package chunker - heuristic_splitter.go implements Tier 2: boundary-driven
// chunking for documents that lack proper Markdown headings but contain
// recognizable structural cues (page breaks, numbered sections, multilingual
// chapter markers, visual separators, all-caps section titles, page footers).
//
// The algorithm finds all candidate boundary positions, then performs greedy
// bin-packing — accumulating blocks between boundaries into chunks until the
// next block would exceed cfg.ChunkSize. Blocks larger than ChunkSize are
// recursively delegated to the legacy splitter for inner segmentation.
package chunker

import (
	"context"
	"sort"
	"strings"
	"unicode/utf8"
)

func init() {
	splitByHeuristics = splitByHeuristicsImpl
}

// boundary marks a candidate split point in the document.
type boundary struct {
	runeStart int // rune offset where the next chunk should start
	priority  int
}

// splitByHeuristicsImpl is the Tier-2 implementation. Falls through to the
// legacy splitter when no heuristic boundaries are found.
//
// profile is currently unused (this tier scans for boundaries directly) but
// is accepted to keep the splitByHeadings / splitByHeuristics signatures
// uniform — see strategy.runTier.
func splitByHeuristicsImpl(text string, cfg SplitterConfig, profile *DocProfile) []Chunk {
	return splitByHeuristicsStructured(text, cfg, profile, false)
}

func splitByHeuristicsSemantic(
	ctx context.Context,
	text string,
	cfg SplitterConfig,
	profile *DocProfile,
	embedder SemanticEmbedder,
) []Chunk {
	coarse := splitByHeuristicsStructured(text, cfg, profile, true)
	refined, err := RefineSemantic(ctx, coarse, cfg, embedder)
	if err != nil {
		return splitByHeuristicsImpl(text, cfg, profile)
	}
	if v := ValidateChunks(refined, len([]rune(text)), cfg.ChunkSize); !v.OK {
		return splitByHeuristicsImpl(text, cfg, profile)
	}
	return refined
}

func splitByHeuristicsStructured(text string, cfg SplitterConfig, _ *DocProfile, keepOversizeBlocks bool) []Chunk {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	totalRunes := len(runes)

	bounds := findHeuristicBoundaries(text, cfg.Languages)
	protSpans := protectedSpansRuneFor(text)
	// Drop any boundary that falls strictly inside a protected region (table,
	// fenced code block, LaTeX block, etc.) — splitting there would cut
	// through atomic content. Boundaries on a span edge are kept since they
	// align with the protected region start/end.
	if len(protSpans) > 0 {
		bounds = dropBoundsInsideSpans(bounds, protSpans)
	}
	// Also drop boundaries inside list blocks (2+ items). Lists should be
	// kept intact so the second half is not orphaned from its list context.
	listSpans := listBlockSpansRune(text)
	if len(listSpans) > 0 {
		bounds = dropBoundsInsideSpans(bounds, listSpans)
	}
	if len(bounds) == 0 {
		if shouldPreserveParagraphBlocks(cfg) {
			if chunks := splitByParagraphBlocks(text, cfg); len(chunks) > 1 {
				return chunks
			}
		}
		return SplitText(text, cfg)
	}

	// Append a sentinel at end-of-document so the bin-packer can flush.
	bounds = append(bounds, boundary{runeStart: totalRunes})
	// Always start with a boundary at offset 0 if not already there.
	if bounds[0].runeStart != 0 {
		bounds = append([]boundary{{runeStart: 0}}, bounds...)
	}

	// Greedy bin-packing.
	var out []Chunk
	seq := 0
	chunkStart := bounds[0].runeStart
	curEnd := chunkStart
	minChunkSize := cfg.ChunkSize / 3
	if minChunkSize < 100 {
		minChunkSize = 100
	}

	for i := 1; i < len(bounds); i++ {
		nextEnd := bounds[i].runeStart
		blockLen := nextEnd - curEnd

		if blockLen > cfg.ChunkSize {
			// The block between the previous and this boundary is itself too
			// large to fit in any chunk. Flush current accumulation, then
			// recursively chunk the oversize block via the legacy splitter.
			// When semantic refinement is enabled, keep that block intact so
			// RefineSemantic can split within the structurally coherent segment.
			if curEnd-chunkStart > 0 {
				out = appendChunk(out, runes, chunkStart, curEnd, &seq)
				chunkStart = curEnd
			}
			if keepOversizeBlocks {
				out = appendChunk(out, runes, curEnd, nextEnd, &seq)
			} else {
				out = appendOversizeBlock(out, runes, curEnd, nextEnd, cfg, &seq)
			}
			curEnd = nextEnd
			chunkStart = nextEnd
			continue
		}

		// Would adding this block exceed the budget?
		accumulated := nextEnd - chunkStart
		if accumulated > cfg.ChunkSize && curEnd-chunkStart >= minChunkSize {
			// Snap the flush end to the nearest preceding sentence boundary
			// so we don't cut mid-sentence. Only snap when the snapped
			// position is still >= minChunkSize from chunkStart.
			flushEnd := snapRuneToSentenceEnd(runes, chunkStart, curEnd, minChunkSize, protSpans)
			out = appendChunk(out, runes, chunkStart, flushEnd, &seq)
			// Snap overlap start to the nearest semantic boundary or line
			// break instead of slicing mid-line / mid-word.
			chunkStart = applyOverlapAligned(runes, flushEnd, cfg.ChunkOverlap, bounds)
		}
		curEnd = nextEnd
	}

	// Flush remaining content.
	if curEnd > chunkStart {
		out = appendChunk(out, runes, chunkStart, curEnd, &seq)
	}
	return coalesceOrphanHeadings(groupReferenceEntries(coalesceTinyHeuristicChunks(out, runes, cfg.ChunkSize, protSpans), runes, cfg.ChunkSize), cfg.ChunkSize, protSpans)
}

func shouldPreserveParagraphBlocks(cfg SplitterConfig) bool {
	switch strings.ToLower(strings.TrimSpace(cfg.Strategy)) {
	case StrategyHeuristic, StrategySemantic, StrategyStructureAware:
		return true
	default:
		return false
	}
}

func splitByParagraphBlocks(text string, cfg SplitterConfig) []Chunk {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	var out []Chunk
	seq := 0
	blockStart := 0
	for i := 0; i < len(runes); i++ {
		end, ok := paragraphBreakEnd(runes, i)
		if !ok {
			continue
		}
		out = appendParagraphBlock(out, runes, blockStart, end, cfg, &seq)
		blockStart = end
		i = end - 1
	}
	out = appendParagraphBlock(out, runes, blockStart, len(runes), cfg, &seq)
	return out
}

func insideAnySpan(pos int, spans []span) bool {
	for _, s := range spans {
		if s.start > pos {
			return false
		}
		if pos >= s.start && pos < s.end {
			return true
		}
	}
	return false
}

func countParagraphBreaks(text string) int {
	runes := []rune(text)
	count := 0
	for i := 0; i < len(runes); i++ {
		end, ok := paragraphBreakEnd(runes, i)
		if !ok {
			continue
		}
		count++
		i = end - 1
	}
	return count
}

func appendParagraphBlock(out []Chunk, runes []rune, start, end int, cfg SplitterConfig, seq *int) []Chunk {
	if end <= start || strings.TrimSpace(string(runes[start:end])) == "" {
		return out
	}
	if cfg.ChunkSize > 0 && end-start > cfg.ChunkSize {
		return appendOversizeBlock(out, runes, start, end, cfg, seq)
	}
	return appendChunk(out, runes, start, end, seq)
}

func paragraphBreakEnd(runes []rune, start int) (int, bool) {
	if start < 0 || start >= len(runes) || runes[start] != '\n' {
		return 0, false
	}
	j := start + 1
	for j < len(runes) && isHorizontalWhitespace(runes[j]) {
		j++
	}
	if j >= len(runes) || runes[j] != '\n' {
		return 0, false
	}
	j++
	for j < len(runes) {
		k := j
		for k < len(runes) && isHorizontalWhitespace(runes[k]) {
			k++
		}
		if k >= len(runes) || runes[k] != '\n' {
			break
		}
		j = k + 1
	}
	return j, true
}

func isHorizontalWhitespace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\r'
}

// findHeuristicBoundaries scans text and returns boundary positions in
// ascending order. Lower-priority duplicates at the same offset are dropped.
func findHeuristicBoundaries(text string, langs []string) []boundary {
	var bounds []boundary

	// Form feeds — strongest single-character boundary.
	for _, idx := range allRuneIndices(text, "\f") {
		bounds = append(bounds, boundary{runeStart: idx, priority: PrioFormFeed})
	}

	// Per-line patterns walk the text once, line by line.
	lines := strings.Split(text, "\n")
	chapterPatterns := ChapterPatternsForLangs(langs)
	pos := 0
	inFence := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
		} else if !inFence {
			runeStart := pos
			added := false
			matchLine := line
			if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
				matchLine = normalized
			}
			if isStickyMarkdownHeadingLine(line) {
				bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioNumberedHead})
				added = true
			}
			for _, pat := range chapterPatterns {
				if pat.MatchString(matchLine) {
					bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioChapterMarker})
					added = true
					break
				}
			}
			if !added && NumberedSectionPattern.MatchString(matchLine) {
				bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioNumberedHead})
				added = true
			}
			if !added && AllCapsHeadingPattern.MatchString(matchLine) {
				bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioAllCapsHeading})
				added = true
			}
			if !added && VisualSeparatorPattern.MatchString(matchLine) {
				bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioVisualSep})
				added = true
			}
			if !added && PageFooterPattern.MatchString(matchLine) {
				bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioPageFooter})
			}
		}
		pos += utf8.RuneCountInString(line)
		if i < len(lines)-1 {
			pos++ // \n
		}
	}

	// Excessive blank blocks (\n{3,}). Match at the *start* of the run so we
	// drop into the next paragraph cleanly.
	for _, idx := range ExcessiveBlanksPattern.FindAllStringIndex(text, -1) {
		runeStart := utf8.RuneCountInString(text[:idx[1]])
		bounds = append(bounds, boundary{runeStart: runeStart, priority: PrioBlankBlock})
	}

	if len(bounds) == 0 {
		return nil
	}

	// Sort by position; drop near-duplicate offsets keeping the highest priority.
	sort.Slice(bounds, func(i, j int) bool {
		if bounds[i].runeStart != bounds[j].runeStart {
			return bounds[i].runeStart < bounds[j].runeStart
		}
		return bounds[i].priority > bounds[j].priority
	})
	deduped := bounds[:0]
	prev := -1
	for _, b := range bounds {
		if b.runeStart != prev {
			deduped = append(deduped, b)
			prev = b.runeStart
		}
	}
	return deduped
}

// dropBoundsInsideSpans returns bounds with entries that fall strictly
// inside any of the (rune-offset) protected spans removed. Bounds at a
// span's start or end are kept — they align with the span edge and don't
// split protected content. spans must be sorted by start.
func dropBoundsInsideSpans(bounds []boundary, spans []span) []boundary {
	if len(spans) == 0 {
		return bounds
	}
	out := bounds[:0]
boundLoop:
	for _, b := range bounds {
		for _, s := range spans {
			if s.start >= b.runeStart {
				break // remaining spans start at or after b — can't contain b
			}
			if b.runeStart < s.end {
				continue boundLoop
			}
		}
		out = append(out, b)
	}
	return out
}

// listBlockSpansRune returns rune-offset spans for each contiguous list block
// in text. A list block is a sequence of lines where each non-blank line
// starts with a Markdown list marker (-, *, +, or numbered "1."). Blocks of
// 2+ items are returned as protected spans so that the heuristic splitter
// does not place a chunk boundary in the middle of a list — splitting there
// would orphan the second half from its list context.
//
// The returned spans are sorted by start offset and do not overlap.
func listBlockSpansRune(text string) []span {
	lines := strings.Split(text, "\n")
	if len(lines) < 2 {
		return nil
	}

	// isListLine checks whether a line starts with a Markdown list marker:
	// unordered (-, *, +) or ordered (digits followed by . or )).
	isListLine := func(line string) bool {
		trimmed := strings.TrimLeft(line, " \t")
		if len(trimmed) < 2 {
			return false
		}
		// Unordered: "- item", "* item", "+ item"
		if (trimmed[0] == '-' || trimmed[0] == '*' || trimmed[0] == '+') &&
			(trimmed[1] == ' ' || trimmed[1] == '\t') {
			return true
		}
		// Ordered: "1. item", "23) item"
		j := 0
		for j < len(trimmed) && trimmed[j] >= '0' && trimmed[j] <= '9' {
			j++
		}
		if j > 0 && j < len(trimmed) && (trimmed[j] == '.' || trimmed[j] == ')') &&
			j+1 < len(trimmed) && (trimmed[j+1] == ' ' || trimmed[j+1] == '\t') {
			return true
		}
		return false
	}

	// Compute rune offset for each line start.
	runePos := 0
	lineStarts := make([]int, len(lines))
	for i, line := range lines {
		lineStarts[i] = runePos
		runePos += utf8.RuneCountInString(line)
		if i < len(lines)-1 {
			runePos++ // \n
		}
	}

	var spans []span
	blockStart := -1
	blockItems := 0

	flushBlock := func(endLineIdx int) {
		if blockStart >= 0 && blockItems >= 2 {
			start := lineStarts[blockStart]
			end := lineStarts[endLineIdx] + utf8.RuneCountInString(lines[endLineIdx])
			if endLineIdx < len(lines)-1 {
				end++ // include trailing \n
			}
			spans = append(spans, span{start: start, end: end})
		}
		blockStart = -1
		blockItems = 0
	}

	for i, line := range lines {
		if isListLine(line) {
			if blockStart < 0 {
				blockStart = i
			}
			blockItems++
		} else if strings.TrimSpace(line) == "" {
			// Blank line: tolerate if the next non-blank line continues the list.
			if blockStart >= 0 {
				nextIsList := false
				for j := i + 1; j < len(lines); j++ {
					if strings.TrimSpace(lines[j]) == "" {
						continue
					}
					nextIsList = isListLine(lines[j])
					break
				}
				if !nextIsList {
					flushBlock(i - 1)
				}
			}
		} else {
			if blockStart >= 0 {
				flushBlock(i - 1)
			}
		}
	}
	if blockStart >= 0 {
		flushBlock(len(lines) - 1)
	}
	return spans
}

// allRuneIndices returns every rune offset where needle starts in text.
// Only used for single-rune needles like form-feed.
func allRuneIndices(text, needle string) []int {
	var out []int
	if needle == "" {
		return out
	}
	pos := 0
	for _, r := range text {
		if string(r) == needle {
			out = append(out, pos)
		}
		pos++
	}
	return out
}

// appendChunk slices runes[start:end] into a Chunk and appends it to out.
// Pure-whitespace slices are skipped (boundary clustering can occasionally
// produce them). The Content stored is the raw slice — Start/End rune
// offsets must match utf8.RuneCountInString(Content) for downstream
// reconstruction code; whitespace stripping for embedding happens in
// Chunk.EmbeddingContent.
func appendChunk(out []Chunk, runes []rune, start, end int, seq *int) []Chunk {
	if end <= start {
		return out
	}
	raw := string(runes[start:end])
	if strings.TrimSpace(raw) == "" {
		return out
	}
	c := Chunk{Content: NormalizeStickyHeadingsInText(raw), Seq: *seq, Start: start, End: end}
	populateStructuralMetadata(&c)
	*seq++
	return append(out, c)
}

// appendOversizeBlock recursively chunks a region that is itself larger than
// cfg.ChunkSize, using the legacy splitter so internal length budgets and
// protected patterns are still respected.
func appendOversizeBlock(out []Chunk, runes []rune, start, end int, cfg SplitterConfig, seq *int) []Chunk {
	if end <= start {
		return out
	}
	subText := string(runes[start:end])
	subs := SplitText(subText, cfg)
	for _, s := range subs {
		out = append(out, Chunk{
			Content:    s.Content,
			Seq:        *seq,
			Start:      start + s.Start,
			End:        start + s.End,
			Tables:     s.Tables,
			Formulas:   s.Formulas,
			CodeBlocks: s.CodeBlocks,
			Images:     s.Images,
			Metadata:   s.Metadata,
		})
		*seq++
	}
	return out
}

// coalesceTinyHeuristicChunks merges chunks whose trimmed rune length is
// below the tiny threshold (50 runes) into their nearest neighbor. This
// eliminates meaningless fragments like "(2)", "因此", or a lone heading
// that carry no retrievable semantic content on their own.
//
// For overlapping chunks (where next.Start < prev.End), the merged chunk
// re-slices from the original runes array so Content stays a literal
// document slice and the End-Start == runeCount invariant holds.
//
// The first and last chunks are never merged away — they anchor the
// document boundary. Interior tiny chunks prefer the previous neighbor
// when it can absorb them without exceeding chunkSize, otherwise the
// next neighbor.
// tinyHeuristicThreshold is the minimum trimmed rune length a chunk must have
// to survive as a standalone chunk in the heuristic splitter. Interior chunks
// below this threshold are coalesced into a neighbor.
func tinyHeuristicThreshold(chunkSize int) int {
	// Raise the floor so fragments below ~100 chars are always coalesced,
	// and scale up with chunkSize so large-chunk configs don't leave
	// relatively tiny debris. chunkSize/4 catches more fragments than
	// the previous chunkSize/3 minimum without being overly aggressive.
	threshold := 100
	if chunkSize > 0 && chunkSize/4 > threshold {
		threshold = chunkSize / 4
	}
	return threshold
}

func coalesceTinyHeuristicChunks(chunks []Chunk, runes []rune, chunkSize int, protSpans []span) []Chunk {
	if len(chunks) <= 2 || chunkSize <= 0 {
		return chunks
	}

	// Copy input to avoid mutating caller's slice.
	out := make([]Chunk, len(chunks))
	copy(out, chunks)

	// Mark tiny interior chunks for merging.
	threshold := tinyHeuristicThreshold(chunkSize)
	merged := make([]bool, len(out))
	for i := 1; i < len(out)-1; i++ {
		trimmedLen := utf8.RuneCountInString(strings.TrimSpace(out[i].Content))
		if trimmedLen < threshold {
			merged[i] = true
		}
	}

	// Merge tiny chunks into neighbors. Walk left-to-right so each
	// merge decision sees the already-merged state of the previous chunk.
	// Consecutive tiny chunks that can't merge backward are accumulated
	// and merged forward as a group into the next non-tiny chunk.
	var result []Chunk
	var tinyGroupStart int // rune start of the first tiny chunk in a forward-merge group
	inTinyGroup := false
	for i := 0; i < len(out); i++ {
		if merged[i] {
			// Try merging into the previous chunk in result.
			if len(result) > 0 && !inTinyGroup {
				prev := &result[len(result)-1]
				newEnd := out[i].End
				newLen := newEnd - prev.Start
				if newLen <= chunkSize && !mergeCrossesProtectedSpan(prev.Start, newEnd, protSpans) {
					prev.End = newEnd
					prev.Content = string(runes[prev.Start:newEnd])
					populateStructuralMetadata(prev)
					continue
				}
			}
			// Can't merge backward — start or extend a forward-merge group.
			if !inTinyGroup {
				tinyGroupStart = out[i].Start
				inTinyGroup = true
			}
			continue
		}
		// Non-tiny chunk: flush any accumulated tiny group by extending
		// this chunk's start backward to cover the group.
		if inTinyGroup {
			if !mergeCrossesProtectedSpan(tinyGroupStart, out[i].End, protSpans) {
				out[i].Start = tinyGroupStart
				out[i].Content = string(runes[out[i].Start:out[i].End])
				populateStructuralMetadata(&out[i])
			}
			inTinyGroup = false
		}
		result = append(result, out[i])
	}
	// If a tiny group remains at the end (no non-tiny chunk follows),
	// merge it into the last result chunk so content isn't lost.
	if inTinyGroup && len(result) > 0 {
		prev := &result[len(result)-1]
		if !mergeCrossesProtectedSpan(prev.Start, out[len(out)-1].End, protSpans) {
			prev.End = out[len(out)-1].End
			prev.Content = string(runes[prev.Start:prev.End])
			populateStructuralMetadata(prev)
		}
	}

	// Re-sequence.
	for i := range result {
		result[i].Seq = i
	}
	return result
}

// applyOverlapAligned returns the rune offset where the next chunk should
// start. The target is `curEnd - overlap`, but we snap to the nearest
// preceding boundary (within 2x overlap) or, failing that, the previous
// newline so chunks don't begin mid-line / mid-word. Falls back to the raw
// target only if neither option is available.
//
// curEnd itself is always a boundary (the bin-packer flushes at boundary
// positions), so we exclude it from the search — picking it would yield
// zero overlap, defeating the purpose of this function.
func applyOverlapAligned(runes []rune, curEnd, overlap int, bounds []boundary) int {
	if overlap <= 0 {
		return curEnd
	}
	target := curEnd - overlap
	if target < 0 {
		target = 0
	}
	// Allowed search window: [curEnd - 2*overlap, curEnd)
	windowStart := curEnd - 2*overlap
	if windowStart < 0 {
		windowStart = 0
	}

	// Prefer a semantic boundary closest to the target overlap point.
	// Picking the latest boundary (closest to curEnd) can result in a
	// much smaller actual overlap than configured. Instead, find the
	// boundary whose position is nearest to `target` (curEnd - overlap),
	// so the actual overlap is as close to the configured value as possible.
	bestBound := -1
	bestDist := -1
	for _, b := range bounds {
		if b.runeStart >= windowStart && b.runeStart < curEnd {
			dist := b.runeStart - target
			if dist < 0 {
				dist = -dist
			}
			if bestBound < 0 || dist < bestDist {
				bestBound = b.runeStart
				bestDist = dist
			}
		}
	}
	if bestBound >= 0 {
		return bestBound
	}

	// Fallback: scan backwards from `target` to the previous newline, but
	// not past windowStart so we keep the overlap roughly the right size.
	for i := target; i > windowStart && i < len(runes); i-- {
		if runes[i] == '\n' {
			return i + 1
		}
	}
	return target
}

// groupReferenceEntries merges consecutive tiny chunks that belong to a
// reference/bibliography section into larger groups of up to chunkSize. A
// chunk is considered a reference entry if its content starts with a pattern
// matching ReferenceEntryPattern (e.g. "[1] Author..." or "1. Author...").
// The function scans for the first chunk whose ContextHeader or Content
// indicates a reference section, then groups subsequent entry chunks until
// the accumulated size reaches chunkSize.
func groupReferenceEntries(chunks []Chunk, runes []rune, chunkSize int) []Chunk {
	if len(chunks) <= 2 || chunkSize <= 0 {
		return chunks
	}

	// Find the start of a reference section by checking ContextHeader or
	// Content for reference section markers.
	refStart := -1
	for i, c := range chunks {
		if ReferenceSectionPattern.MatchString(c.Content) || ReferenceSectionPattern.MatchString(c.ContextHeader) {
			refStart = i
			break
		}
	}
	if refStart < 0 {
		return chunks
	}

	// The reference section heading chunk itself stays standalone.
	// Group subsequent chunks that look like reference entries.
	out := make([]Chunk, 0, len(chunks))
	for i := 0; i < refStart; i++ {
		out = append(out, chunks[i])
	}
	out = append(out, chunks[refStart])

	// Accumulate reference entries into groups of up to chunkSize.
	groupStart := refStart + 1
	accumLen := 0
	for i := refStart + 1; i < len(chunks); i++ {
		c := chunks[i]
		entryLen := utf8.RuneCountInString(strings.TrimSpace(c.Content))
		isEntry := ReferenceEntryPattern.MatchString(c.Content)

		if isEntry && accumLen+entryLen <= chunkSize {
			// Accumulate into current group.
			accumLen += entryLen
			continue
		}

		// Flush the accumulated group before starting a new one.
		if i > groupStart {
			groupEnd := chunks[i-1].End
			groupContent := string(runes[chunks[groupStart].Start:groupEnd])
			out = append(out, Chunk{
				Content: groupContent,
				Start:   chunks[groupStart].Start,
				End:     groupEnd,
			})
		}

		if isEntry {
			// Start a new group with this entry.
			groupStart = i
			accumLen = entryLen
		} else {
			// Non-reference entry — emit as standalone.
			out = append(out, c)
			groupStart = i + 1
			accumLen = 0
		}
	}

	// Flush remaining group.
	if groupStart < len(chunks) {
		groupEnd := chunks[len(chunks)-1].End
		groupContent := string(runes[chunks[groupStart].Start:groupEnd])
		out = append(out, Chunk{
			Content: groupContent,
			Start:   chunks[groupStart].Start,
			End:     groupEnd,
		})
	}

	// Re-sequence.
	for i := range out {
		out[i].Seq = i
		populateStructuralMetadata(&out[i])
	}
	return out
}
