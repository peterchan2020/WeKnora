// Package chunker - heading_splitter.go implements Tier 1: Markdown
// heading-aware chunking. Documents with proper heading structure are split
// at heading boundaries and each chunk is prefixed with a breadcrumb of
// active heading context (e.g. "# Chapter 1\n## Section 1.2").
package chunker

import (
	"context"
	"regexp"
	"strings"
	"unicode/utf8"
)

// init wires this implementation into the strategy resolver.
func init() {
	splitByHeadings = splitByHeadingsImpl
}

// headingBoundary marks where a section starts. The first boundary is at
// rune offset 0 (covers any preamble before the first heading), subsequent
// boundaries sit at headings whose level is <= primaryLevel.
type headingBoundary struct {
	runeStart int
	line      string // raw heading line, "" when this is the leading boundary
}

// splitByHeadingsImpl is the Tier-1 implementation. It falls through to the
// legacy splitter when the document has no usable heading structure or when
// the heading split would produce a single section anyway.
//
// profile may be nil; we compute one on demand. When the strategy resolver
// already ran the profiler (auto strategy), the same profile is threaded
// through here so we don't re-scan the entire document.
func splitByHeadingsImpl(text string, cfg SplitterConfig, profile *DocProfile) []Chunk {
	return splitByHeadingsStructured(text, cfg, profile, false)
}

func splitByHeadingsSemantic(
	ctx context.Context,
	text string,
	cfg SplitterConfig,
	profile *DocProfile,
	embedder SemanticEmbedder,
) []Chunk {
	coarse := splitByHeadingsStructured(text, cfg, profile, true)
	refined, err := RefineSemantic(ctx, coarse, cfg, embedder)
	if err != nil {
		return splitByHeadingsImpl(text, cfg, profile)
	}
	if v := ValidateChunks(refined, len([]rune(text)), cfg.ChunkSize); !v.OK {
		return splitByHeadingsImpl(text, cfg, profile)
	}
	return refined
}

func splitByHeadingsStructured(text string, cfg SplitterConfig, profile *DocProfile, keepOversizeSections bool) []Chunk {
	if text == "" {
		return nil
	}
	if profile == nil {
		profile = ProfileDocument(text)
	}
	primaryLevel := profile.DominantHeadingLevel()
	if primaryLevel == 0 {
		return SplitText(text, cfg)
	}

	bounds := findHeadingBoundaries(text, primaryLevel)
	if len(bounds) <= 1 {
		// Single heading at the top (or no heading). For small ChunkSize
		// (child splitting), emit the entire content as one chunk so the
		// heading stays with its body. If oversized, split via legacy but
		// attach the heading to the first chunk's ContextHeader.
		if cfg.ChunkSize < 500 {
			if utf8.RuneCountInString(text) <= cfg.ChunkSize {
				// Heading-only section: no retrievable body content. Skip
				// emitting — the parent already carries this heading context.
				if isHeadingOnlySection(text) {
					return nil
				}
				bc := ""
				if len(bounds) == 1 && bounds[0].line != "" {
					line := bounds[0].line
					if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
						line = normalized
					}
					m := MarkdownHeadingPattern.FindStringSubmatch(line)
					if m != nil {
						bc = strings.Repeat("#", len(m[1])) + " " + strings.TrimSpace(m[2])
					} else if NumberedSectionPattern.MatchString(line) {
						bc = line
					}
				}
				return []Chunk{{
					Content:       text,
					ContextHeader: bc,
					Seq:           0,
					Start:         0,
					End:           utf8.RuneCountInString(text),
				}}
			}
			// Oversized single-heading content: split via legacy, attach
			// heading to first chunk's ContextHeader.
			subChunks := SplitText(text, cfg)
			if len(bounds) == 1 && bounds[0].line != "" {
				line := bounds[0].line
				if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
					line = normalized
				}
				m := MarkdownHeadingPattern.FindStringSubmatch(line)
				var bc string
				if m != nil {
					bc = strings.Repeat("#", len(m[1])) + " " + strings.TrimSpace(m[2])
				} else if NumberedSectionPattern.MatchString(line) {
					bc = line
				}
				if bc != "" && len(subChunks) > 0 {
					subChunks[0].ContextHeader = mergeBreadcrumbs(bc, subChunks[0].ContextHeader)
				}
			}
			return coalesceOrphanHeadings(coalesceTinyChunks(subChunks, cfg.ChunkSize), cfg.ChunkSize)
		}
		return SplitText(text, cfg)
	}

	runes := []rune(text)
	hierarchy := NewHeadingHierarchy()

	// Pre-walk every heading (not just primary-level) so the hierarchy
	// reflects the full nesting context for each section's start. We only
	// snapshot the breadcrumb at section boundaries; deeper sub-headings
	// inside a section update the hierarchy but do not change the chunk's
	// breadcrumb (chunks within a section share one breadcrumb).
	var out []Chunk
	seq := 0

	for i, b := range bounds {
		endRune := len(runes)
		if i+1 < len(bounds) {
			endRune = bounds[i+1].runeStart
		}
		if b.line != "" {
			hierarchy.Observe(b.line)
		}
		// Catch sub-headings that occur between this primary boundary and
		// the next so the hierarchy stays in sync for subsequent sections.
		// We intentionally do this after observing the section header so
		// the breadcrumb reflects the section-leading heading.
		breadcrumb := hierarchy.BreadcrumbWithHashes()
		observeSubHeadings(runes[b.runeStart:endRune], primaryLevel, hierarchy)

		sectionRunes := runes[b.runeStart:endRune]
		sectionContent := NormalizeStickyHeadingsInText(string(sectionRunes))
		secLen := len(sectionRunes)
		if secLen == 0 {
			continue
		}

		// Section body guard: for small ChunkSize (typical in child
		// splitting), skip emitting a section as a standalone chunk when
		// it has less than minSectionBody chars of non-heading content.
		// The heading will be absorbed by coalesceOrphanHeadings into
		// the adjacent section instead, preventing isolated heading-only
		// tiny chunks.
		const minSectionBody = 50
		if cfg.ChunkSize < 500 {
			// Check how much non-heading content this section has.
			nonHeadingLen := 0
			for _, line := range strings.Split(sectionContent, "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				ml := line
				if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
					ml = normalized
				}
				if MarkdownHeadingPattern.MatchString(ml) || NumberedSectionPattern.MatchString(ml) {
					continue
				}
				nonHeadingLen += utf8.RuneCountInString(line)
			}
			if nonHeadingLen < minSectionBody {
				continue // skip — heading gets absorbed by coalesceOrphanHeadings
			}
		}

		bcLen := utf8.RuneCountInString(breadcrumb)
		// Single-chunk section: emit as-is, breadcrumb tracked separately.
		// The breadcrumb is delivered via Chunk.ContextHeader (not Content)
		// to preserve End-Start == len(Content) invariants relied on by
		// document reconstruction (knowledge.go:2278+).
		if bcLen+2+secLen <= cfg.ChunkSize {
			chunk := Chunk{
				Content:       sectionContent,
				ContextHeader: breadcrumb,
				Seq:           seq,
				Start:         b.runeStart,
				End:           endRune,
			}
			populateStructuralMetadata(&chunk)
			out = append(out, chunk)
			seq++
			continue
		}

		if keepOversizeSections {
			chunk := Chunk{
				Content:       sectionContent,
				ContextHeader: breadcrumb,
				Seq:           seq,
				Start:         b.runeStart,
				End:           endRune,
			}
			populateStructuralMetadata(&chunk)
			out = append(out, chunk)
			seq++
			continue
		}

		// Section too large: defer to the legacy splitter for inner
		// segmentation. Sub-chunks inherit the same breadcrumb via
		// ContextHeader. We do NOT shrink the inner ChunkSize budget here
		// because the breadcrumb no longer counts against Content size.
		subChunks := SplitText(sectionContent, cfg)
		for _, sub := range subChunks {
			chunk := Chunk{
				Content:       sub.Content,
				ContextHeader: breadcrumb,
				Seq:           seq,
				Start:         b.runeStart + sub.Start,
				End:           b.runeStart + sub.End,
				Tables:        sub.Tables,
				Formulas:      sub.Formulas,
				CodeBlocks:    sub.CodeBlocks,
				Images:        sub.Images,
				Metadata:      sub.Metadata,
			}
			out = append(out, chunk)
			seq++
		}
	}

	return coalesceOrphanHeadings(coalesceTinyChunks(out, cfg.ChunkSize), cfg.ChunkSize)
}

// coalesceTinyChunks merges adjacent small chunks under their shared heading
// context so that documents whose primary sections are mostly short (FAQs,
// install logs, change-lists) don't trip the validator's "too many tiny
// chunks" rule and fall through all the way to legacy. The merged breadcrumb
// is the line-prefix shared by both inputs; the original sub-headings remain
// visible because heading_splitter includes the heading line in each
// section's Content.
//
// Safety:
//   - We only merge when cur.End == next.Start. That preserves the
//     End-Start == len([]rune(Content)) invariant that document
//     reconstruction relies on, and naturally skips legacy sub-chunks (which
//     may overlap due to ChunkOverlap).
//   - We stop accumulating once the running chunk reaches the merge target
//     (≈ ChunkSize/2) so we don't aggressively pack chunks beyond what the
//     validator considers comfortable.
func coalesceTinyChunks(in []Chunk, chunkSize int) []Chunk {
	if len(in) <= 1 || chunkSize <= 0 {
		return in
	}
	target := chunkSize / 3
	if target < 200 {
		target = 200
	}

	out := make([]Chunk, 0, len(in))
	cur := in[0]
	curLen := utf8.RuneCountInString(cur.Content)

	for i := 1; i < len(in); i++ {
		next := in[i]
		nextLen := utf8.RuneCountInString(next.Content)
		if header, ok := pureHeadingChunkHeader(cur); ok && cur.End == next.Start {
			// Heading-only chunk absorbs the next chunk: merge the heading
			// into the next chunk's ContextHeader so the title is never orphaned.
			// Always merge when combined length fits within ChunkSize, regardless
			// of whether the next chunk is above target.
			if curLen+nextLen <= chunkSize {
				next.ContextHeader = mergeBreadcrumbs(header, next.ContextHeader)
				cur = next
				curLen = nextLen
				continue
			}
			// Too large to merge content — still promote heading to ContextHeader
			// so the next chunk carries the title context even if content stays separate.
			next.ContextHeader = mergeBreadcrumbs(header, next.ContextHeader)
			out = append(out, cur)
			cur = next
			curLen = nextLen
			continue
		}
		// Adjacent + still-small + would not blow the size budget → merge.
		if curLen < target && curLen+nextLen <= chunkSize {
			cur.Content += next.Content
			cur.ContextHeader = commonHeadingPrefix(cur.ContextHeader, next.ContextHeader)
			populateStructuralMetadata(&cur)
			cur.End = next.End
			curLen += nextLen
			continue
		}
		out = append(out, cur)
		cur = next
		curLen = nextLen
	}
	out = append(out, cur)

	// Filter near-empty chunks (<10 trimmed chars) by merging into neighbors.
	out = coalesceNearEmptyChunks(out, chunkSize)

	// Re-sequence — downstream code (knowledge.go) expects Seq to be a dense
	// 0..N-1 range over the returned slice.
	for i := range out {
		out[i].Seq = i
	}
	return out
}

// artifactPattern matches pure-number or uppercase-label patterns that are
// PDF numbering artifacts with no retrievable semantic content, e.g.
// "7.6.4.2.", "7.6.5.", "TABLE IV.", "FIG. 3.".
var artifactPattern = regexp.MustCompile(`^[\d.]+\s*$|^TABLE\s+\w+\.?\s*$|^FIG\.?\s*\w+\.?\s*$`)

// coalesceNearEmptyChunks merges chunks with fewer than 10 trimmed characters
// (or fewer than 20 for artifact patterns like "7.6.4.2." or "TABLE IV.")
// into their nearest neighbor. These are typically PDF artifacts that carry
// no retrievable semantic content on their own.
func coalesceNearEmptyChunks(chunks []Chunk, chunkSize int) []Chunk {
	if len(chunks) <= 2 || chunkSize <= 0 {
		return chunks
	}
	const nearEmptyThreshold = 10
	const artifactThreshold = 20

	isNearEmpty := func(c Chunk) (bool, string) {
		trimmed := strings.TrimSpace(c.Content)
		trimmedLen := utf8.RuneCountInString(trimmed)
		if trimmedLen >= artifactThreshold {
			return false, trimmed
		}
		// Artifact patterns get a higher threshold (20 instead of 10).
		if trimmedLen >= nearEmptyThreshold && trimmedLen < artifactThreshold {
			if artifactPattern.MatchString(trimmed) {
				return true, trimmed
			}
			return false, trimmed
		}
		if trimmedLen < nearEmptyThreshold {
			return true, trimmed
		}
		return false, trimmed
	}

	out := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		empty, trimmed := isNearEmpty(c)
		if !empty {
			out = append(out, c)
			continue
		}
		// Near-empty chunk: merge into previous or next neighbor.
		if len(out) > 0 {
			prev := &out[len(out)-1]
			prevLen := utf8.RuneCountInString(prev.Content)
			chunkLen := utf8.RuneCountInString(c.Content)
			if prevLen+chunkLen+2 <= chunkSize {
				prev.Content = strings.TrimSpace(prev.Content) + "\n" + trimmed
				prev.End = c.End
				populateStructuralMetadata(prev)
				continue
			}
		}
		// Can't merge backward — keep as-is (will be picked up by
		// coalesceOrphanHeadings or the heuristic coalescer).
		out = append(out, c)
	}

	return out
}

func pureHeadingChunkHeader(chunk Chunk) (string, bool) {
	trimmed := strings.TrimSpace(chunk.Content)
	if trimmed == "" {
		return "", false
	}
	// Accept multi-line chunks where every non-blank line is a heading.
	// This handles PDF-extracted headings like "# 1Introduction\n\n"
	// which have a heading line plus trailing blank lines.
	var headingLines []string
	for _, line := range strings.Split(trimmed, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		matchLine := line
		if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
			matchLine = normalized
		}
		m := MarkdownHeadingPattern.FindStringSubmatch(matchLine)
		if m == nil {
			// Also accept numbered section lines without # prefix (common in
			// PDF extraction), e.g. "7.1. Induced and spontaneous mutation".
			if NumberedSectionPattern.MatchString(matchLine) {
				headingLines = append(headingLines, matchLine)
				continue
			}
			return "", false // non-heading content found
		}
		headingLines = append(headingLines, matchLine)
	}
	if len(headingLines) == 0 {
		return "", false
	}
	if chunk.ContextHeader != "" {
		return chunk.ContextHeader, true
	}
	// Build the heading text from all heading lines.
	var resultParts []string
	for _, hl := range headingLines {
		m := MarkdownHeadingPattern.FindStringSubmatch(hl)
		if m != nil {
			resultParts = append(resultParts, strings.Repeat("#", len(m[1]))+" "+strings.TrimSpace(m[2]))
		} else {
			// Numbered section without # prefix: return as-is for context.
			resultParts = append(resultParts, hl)
		}
	}
	return strings.Join(resultParts, "\n"), true
}

// isHeadingOnlySection returns true when the section contains no retrievable
// body content — just Markdown headings, numbered sections, or blank lines.
// Used by splitByHeadingsStructured to skip emitting tiny heading-only chunks.
func isHeadingOnlySection(text string) bool {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		ml := line
		if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
			ml = normalized
		}
		if MarkdownHeadingPattern.MatchString(ml) || NumberedSectionPattern.MatchString(ml) {
			continue
		}
		return false // found non-heading content
	}
	return true
}

// commonHeadingPrefix returns the longest line-aligned prefix shared by two
// breadcrumb strings. Heading hierarchies are emitted as
// "# Top\n## Section\n### Sub", so a line-by-line comparison is sufficient
// and avoids partial-line truncation that would corrupt the breadcrumb.
func commonHeadingPrefix(a, b string) string {
	if a == b {
		return a
	}
	la := strings.Split(a, "\n")
	lb := strings.Split(b, "\n")
	n := len(la)
	if len(lb) < n {
		n = len(lb)
	}
	common := 0
	for i := 0; i < n; i++ {
		if la[i] != lb[i] {
			break
		}
		common = i + 1
	}
	if common == 0 {
		return ""
	}
	return strings.Join(la[:common], "\n")
}

// findHeadingBoundaries returns one boundary at offset 0 plus one per
// Markdown heading at level <= primaryLevel that sits outside fenced code
// blocks. Heading detection is line-oriented — a heading must occupy a
// whole line to be recognized.
func findHeadingBoundaries(text string, primaryLevel int) []headingBoundary {
	runes := []rune(text)
	bounds := []headingBoundary{{runeStart: 0}}
	if len(runes) == 0 {
		return bounds
	}

	pos := 0
	inFence := false
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			pos += utf8.RuneCountInString(line)
			if i < len(lines)-1 {
				pos++ // newline
			}
			continue
		}
		if !inFence {
			matchLine := line
			if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
				matchLine = normalized
			}
			m := MarkdownHeadingPattern.FindStringSubmatch(matchLine)
			if m != nil {
				level := len(m[1])
				if level >= 1 && level <= primaryLevel && pos > 0 {
					bounds = append(bounds, headingBoundary{
						runeStart: pos,
						line:      matchLine,
					})
				}
				if level >= 1 && level <= primaryLevel && pos == 0 {
					// First line is a heading — replace the leading boundary
					bounds[0].line = matchLine
				}
			}
		}
		pos += utf8.RuneCountInString(line)
		if i < len(lines)-1 {
			pos++ // account for the \n that strings.Split removed
		}
	}
	return bounds
}

// observeSubHeadings walks the section's lines and feeds every Markdown
// heading deeper than primaryLevel into the hierarchy. This keeps the
// hierarchy state correct so the breadcrumb at the next primary section
// reflects the truly active stack.
func observeSubHeadings(runes []rune, primaryLevel int, h *HeadingHierarchy) {
	if len(runes) == 0 {
		return
	}
	text := string(runes)
	inFence := false
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		matchLine := line
		if normalized, ok := normalizeStickyPDFHeadingLine(line); ok {
			matchLine = normalized
		}
		m := MarkdownHeadingPattern.FindStringSubmatch(matchLine)
		if m == nil {
			continue
		}
		level := len(m[1])
		if level > primaryLevel {
			h.Observe(matchLine)
		}
	}
}
