package chunker

import "strings"

// normalizeStickyPDFHeadingLine returns a matching-only normalization for
// headings produced by PDF parsers with missing spaces. It does not alter the
// original chunk content; callers use the normalized line only for boundary
// detection, profiling, and breadcrumb construction.
func normalizeStickyPDFHeadingLine(line string) (string, bool) {
	if m := StickyMarkdownHeadingPattern.FindStringSubmatch(line); m != nil {
		return m[1] + " " + m[2] + " " + m[3], true
	}
	if m := StickyNumberedSectionPattern.FindStringSubmatch(line); m != nil {
		return m[1] + " " + m[2], true
	}
	return line, false
}

func isStickyMarkdownHeadingLine(line string) bool {
	return StickyMarkdownHeadingPattern.MatchString(line)
}

// NormalizeStickyHeadingsInText fixes all sticky PDF heading lines in the
// given text by inserting a space between the section number and the title.
// Each line is processed independently so only heading lines are modified.
// This improves vector search quality by making headings like "# 1Introduction"
// become "# 1 Introduction" which is more readable and better matched by
// embedding models.
func NormalizeStickyHeadingsInText(text string) string {
	if text == "" {
		return text
	}
	lines := strings.Split(text, "\n")
	changed := false
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if normalized, ok := normalizeStickyPDFHeadingLine(trimmed); ok {
			// Preserve leading whitespace from the original line.
			leading := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
			lines[i] = leading + normalized
			changed = true
		}
	}
	if !changed {
		return text
	}
	return strings.Join(lines, "\n")
}