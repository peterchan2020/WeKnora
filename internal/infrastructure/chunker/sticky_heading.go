package chunker

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
