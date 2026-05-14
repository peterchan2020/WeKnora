package chunker

import (
	"fmt"
	"regexp"
	"strings"
)

// TableElement describes a table-like region found inside chunk content.
// Offsets are byte offsets within the inspected text; callers that persist
// document-level rune offsets should convert them at the boundary.
type TableElement struct {
	Kind    string
	Content string
	Columns []string
	Rows    int
	Start   int
	End     int
}

// FormulaRef describes a LaTeX/math region found inside chunk content.
// Kind is one of "block", "display", "inline", or "environment".
type FormulaRef struct {
	Text  string
	Kind  string
	Start int
	End   int
}

// CodeBlock describes a fenced code block found inside chunk content.
type CodeBlock struct {
	Language string
	Content  string
	Start    int
	End      int
}

var (
	markdownTableBlockPattern = regexp.MustCompile(`(?m)(?:^[ \t]*\|[^\n]*\|[ \t]*\r?\n)+`)
	htmlTableBlockPattern     = regexp.MustCompile(`(?is)<table\b[^>]*>.*?</table>`)
	codeBlockPattern          = regexp.MustCompile("(?s)```([A-Za-z0-9_+.#-]*)[ \t]*\r?\n(.*?)```")
	latexBlockPattern         = regexp.MustCompile(`(?s)\$\$.*?\$\$`)
	latexDisplayPattern       = regexp.MustCompile(`(?s)\\\[.*?\\\]`)
	latexParenPattern         = regexp.MustCompile(`(?s)\\\(.*?\\\)`)
	latexInlinePattern        = regexp.MustCompile(`(?s)(^|[^\\$])(\$[^$\n]+?\$)`)
	latexEnvPattern           = regexp.MustCompile(`(?s)\\begin\{[a-zA-Z]+\*?\}.*?\\end\{[a-zA-Z]+\*?\}`)
)

// ExtractTableElements extracts Markdown and HTML tables from text. It is
// intentionally conservative: imperfect Markdown tables are still returned as
// table_text-like elements, but only well-formed Markdown separator rows yield
// column metadata.
func ExtractTableElements(text string) []TableElement {
	var out []TableElement
	for _, loc := range htmlTableBlockPattern.FindAllStringIndex(text, -1) {
		content := text[loc[0]:loc[1]]
		out = append(out, TableElement{
			Kind:    "html",
			Content: content,
			Rows:    countHTMLRows(content),
			Start:   loc[0],
			End:     loc[1],
		})
	}

	for _, loc := range markdownTableBlockPattern.FindAllStringIndex(text, -1) {
		content := strings.TrimRight(text[loc[0]:loc[1]], "\r\n")
		lines := nonEmptyLines(content)
		if len(lines) < 2 || !isMarkdownTableSeparator(lines[1]) {
			continue
		}
		out = append(out, TableElement{
			Kind:    "markdown",
			Content: content,
			Columns: parseMarkdownTableColumns(lines[0]),
			Rows:    countMarkdownDataRows(lines),
			Start:   loc[0],
			End:     loc[0] + len(content),
		})
	}
	return sortTableElements(out)
}

// ExtractFormulaRefs extracts LaTeX formulas in common Markdown forms,
// including \begin{env}...\end{env} environments (equation, align, theorem, etc).
func ExtractFormulaRefs(text string) []FormulaRef {
	var out []FormulaRef
	out = appendFormulaMatches(out, text, latexEnvPattern, "environment", 0)
	out = appendFormulaMatches(out, text, latexBlockPattern, "block", 0)
	out = appendFormulaMatches(out, text, latexDisplayPattern, "display", 0)
	out = appendFormulaMatches(out, text, latexParenPattern, "inline", 0)
	for _, m := range latexInlinePattern.FindAllStringSubmatchIndex(text, -1) {
		if len(m) < 6 || m[4] < 0 || m[5] < 0 {
			continue
		}
		out = append(out, FormulaRef{Text: text[m[4]:m[5]], Kind: "inline", Start: m[4], End: m[5]})
	}
	return sortFormulaRefs(dedupeFormulaRefs(out))
}

// ExtractCodeBlocks extracts fenced code blocks and their language hints.
func ExtractCodeBlocks(text string) []CodeBlock {
	var out []CodeBlock
	for _, m := range codeBlockPattern.FindAllStringSubmatchIndex(text, -1) {
		lang := ""
		if m[2] >= 0 && m[3] >= 0 {
			lang = strings.ToLower(strings.TrimSpace(text[m[2]:m[3]]))
		}
		content := text[m[0]:m[1]]
		out = append(out, CodeBlock{Language: lang, Content: content, Start: m[0], End: m[1]})
	}
	return out
}

func appendFormulaMatches(out []FormulaRef, text string, re *regexp.Regexp, kind string, offset int) []FormulaRef {
	for _, loc := range re.FindAllStringIndex(text, -1) {
		out = append(out, FormulaRef{Text: text[loc[0]:loc[1]], Kind: kind, Start: loc[0] + offset, End: loc[1] + offset})
	}
	return out
}

func nonEmptyLines(text string) []string {
	raw := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	lines := make([]string, 0, len(raw))
	for _, line := range raw {
		if strings.TrimSpace(line) != "" {
			lines = append(lines, strings.TrimSpace(line))
		}
	}
	return lines
}

func isMarkdownTableSeparator(line string) bool {
	cells := markdownCells(line)
	if len(cells) == 0 {
		return false
	}
	for _, cell := range cells {
		cell = strings.Trim(cell, " :-")
		if cell != "" {
			return false
		}
	}
	return strings.Contains(line, "---")
}

func parseMarkdownTableColumns(line string) []string {
	cells := markdownCells(line)
	cols := make([]string, 0, len(cells))
	for i, cell := range cells {
		cell = strings.TrimSpace(cell)
		if cell == "" {
			cell = fmt.Sprintf("column_%d", i+1)
		}
		cols = append(cols, cell)
	}
	return cols
}

func markdownCells(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "|")
	line = strings.TrimSuffix(line, "|")
	parts := strings.Split(line, "|")
	cells := make([]string, 0, len(parts))
	for _, part := range parts {
		cells = append(cells, strings.TrimSpace(part))
	}
	return cells
}

func countMarkdownDataRows(lines []string) int {
	if len(lines) <= 2 {
		return 0
	}
	rows := 0
	for _, line := range lines[2:] {
		if strings.Contains(line, "|") {
			rows++
		}
	}
	return rows
}

var htmlTableRowPattern = regexp.MustCompile(`(?is)<tr\b`)

func countHTMLRows(text string) int {
	return len(htmlTableRowPattern.FindAllStringIndex(text, -1))
}

func sortTableElements(in []TableElement) []TableElement {
	for i := 1; i < len(in); i++ {
		for j := i; j > 0 && in[j].Start < in[j-1].Start; j-- {
			in[j], in[j-1] = in[j-1], in[j]
		}
	}
	return in
}

func sortFormulaRefs(in []FormulaRef) []FormulaRef {
	for i := 1; i < len(in); i++ {
		for j := i; j > 0 && in[j].Start < in[j-1].Start; j-- {
			in[j], in[j-1] = in[j-1], in[j]
		}
	}
	return in
}

func dedupeFormulaRefs(in []FormulaRef) []FormulaRef {
	var out []FormulaRef
	for _, f := range in {
		covered := false
		// Since formulas are sorted by Start, only the last kept formula
		// can contain a new one (wider spans come first in the sort).
		if len(out) > 0 {
			kept := out[len(out)-1]
			if f.Start >= kept.Start && f.End <= kept.End {
				covered = true
			}
		}
		if !covered {
			out = append(out, f)
		}
	}
	return out
}
