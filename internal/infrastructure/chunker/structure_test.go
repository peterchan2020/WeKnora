package chunker

import (
	"fmt"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// 1. ExtractTableElements
// ---------------------------------------------------------------------------

func TestExtractTableElements_MarkdownTable(t *testing.T) {
	text := "Some intro\n\n| Name | Age | City |\n| --- | --- | --- |\n| Alice | 30 | NYC |\n| Bob | 25 | LA |\n\nMore text"
	tables := ExtractTableElements(text)
	if len(tables) != 1 {
		t.Fatalf("expected 1 table, got %d", len(tables))
	}
	tbl := tables[0]
	if tbl.Kind != "markdown" {
		t.Errorf("Kind = %q, want %q", tbl.Kind, "markdown")
	}
	wantCols := []string{"Name", "Age", "City"}
	if len(tbl.Columns) != len(wantCols) {
		t.Errorf("Columns length = %d, want %d", len(tbl.Columns), len(wantCols))
	} else {
		for i, col := range wantCols {
			if tbl.Columns[i] != col {
				t.Errorf("Columns[%d] = %q, want %q", i, tbl.Columns[i], col)
			}
		}
	}
	if tbl.Rows != 2 {
		t.Errorf("Rows = %d, want 2", tbl.Rows)
	}
	if text[tbl.Start:tbl.End] != tbl.Content {
		t.Errorf("byte offset slice does not match Content")
	}
}

func TestExtractTableElements_HTMLTable(t *testing.T) {
	text := "Before\n<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>\nAfter"
	tables := ExtractTableElements(text)
	if len(tables) != 1 {
		t.Fatalf("expected 1 table, got %d", len(tables))
	}
	tbl := tables[0]
	if tbl.Kind != "html" {
		t.Errorf("Kind = %q, want %q", tbl.Kind, "html")
	}
	if tbl.Rows != 2 {
		t.Errorf("Rows = %d, want 2", tbl.Rows)
	}
	if text[tbl.Start:tbl.End] != tbl.Content {
		t.Errorf("byte offset slice does not match Content")
	}
}

func TestExtractTableElements_MixedTables(t *testing.T) {
	mdTable := "| H1 | H2 |\n| --- | --- |\n| a | b |\n"
	htmlTable := "<table><tr><td>x</td></tr></table>"
	text := mdTable + "\n" + htmlTable
	tables := ExtractTableElements(text)
	if len(tables) != 2 {
		t.Fatalf("expected 2 tables, got %d", len(tables))
	}
	// Verify sorted by Start
	for i := 1; i < len(tables); i++ {
		if tables[i].Start < tables[i-1].Start {
			t.Errorf("tables not sorted by Start: tables[%d].Start=%d > tables[%d].Start=%d", i-1, tables[i-1].Start, i, tables[i].Start)
		}
	}
	kinds := map[string]bool{}
	for _, tbl := range tables {
		kinds[tbl.Kind] = true
	}
	if !kinds["markdown"] || !kinds["html"] {
		t.Errorf("expected both markdown and html kinds, got %v", kinds)
	}
}

func TestExtractTableElements_BrokenTable(t *testing.T) {
	// A Markdown table with inconsistent column counts in the data rows.
	// The separator row is well-formed so the table is still extracted.
	text := "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |\n"
	tables := ExtractTableElements(text)
	if len(tables) != 1 {
		// The implementation is conservative: imperfect tables are still returned.
		// If the implementation skips it, that is also acceptable.
		t.Logf("broken table: got %d tables (implementation may skip)", len(tables))
		return
	}
	tbl := tables[0]
	if tbl.Kind != "markdown" {
		t.Errorf("Kind = %q, want %q", tbl.Kind, "markdown")
	}
	if tbl.Rows < 1 {
		t.Errorf("Rows = %d, want at least 1", tbl.Rows)
	}
}

func TestExtractTableElements_EmptyTableHeader(t *testing.T) {
	text := "| | |\n| --- | --- |\n| val1 | val2 |\n"
	tables := ExtractTableElements(text)
	if len(tables) != 1 {
		t.Fatalf("expected 1 table, got %d", len(tables))
	}
	tbl := tables[0]
	wantCols := []string{"column_1", "column_2"}
	if len(tbl.Columns) != len(wantCols) {
		t.Fatalf("Columns length = %d, want %d", len(tbl.Columns), len(wantCols))
	}
	for i, want := range wantCols {
		if tbl.Columns[i] != want {
			t.Errorf("Columns[%d] = %q, want %q", i, tbl.Columns[i], want)
		}
	}
}

// ---------------------------------------------------------------------------
// 2. ExtractFormulaRefs
// ---------------------------------------------------------------------------

func TestExtractFormulaRefs_BlockMath(t *testing.T) {
	text := "Some text $$E = mc^2$$ more text"
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "block" {
			found = true
			if r.Text != "$$E = mc^2$$" {
				t.Errorf("Text = %q, want %q", r.Text, "$$E = mc^2$$")
			}
			if text[r.Start:r.End] != r.Text {
				t.Errorf("byte offset slice does not match Text")
			}
		}
	}
	if !found {
		t.Error("expected a block formula ref, found none")
	}
}

func TestExtractFormulaRefs_DisplayMath(t *testing.T) {
	text := "Some text \\[E = mc^2\\] more text"
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "display" {
			found = true
			if r.Text != "\\[E = mc^2\\]" {
				t.Errorf("Text = %q, want %q", r.Text, "\\[E = mc^2\\]")
			}
			if text[r.Start:r.End] != r.Text {
				t.Errorf("byte offset slice does not match Text")
			}
		}
	}
	if !found {
		t.Error("expected a display formula ref, found none")
	}
}

func TestExtractFormulaRefs_InlineParen(t *testing.T) {
	text := "Some text \\(x^2\\) more text"
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "inline" {
			found = true
			if r.Text != "\\(x^2\\)" {
				t.Errorf("Text = %q, want %q", r.Text, "\\(x^2\\)")
			}
			if text[r.Start:r.End] != r.Text {
				t.Errorf("byte offset slice does not match Text")
			}
		}
	}
	if !found {
		t.Error("expected an inline (paren) formula ref, found none")
	}
}

func TestExtractFormulaRefs_InlineDollar(t *testing.T) {
	text := "Some text $x^2$ more text"
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "inline" && strings.Contains(r.Text, "$") && !strings.Contains(r.Text, "$$") {
			found = true
			if r.Text != "$x^2$" {
				t.Errorf("Text = %q, want %q", r.Text, "$x^2$")
			}
			if text[r.Start:r.End] != r.Text {
				t.Errorf("byte offset slice does not match Text")
			}
		}
	}
	if !found {
		t.Error("expected an inline (dollar) formula ref, found none")
	}
}

func TestExtractFormulaRefs_MixedFormulas(t *testing.T) {
	text := "Block: $$E=mc^2$$\nDisplay: \\[a+b\\]\nInline paren: \\(x^2\\)\nInline dollar: $y=1$\n"
	refs := ExtractFormulaRefs(text)
	if len(refs) < 4 {
		t.Fatalf("expected at least 4 formula refs, got %d", len(refs))
	}
	// Verify sorted by Start
	for i := 1; i < len(refs); i++ {
		if refs[i].Start < refs[i-1].Start {
			t.Errorf("refs not sorted by Start at index %d", i)
		}
	}
	// Verify no duplicate spans
	for i := 0; i < len(refs); i++ {
		for j := i + 1; j < len(refs); j++ {
			if refs[i].Start == refs[j].Start && refs[i].End == refs[j].End {
				t.Errorf("duplicate formula ref at Start=%d End=%d", refs[i].Start, refs[i].End)
			}
		}
	}
	// Verify byte offsets match
	for _, r := range refs {
		if text[r.Start:r.End] != r.Text {
			t.Errorf("byte offset mismatch for ref %q: text[%d:%d]=%q", r.Text, r.Start, r.End, text[r.Start:r.End])
		}
	}
}

func TestExtractFormulaRefs_FormulaNotSplit(t *testing.T) {
	// Verify that SplitText with a small chunk size does NOT split a formula
	// across chunks. Place $$E=mc^2$$ in the middle of text.
	prefix := strings.Repeat("word ", 30) // ~150 chars
	formula := "$$E=mc^2$$"
	suffix := strings.Repeat("word ", 30) // ~150 chars
	text := prefix + formula + suffix

	cfg := SplitterConfig{
		ChunkSize:    80,
		ChunkOverlap: 10,
		Separators:   []string{"\n\n", "\n", " "},
	}
	chunks := SplitText(text, cfg)

	// Find the byte range of the formula in the original text
	formulaStart := strings.Index(text, formula)
	_ = formulaStart + len(formula) // formulaEnd byte offset (used for rune conversion below)

	for i, ch := range chunks {
		// If the full formula is in this chunk, that's fine.
		if strings.Contains(ch.Content, formula) {
			continue
		}
		// Check that no partial formula appears at the boundary.
		// A partial formula would have an odd number of $$ delimiters.
		if strings.Contains(ch.Content, "$$") {
			count := strings.Count(ch.Content, "$$")
			if count%2 != 0 {
				t.Errorf("chunk %d has odd number of $$ delimiters, formula may be split: %q", i, ch.Content)
			}
		}
		// Also check that no chunk boundary falls within the formula's byte range.
		// Chunk Start/End are rune offsets; convert formula byte offsets to rune offsets.
		formulaRuneStart := runeLen(text[:formulaStart])
		formulaRuneEnd := formulaRuneStart + runeLen(formula)
		if ch.End > formulaRuneStart && ch.Start < formulaRuneEnd {
			// This chunk overlaps the formula range. Verify the full formula is present.
			if !strings.Contains(ch.Content, formula) {
				t.Errorf("chunk %d overlaps formula rune range [%d,%d) but does not contain full formula; content=%q", i, formulaRuneStart, formulaRuneEnd, ch.Content)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// 3. ExtractCodeBlocks
// ---------------------------------------------------------------------------

func TestExtractCodeBlocks_GoBlock(t *testing.T) {
	text := "Before\n```go\nfmt.Println(\"hello\")\n```\nAfter"
	blocks := ExtractCodeBlocks(text)
	if len(blocks) != 1 {
		t.Fatalf("expected 1 code block, got %d", len(blocks))
	}
	b := blocks[0]
	if b.Language != "go" {
		t.Errorf("Language = %q, want %q", b.Language, "go")
	}
	if !strings.Contains(b.Content, "```go") || !strings.Contains(b.Content, "```") {
		t.Errorf("Content should include fences, got %q", b.Content)
	}
	if text[b.Start:b.End] != b.Content {
		t.Errorf("byte offset slice does not match Content")
	}
}

func TestExtractCodeBlocks_NoLanguage(t *testing.T) {
	text := "Before\n```\nsome code\n```\nAfter"
	blocks := ExtractCodeBlocks(text)
	if len(blocks) != 1 {
		t.Fatalf("expected 1 code block, got %d", len(blocks))
	}
	b := blocks[0]
	if b.Language != "" {
		t.Errorf("Language = %q, want empty string", b.Language)
	}
}

func TestExtractCodeBlocks_Multiple(t *testing.T) {
	text := "```go\ncode1\n```\n\n```python\ncode2\n```\n\n```javascript\ncode3\n```"
	blocks := ExtractCodeBlocks(text)
	if len(blocks) != 3 {
		t.Fatalf("expected 3 code blocks, got %d", len(blocks))
	}
	wantLangs := []string{"go", "python", "javascript"}
	for i, want := range wantLangs {
		if blocks[i].Language != want {
			t.Errorf("blocks[%d].Language = %q, want %q", i, blocks[i].Language, want)
		}
	}
	// Verify sorted by Start
	for i := 1; i < len(blocks); i++ {
		if blocks[i].Start < blocks[i-1].Start {
			t.Errorf("code blocks not sorted by Start at index %d", i)
		}
	}
}

// ---------------------------------------------------------------------------
// 4. ExtractImageRefs
// ---------------------------------------------------------------------------

func TestExtractImageRefs_SimpleImage(t *testing.T) {
	text := "See ![alt text](image.png) for details"
	refs := ExtractImageRefs(text)
	if len(refs) != 1 {
		t.Fatalf("expected 1 image ref, got %d", len(refs))
	}
	r := refs[0]
	if r.OriginalRef != "image.png" {
		t.Errorf("OriginalRef = %q, want %q", r.OriginalRef, "image.png")
	}
	if r.AltText != "alt text" {
		t.Errorf("AltText = %q, want %q", r.AltText, "alt text")
	}
	if text[r.Start:r.End] != "![alt text](image.png)" {
		t.Errorf("byte offset slice does not match expected image syntax")
	}
}

func TestExtractImageRefs_MultipleImages(t *testing.T) {
	text := "![img1](a.png) and ![img2](b.png)"
	refs := ExtractImageRefs(text)
	if len(refs) != 2 {
		t.Fatalf("expected 2 image refs, got %d", len(refs))
	}
	wantRefs := []string{"a.png", "b.png"}
	for i, want := range wantRefs {
		if refs[i].OriginalRef != want {
			t.Errorf("refs[%d].OriginalRef = %q, want %q", i, refs[i].OriginalRef, want)
		}
	}
}

func TestExtractImageRefs_ImageWithComplexURL(t *testing.T) {
	text := "![diagram](https://example.com/path_(item)/image.png)"
	refs := ExtractImageRefs(text)
	if len(refs) != 1 {
		t.Fatalf("expected 1 image ref, got %d", len(refs))
	}
	r := refs[0]
	wantURL := "https://example.com/path_(item)/image.png"
	if r.OriginalRef != wantURL {
		t.Errorf("OriginalRef = %q, want %q", r.OriginalRef, wantURL)
	}
	if r.AltText != "diagram" {
		t.Errorf("AltText = %q, want %q", r.AltText, "diagram")
	}
}

// ---------------------------------------------------------------------------
// 5. SplitText structural metadata integration
// ---------------------------------------------------------------------------

func TestSplitText_TableMetadata(t *testing.T) {
	text := "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |\n"
	cfg := SplitterConfig{ChunkSize: 512, ChunkOverlap: 0, Separators: []string{"\n\n", "\n"}}
	chunks := SplitText(text, cfg)
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	found := false
	for _, ch := range chunks {
		if len(ch.Tables) > 0 {
			found = true
			tc, ok := ch.Metadata["table_count"]
			if !ok {
				t.Error("Metadata missing table_count")
			} else if tc.(int) != len(ch.Tables) {
				t.Errorf("table_count = %v, want %d", tc, len(ch.Tables))
			}
			if _, ok := ch.Metadata["table_columns"]; !ok {
				t.Error("Metadata missing table_columns")
			}
			if _, ok := ch.Metadata["table_summary"]; !ok {
				t.Error("Metadata missing table_summary")
			}
		}
	}
	if !found {
		t.Error("no chunk contains table metadata")
	}
}

func TestSplitText_FormulaMetadata(t *testing.T) {
	text := "The energy equation is $$E=mc^2$$ and the inline version is $E=mc^2$."
	cfg := SplitterConfig{ChunkSize: 512, ChunkOverlap: 0, Separators: []string{"\n\n", "\n"}}
	chunks := SplitText(text, cfg)
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	found := false
	for _, ch := range chunks {
		if len(ch.Formulas) > 0 {
			found = true
			fc, ok := ch.Metadata["formula_count"]
			if !ok {
				t.Error("Metadata missing formula_count")
			} else if fc.(int) != len(ch.Formulas) {
				t.Errorf("formula_count = %v, want %d", fc, len(ch.Formulas))
			}
		}
	}
	if !found {
		t.Error("no chunk contains formula metadata")
	}
}

func TestSplitText_CodeBlockMetadata(t *testing.T) {
	text := "```go\nfmt.Println(\"hello\")\n```\n"
	cfg := SplitterConfig{ChunkSize: 512, ChunkOverlap: 0, Separators: []string{"\n\n", "\n"}}
	chunks := SplitText(text, cfg)
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	found := false
	for _, ch := range chunks {
		if len(ch.CodeBlocks) > 0 {
			found = true
			cc, ok := ch.Metadata["code_block_count"]
			if !ok {
				t.Error("Metadata missing code_block_count")
			} else if cc.(int) != len(ch.CodeBlocks) {
				t.Errorf("code_block_count = %v, want %d", cc, len(ch.CodeBlocks))
			}
		}
	}
	if !found {
		t.Error("no chunk contains code block metadata")
	}
}

func TestSplitText_ImageMetadata(t *testing.T) {
	text := "See ![diagram](img.png) for details.\n"
	cfg := SplitterConfig{ChunkSize: 512, ChunkOverlap: 0, Separators: []string{"\n\n", "\n"}}
	chunks := SplitText(text, cfg)
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk")
	}
	found := false
	for _, ch := range chunks {
		if len(ch.Images) > 0 {
			found = true
			ic, ok := ch.Metadata["image_count"]
			if !ok {
				t.Error("Metadata missing image_count")
			} else if ic.(int) != len(ch.Images) {
				t.Errorf("image_count = %v, want %d", ic, len(ch.Images))
			}
		}
	}
	if !found {
		t.Error("no chunk contains image metadata")
	}
}

// ---------------------------------------------------------------------------
// 6. Mixed document golden test
// ---------------------------------------------------------------------------

func TestSplitText_MixedDocument(t *testing.T) {
	doc := strings.Join([]string{
		"# Physics Reference",
		"",
		"The energy-mass equivalence is fundamental:",
		"",
		"$$E=mc^2$$",
		"",
		"And the inline version $E=mc^2$ appears often.",
		"",
		"## Constants Table",
		"",
		"| Symbol | Value | Unit |",
		"| --- | --- | --- |",
		"| c | 299792458 | m/s |",
		"| h | 6.626e-34 | J*s |",
		"| G | 6.674e-11 | N*m^2/kg^2 |",
		"",
		"## HTML Data",
		"",
		"<table><tr><td>Row1Col1</td><td>Row1Col2</td></tr><tr><td>Row2Col1</td><td>Row2Col2</td></tr></table>",
		"",
		"## Code Example",
		"",
		"```go",
		"package main",
		"",
		"import \"fmt\"",
		"",
		"func main() {",
		"\tfmt.Println(\"hello\")",
		"}",
		"```",
		"",
		"## Diagrams",
		"",
		"![energy diagram](energy.png)",
		"",
		"![wave diagram](wave.png)",
		"",
		"The quantum expression \\(\\psi(x)\\) is also important.",
		"",
		"Regular paragraph with enough text to fill space. " +
			"This is additional content to make the document longer. " +
			"We need sufficient text to exercise the chunking logic. " +
			"More filler text here to pad the document length. " +
			"Still more text to ensure multiple chunks are produced. " +
			"Final padding text to reach the chunk boundary.",
	}, "\n")

	cfg := SplitterConfig{
		ChunkSize:    200,
		ChunkOverlap: 20,
		Separators:   []string{"\n\n", "\n", " "},
	}
	chunks := SplitText(doc, cfg)

	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk from mixed document")
	}

	// 6a. No chunk has a formula split across boundaries.
	for i, ch := range chunks {
		// Check $$ delimiters are paired
		blockDollarCount := strings.Count(ch.Content, "$$")
		if blockDollarCount%2 != 0 {
			t.Errorf("chunk %d has odd number of $$ delimiters, block formula may be split", i)
		}
		// Check \[ \] are paired
		openDisplay := strings.Count(ch.Content, "\\[")
		closeDisplay := strings.Count(ch.Content, "\\]")
		if openDisplay != closeDisplay {
			t.Errorf("chunk %d has unbalanced \\[ (%d) / \\] (%d) delimiters", i, openDisplay, closeDisplay)
		}
		// Check \( \) are paired
		openInline := strings.Count(ch.Content, "\\(")
		closeInline := strings.Count(ch.Content, "\\)")
		if openInline != closeInline {
			t.Errorf("chunk %d has unbalanced \\( (%d) / \\) (%d) delimiters", i, openInline, closeInline)
		}
	}

	// 6b. Table elements are found in the appropriate chunks.
	totalTables := 0
	for _, ch := range chunks {
		totalTables += len(ch.Tables)
	}
	if totalTables < 1 {
		t.Error("expected at least one table element across all chunks")
	}

	// 6c. Code blocks are found in the appropriate chunks.
	totalCodeBlocks := 0
	for _, ch := range chunks {
		totalCodeBlocks += len(ch.CodeBlocks)
	}
	if totalCodeBlocks < 1 {
		t.Error("expected at least one code block across all chunks")
	}

	// 6d. Images are found in the appropriate chunks.
	totalImages := 0
	for _, ch := range chunks {
		totalImages += len(ch.Images)
	}
	if totalImages < 1 {
		t.Error("expected at least one image across all chunks")
	}

	// 6e. All structural metadata counts are correct.
	for i, ch := range chunks {
		tc, _ := ch.Metadata["table_count"].(int)
		if tc != len(ch.Tables) {
			t.Errorf("chunk %d: table_count=%d but len(Tables)=%d", i, tc, len(ch.Tables))
		}
		fc, _ := ch.Metadata["formula_count"].(int)
		if fc != len(ch.Formulas) {
			t.Errorf("chunk %d: formula_count=%d but len(Formulas)=%d", i, fc, len(ch.Formulas))
		}
		cc, _ := ch.Metadata["code_block_count"].(int)
		if cc != len(ch.CodeBlocks) {
			t.Errorf("chunk %d: code_block_count=%d but len(CodeBlocks)=%d", i, cc, len(ch.CodeBlocks))
		}
		ic, _ := ch.Metadata["image_count"].(int)
		if ic != len(ch.Images) {
			t.Errorf("chunk %d: image_count=%d but len(Images)=%d", i, ic, len(ch.Images))
		}
	}

	// 6f. Lines that fit within the configured chunk budget and do not embed
	// protected sub-spans should survive as intact chunk content. Longer prose
	// may be split by separators; lines with inline protected spans may split
	// around those spans. The assertions above cover structure-specific
	// invariants.
	originalLines := strings.Split(doc, "\n")
	for _, line := range originalLines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || runeLen(trimmed) > cfg.ChunkSize || containsEmbeddedProtectedSpan(trimmed) {
			continue
		}
		found := false
		for _, ch := range chunks {
			if strings.Contains(ch.Content, trimmed) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("original line %q not found in any chunk", trimmed)
		}
	}
}

func containsEmbeddedProtectedSpan(line string) bool {
	for _, pattern := range protectedPatterns {
		loc := pattern.FindStringIndex(line)
		if loc != nil && (loc[0] > 0 || loc[1] < len(line)) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// 7. LaTeX environment protection
// ---------------------------------------------------------------------------

func TestExtractFormulaRefs_LaTeXEnvironment(t *testing.T) {
	text := "Before \\begin{equation}E = mc^2\\end{equation} after"
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "environment" {
			found = true
			want := `\begin{equation}E = mc^2\end{equation}`
			if r.Text != want {
				t.Errorf("Text = %q, want %q", r.Text, want)
			}
			if text[r.Start:r.End] != r.Text {
				t.Errorf("byte offset slice does not match Text")
			}
		}
	}
	if !found {
		t.Error("expected an environment formula ref, found none")
	}
}

func TestExtractFormulaRefs_LaTeXAlignEnv(t *testing.T) {
	text := "Proof:\n\\begin{align}a &= b + c \\\\ d &= e\\end{align}\nDone."
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "environment" {
			found = true
			if !strings.Contains(r.Text, `\begin{align}`) || !strings.Contains(r.Text, `\end{align}`) {
				t.Errorf("environment ref should contain begin/end align, got %q", r.Text)
			}
			if text[r.Start:r.End] != r.Text {
				t.Errorf("byte offset slice does not match Text")
			}
		}
	}
	if !found {
		t.Error("expected an align environment formula ref, found none")
	}
}

func TestExtractFormulaRefs_LaTeXTheoremEnv(t *testing.T) {
	text := "\\begin{theorem}Every bounded sequence has a convergent subsequence.\\end{theorem}"
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "environment" {
			found = true
		}
	}
	if !found {
		t.Error("expected a theorem environment formula ref, found none")
	}
}

func TestExtractFormulaRefs_EnvironmentNotSplit(t *testing.T) {
	// Verify that SplitText does NOT split a LaTeX environment across chunks.
	prefix := strings.Repeat("word ", 30)
	env := `\begin{equation}E=mc^2\end{equation}`
	suffix := strings.Repeat("word ", 30)
	text := prefix + env + suffix

	cfg := SplitterConfig{
		ChunkSize:    80,
		ChunkOverlap: 10,
		Separators:   []string{"\n\n", "\n", " "},
	}
	chunks := SplitText(text, cfg)

	envStart := strings.Index(text, env)
	envRuneStart := runeLen(text[:envStart])
	envRuneEnd := envRuneStart + runeLen(env)

	for i, ch := range chunks {
		// If the full environment is in this chunk, that's fine.
		if strings.Contains(ch.Content, env) {
			continue
		}
		// Check that no chunk boundary falls within the environment's rune range.
		if ch.End > envRuneStart && ch.Start < envRuneEnd {
			t.Errorf("chunk %d overlaps environment rune range [%d,%d) but does not contain full environment; content=%q", i, envRuneStart, envRuneEnd, ch.Content)
		}
	}
}

func TestExtractFormulaRefs_StarEnv(t *testing.T) {
	text := `\\begin{align*}x &= y\\end{align*}`
	refs := ExtractFormulaRefs(text)
	found := false
	for _, r := range refs {
		if r.Kind == "environment" {
			found = true
		}
	}
	if !found {
		t.Error("expected an align* environment formula ref, found none")
	}
}

// ---------------------------------------------------------------------------
// 8. Code AST splitter: fence preservation
// ---------------------------------------------------------------------------

func TestSplitFencedCodeUnitWithTreeSitter_PreservesFences(t *testing.T) {
	// Build a large Go code block that will exceed maxSize and be split.
	lines := []string{"```go", "package main"}
	for i := 0; i < 30; i++ {
		lines = append(lines, "")
		lines = append(lines, fmt.Sprintf("func Func%02d() { println(%d) }", i, i))
	}
	lines = append(lines, "```")
	block := strings.Join(lines, "\n")

	units := splitFencedCodeUnitWithTreeSitter(block, 7, 200)
	if len(units) < 2 {
		t.Fatalf("expected multiple split units, got %d", len(units))
	}

	// Verify byte-for-byte reconstruction.
	var reconstructed strings.Builder
	for _, unit := range units {
		reconstructed.WriteString(unit.text)
	}
	if reconstructed.String() != block {
		t.Fatalf("AST split should preserve the original fenced code block exactly")
	}

	// Verify each unit's rune span matches its text length.
	for i, unit := range units {
		span := unit.end - unit.start
		textRunes := runeLen(unit.text)
		if span != textRunes {
			t.Errorf("unit %d: span=%d but text runes=%d", i, span, textRunes)
		}
	}
}
