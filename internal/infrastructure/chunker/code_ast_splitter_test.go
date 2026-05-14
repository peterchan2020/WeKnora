package chunker

import (
	"strings"
	"testing"
)

func TestSplitFencedCodeUnitWithTreeSitter_SplitsOnTopLevelNodes(t *testing.T) {
	block := strings.Join([]string{
		"```go",
		"package demo",
		"",
		"import \"fmt\"",
		"",
		"func Alpha() {",
		"\tfmt.Println(\"alpha\")",
		"}",
		"",
		"func Beta() {",
		"\tfmt.Println(\"beta\")",
		"}",
		"",
		"func Gamma() {",
		"\tfmt.Println(\"gamma\")",
		"}",
		"```",
	}, "\n")

	units := splitFencedCodeUnitWithTreeSitter(block, 7, 70)
	if len(units) < 2 {
		t.Fatalf("expected AST splitter to produce multiple units, got %d", len(units))
	}
	var reconstructed strings.Builder
	for _, unit := range units {
		reconstructed.WriteString(unit.text)
		if span := unit.end - unit.start; span != runeLen(unit.text) {
			t.Fatalf("unit span mismatch: span=%d text runes=%d unit=%q", span, runeLen(unit.text), unit.text)
		}
	}
	if reconstructed.String() != block {
		t.Fatalf("AST split should preserve the original fenced code block")
	}
	for _, unit := range units {
		if strings.Contains(unit.text, "func Alpha") && strings.Contains(unit.text, "func Gamma") {
			t.Fatalf("expected top-level functions to be split across units, got %q", unit.text)
		}
	}
}

func TestSplitFencedCodeUnitWithTreeSitter_UnknownLanguageFallsBack(t *testing.T) {
	block := "```unknown\none\ntwo\nthree\n```"
	if units := splitFencedCodeUnitWithTreeSitter(block, 0, 8); len(units) != 0 {
		t.Fatalf("unknown language should not use Tree-sitter, got %d units", len(units))
	}
}
