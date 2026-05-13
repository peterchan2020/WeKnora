package chunker

import (
	"strings"
	"testing"
)

func TestSemanticSentences_DoesNotSplitOnColonOrSemicolon(t *testing.T) {
	text := "Plan: keep this clause together; it is not a new sentence. Next sentence."
	sentences := semanticSentences(text)
	if len(sentences) != 2 {
		t.Fatalf("expected 2 sentences, got %d: %#v", len(sentences), sentences)
	}
	first := text[sentences[0].start:sentences[0].end]
	if !strings.Contains(first, "together; it") {
		t.Fatalf("first sentence should retain semicolon clause, got %q", first)
	}
	if !strings.Contains(first, "Plan: keep") {
		t.Fatalf("first sentence should retain colon clause, got %q", first)
	}
}
