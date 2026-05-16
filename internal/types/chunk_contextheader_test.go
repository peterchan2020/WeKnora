package types

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestChunkContextHeaderVisibleInJSON(t *testing.T) {
	c := Chunk{
		ID:            "test",
		Content:       "body",
		ContextHeader: "# Heading",
	}
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	// ContextHeader is now persisted and visible in JSON
	if !strings.Contains(string(b), "context_header") {
		t.Errorf("context_header field missing from JSON: %s", string(b))
	}
	if !strings.Contains(string(b), "# Heading") {
		t.Errorf("ContextHeader value missing from JSON: %s", string(b))
	}
}

func TestChunkContextHeaderEmptyOmitted(t *testing.T) {
	c := Chunk{
		ID:      "test",
		Content: "body",
	}
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	// Empty ContextHeader should still appear as empty string in JSON
	// (not omitted, since the field has no omitempty tag)
	if !strings.Contains(string(b), "context_header") {
		t.Errorf("context_header field missing from JSON for empty value: %s", string(b))
	}
}

func TestParsedChunkEmbeddingContent(t *testing.T) {
	pc := ParsedChunk{Content: "body", ContextHeader: "# H"}
	got := pc.EmbeddingContent()
	want := "# H\n\nbody"
	if got != want {
		t.Errorf("EmbeddingContent mismatch: got %q, want %q", got, want)
	}
	pc2 := ParsedChunk{Content: "only body"}
	if pc2.EmbeddingContent() != "only body" {
		t.Errorf("EmbeddingContent without header should equal Content")
	}
}

func TestChunkEmbeddingContent(t *testing.T) {
	c := Chunk{Content: "body", ContextHeader: "# H"}
	got := c.EmbeddingContent()
	want := "# H\n\nbody"
	if got != want {
		t.Errorf("EmbeddingContent mismatch: got %q, want %q", got, want)
	}
	c2 := Chunk{Content: "only body"}
	if c2.EmbeddingContent() != "only body" {
		t.Errorf("EmbeddingContent without header should equal Content")
	}
}