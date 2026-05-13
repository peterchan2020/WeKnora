package chunker

import (
	"strings"
	"testing"
)

func TestSplitText_HTMLTableRespectsChunkBudget(t *testing.T) {
	var b strings.Builder
	b.WriteString("before\n<TABLE>")
	for i := 0; i < 24; i++ {
		b.WriteString("<TR><TD>")
		b.WriteString(strings.Repeat("cell", 8))
		b.WriteString("</TD></TR>")
	}
	b.WriteString("</TABLE>\nafter")

	cfg := SplitterConfig{
		ChunkSize:    120,
		ChunkOverlap: 0,
		Separators:   []string{"\n\n", "\n", " "},
	}
	chunks := SplitText(b.String(), cfg)
	if len(chunks) < 2 {
		t.Fatalf("expected HTML table to split into multiple chunks, got %d", len(chunks))
	}

	maxSize := maxChunkSizeForBudget(cfg.ChunkSize)
	for i, chunk := range chunks {
		if got := runeLen(chunk.Content); got > maxSize {
			t.Fatalf("chunk[%d] length = %d, want <= %d", i, got, maxSize)
		}
		if strings.Contains(chunk.Content, "<TR>") &&
			strings.Count(chunk.Content, "<TR>") != strings.Count(chunk.Content, "</TR>") {
			t.Fatalf("chunk[%d] split an HTML table row: %q", i, chunk.Content)
		}
	}
}
