package chunker

import (
	"sort"
	"strings"

	tree_sitter "github.com/tree-sitter/go-tree-sitter"
	tree_sitter_go "github.com/tree-sitter/tree-sitter-go/bindings/go"
	tree_sitter_java "github.com/tree-sitter/tree-sitter-java/bindings/go"
	tree_sitter_javascript "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	tree_sitter_python "github.com/tree-sitter/tree-sitter-python/bindings/go"
	tree_sitter_typescript "github.com/tree-sitter/tree-sitter-typescript/bindings/go"
)

type fencedCodeParts struct {
	language  string
	body      string
	bodyStart int
	bodyEnd   int
}

// splitFencedCodeUnitWithTreeSitter splits an oversized fenced code block at
// top-level AST node boundaries. It preserves byte-for-byte reconstruction of
// the original fenced block; if parsing is unavailable or unhelpful callers
// should fall back to the legacy forced splitter.
func splitFencedCodeUnitWithTreeSitter(text string, runeStart, maxSize int) []splitUnit {
	if maxSize <= 0 || runeLen(text) <= maxSize {
		return nil
	}
	parts, ok := parseFencedCodeParts(text)
	if !ok || strings.TrimSpace(parts.body) == "" {
		return nil
	}
	language, ok := treeSitterLanguage(parts.language)
	if !ok {
		return nil
	}

	parser := tree_sitter.NewParser()
	defer parser.Close()
	if err := parser.SetLanguage(language); err != nil {
		return nil
	}
	tree := parser.Parse([]byte(parts.body), nil)
	if tree == nil {
		return nil
	}
	defer tree.Close()

	root := tree.RootNode()
	if root == nil || root.HasError() || root.NamedChildCount() == 0 {
		return nil
	}

	cuts := astBoundaryCuts(root, parts.bodyStart, parts.bodyEnd, len(text))
	if len(cuts) == 0 {
		return nil
	}
	return splitByByteCuts(text, runeStart, maxSize, cuts)
}

func parseFencedCodeParts(text string) (fencedCodeParts, bool) {
	match := codeBlockPattern.FindStringSubmatchIndex(text)
	if len(match) < 6 || match[0] != 0 || match[1] != len(text) {
		return fencedCodeParts{}, false
	}
	language := ""
	if match[2] >= 0 && match[3] >= 0 {
		language = strings.ToLower(strings.TrimSpace(text[match[2]:match[3]]))
	}
	return fencedCodeParts{
		language:  language,
		body:      text[match[4]:match[5]],
		bodyStart: match[4],
		bodyEnd:   match[5],
	}, true
}

func treeSitterLanguage(language string) (*tree_sitter.Language, bool) {
	language = strings.ToLower(strings.TrimSpace(language))
	language = strings.TrimPrefix(language, ".")
	switch language {
	case "go", "golang":
		return tree_sitter.NewLanguage(tree_sitter_go.Language()), true
	case "py", "python", "python3":
		return tree_sitter.NewLanguage(tree_sitter_python.Language()), true
	case "js", "jsx", "javascript", "mjs", "cjs":
		return tree_sitter.NewLanguage(tree_sitter_javascript.Language()), true
	case "ts", "typescript":
		return tree_sitter.NewLanguage(tree_sitter_typescript.LanguageTypescript()), true
	case "tsx":
		return tree_sitter.NewLanguage(tree_sitter_typescript.LanguageTSX()), true
	case "java":
		return tree_sitter.NewLanguage(tree_sitter_java.Language()), true
	default:
		return nil, false
	}
}

func astBoundaryCuts(root *tree_sitter.Node, bodyStart, bodyEnd, textLen int) []int {
	cuts := make([]int, 0, root.NamedChildCount()+2)
	for i := uint(0); i < root.NamedChildCount(); i++ {
		child := root.NamedChild(i)
		if child == nil || child.EndByte() == 0 {
			continue
		}
		cut := bodyStart + int(child.EndByte())
		if cut > bodyStart && cut < bodyEnd {
			cuts = append(cuts, cut)
		}
	}
	cuts = append(cuts, bodyEnd)
	cuts = append(cuts, textLen)
	sort.Ints(cuts)
	out := cuts[:0]
	last := -1
	for _, cut := range cuts {
		if cut > 0 && cut <= textLen && cut != last {
			out = append(out, cut)
			last = cut
		}
	}
	return out
}

func splitByByteCuts(text string, runeStart, maxSize int, cuts []int) []splitUnit {
	var units []splitUnit
	segmentStart := 0
	lastFit := 0

	for _, cut := range cuts {
		if cut <= segmentStart || cut > len(text) {
			continue
		}
		if runeLen(text[segmentStart:cut]) <= maxSize {
			lastFit = cut
			continue
		}
		if lastFit > segmentStart {
			units = append(units, splitUnitFromByteRange(text, runeStart, segmentStart, lastFit))
			segmentStart = lastFit
			lastFit = segmentStart
			if runeLen(text[segmentStart:cut]) <= maxSize {
				lastFit = cut
				continue
			}
		}
		units = append(units, forcedSplitByteRange(text, runeStart, segmentStart, cut, maxSize)...)
		segmentStart = cut
		lastFit = segmentStart
	}

	if segmentStart < len(text) {
		units = append(units, splitUnitFromByteRange(text, runeStart, segmentStart, len(text)))
	}
	if len(units) <= 1 {
		return nil
	}
	return units
}

func forcedSplitByteRange(text string, runeStart, startByte, endByte, maxSize int) []splitUnit {
	if startByte >= endByte {
		return nil
	}
	segment := text[startByte:endByte]
	runes := []rune(segment)
	startRuneOffset := runeLen(text[:startByte])
	var units []splitUnit
	offset := 0
	for offset < len(runes) {
		chunkEnd := forcedSplitEnd(runes, offset, maxSize)
		chunkText := string(runes[offset:chunkEnd])
		units = append(units, splitUnit{
			text:  chunkText,
			start: runeStart + startRuneOffset + offset,
			end:   runeStart + startRuneOffset + chunkEnd,
		})
		offset = chunkEnd
	}
	return units
}

func splitUnitFromByteRange(text string, runeStart, startByte, endByte int) splitUnit {
	startRune := runeLen(text[:startByte])
	content := text[startByte:endByte]
	return splitUnit{
		text:  content,
		start: runeStart + startRune,
		end:   runeStart + startRune + runeLen(content),
	}
}
