package chunker

import (
	_ "embed"
	"encoding/json"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/neurosnap/sentences"
	"github.com/neurosnap/sentences/english"
)

//go:embed data/chinese_punkt.json
var chinesePunktJSON []byte

// chineseRawData mirrors the JSON layout produced by the punkt training
// pipeline in the peterchan2020/sentences fork. Kept private to the chunker
// package since it's only used at init.
type chineseRawData struct {
	SentStarters []string       `json:"sent_starters"`
	Collocations [][]string     `json:"collocations"`
	AbbrevTypes  []string       `json:"abbrev_types"`
	OrthoContext map[string]int `json:"ortho_context"`
}

var (
	punktOnce        sync.Once
	punktEnglishTok  *sentences.DefaultSentenceTokenizer
	punktChineseTok  *sentences.DefaultSentenceTokenizer
	punktEnglishErr  error
	punktChineseErr  error
)

// supervisedAbbrevs are abbreviations from the WeKnora academic corpus that
// punkt's unsupervised training may miss. Adding them to the Storage ensures
// the tokenizer never splits on "Fig.", "Eq.", "et al.", etc.
var supervisedAbbrevs = []string{
	"fig", "figs", "eq", "eqs", "vs", "etc", "al",
	"dr", "prof", "mr", "mrs", "ms",
	"no", "nos", "vol", "pp", "ref", "refs", "sec", "sect",
	"approx", "max", "min", "avg",
	"dept", "univ", "inc", "ltd",
	"ed", "eds", "rev", "est",
	"jan", "feb", "mar", "apr", "jun", "jul", "aug",
	"sep", "sept", "oct", "nov", "dec",
	"st", "nd", "rd", "th",
}

func initPunktTokenizers() {
	punktEnglishTok, punktEnglishErr = english.NewSentenceTokenizer(nil)
	if punktEnglishTok != nil {
		for _, abbr := range supervisedAbbrevs {
			punktEnglishTok.Storage.AbbrevTypes.Add(abbr)
		}
	}

	storage, err := loadChinesePunktStorage()
	if err != nil {
		punktChineseErr = err
		return
	}
	for _, abbr := range supervisedAbbrevs {
		storage.AbbrevTypes.Add(abbr)
	}
	punktChineseTok, punktChineseErr = english.NewSentenceTokenizer(storage)
}

func loadChinesePunktStorage() (*sentences.Storage, error) {
	var raw chineseRawData
	if err := json.Unmarshal(chinesePunktJSON, &raw); err != nil {
		return nil, err
	}
	s := &sentences.Storage{
		AbbrevTypes:  make(sentences.SetString),
		SentStarters: make(sentences.SetString),
		Collocations: make(sentences.SetString),
		OrthoContext: make(sentences.SetString),
	}
	for _, a := range raw.AbbrevTypes {
		s.AbbrevTypes.Add(a)
	}
	for _, st := range raw.SentStarters {
		s.SentStarters.Add(st)
	}
	for _, pair := range raw.Collocations {
		if len(pair) == 2 {
			s.Collocations.Add(pair[0] + "," + pair[1])
		}
	}
	for k, v := range raw.OrthoContext {
		s.OrthoContext[k] = v
	}
	return s, nil
}

// punktTokenizerFor returns the trained tokenizer that best matches the script
// of text. Falls back to the English tokenizer when initialization of the
// Chinese tokenizer failed.
func punktTokenizerFor(text string) *sentences.DefaultSentenceTokenizer {
	punktOnce.Do(initPunktTokenizers)
	if isMostlyCJK(text) && punktChineseTok != nil {
		return punktChineseTok
	}
	if punktEnglishTok != nil {
		return punktEnglishTok
	}
	return punktChineseTok
}

// isMostlyCJK returns true when at least 30% of the runes in text are in the
// CJK Unified Ideographs ranges. The 30% threshold lets mixed-script text
// (e.g. Chinese paragraphs with embedded English citations) still route to
// the Chinese tokenizer, which understands CJK sentence punctuation.
func isMostlyCJK(text string) bool {
	if text == "" {
		return false
	}
	total := 0
	cjk := 0
	for _, r := range text {
		if unicode.IsSpace(r) {
			continue
		}
		total++
		if isCJKRune(r) {
			cjk++
		}
		if total >= 256 {
			break
		}
	}
	if total == 0 {
		return false
	}
	return cjk*10 >= total*3
}

func isCJKRune(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF: // CJK Unified Ideographs
		return true
	case r >= 0x3400 && r <= 0x4DBF: // CJK Extension A
		return true
	case r >= 0x3000 && r <= 0x303F: // CJK Symbols and Punctuation
		return true
	case r >= 0xFF00 && r <= 0xFFEF: // Halfwidth and Fullwidth Forms
		return true
	}
	return false
}

// splitBySentencesPunkt uses the punkt algorithm to split text into
// sentence-level units. Returns nil if the tokenizer fails to initialize,
// produces no useful splits, or the text is too short to be worth splitting
// (caller should fall back to the legacy separator-based path).
//
// Position fields on the returned splitUnits are rune offsets relative to the
// start of text, matching the convention used by the rest of the chunker.
func splitBySentencesPunkt(text string) []splitUnit {
	if text == "" {
		return nil
	}
	tok := punktTokenizerFor(text)
	if tok == nil {
		return nil
	}

	sents := tok.Tokenize(text)
	if len(sents) <= 1 {
		return nil
	}

	units := make([]splitUnit, 0, len(sents))
	for _, s := range sents {
		startRune := utf8.RuneCountInString(text[:s.Start])
		endRune := startRune + utf8.RuneCountInString(s.Text)
		units = append(units, splitUnit{
			text:       s.Text,
			start:      startRune,
			end:        endRune,
			isSentence: true,
		})
	}
	return units
}
