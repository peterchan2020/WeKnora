// Package handler — chunker_debug.go exposes a read-only preview endpoint
// that runs the adaptive chunker on supplied text without touching the DB.
// Semantic previews resolve the selected embedding model because semantic
// chunking needs embeddings to choose breakpoints. Used by the KB editor's
// debug panel so users can experiment with chunking parameters before
// committing to a re-index.
package handler

import (
	"context"
	"errors"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	modelsvc "github.com/Tencent/WeKnora/internal/application/service"
	"github.com/Tencent/WeKnora/internal/infrastructure/chunker"
	"github.com/Tencent/WeKnora/internal/logger"
	"github.com/Tencent/WeKnora/internal/types/interfaces"
	"github.com/gin-gonic/gin"
)

// previewMaxChars caps the input text size so a single preview request
// cannot tie up the splitter for long. Chosen to bound worst-case CPU
// well under the previewTimeout: at 64k runes even the heaviest tier
// chain finishes in well under a second on commodity hardware.
//
// SECURITY NOTE: the splitter is CPU-bound and does NOT accept a
// context.Context. When previewTimeout fires the handler returns to the
// caller, but the worker goroutine keeps running until the splitter
// finishes naturally. The 64k ceiling is the primary mitigation against
// goroutine pile-up under repeated authenticated requests. If the
// splitter ever gains context-awareness, the goroutine-wrapper in
// PreviewChunking should switch to it for true cancellation.
const previewMaxChars = 64 * 1024

// previewMaxChunks caps the number of chunks returned in a single preview
// response so the UI doesn't choke on pathological splits. Stats are
// computed over the full chunk set before truncation so the displayed
// avg/min/max/stddev stay representative.
const previewMaxChunks = 500

// previewTimeout caps how long the handler waits for the splitter
// goroutine before returning a 504. See note above on previewMaxChars.
const previewTimeout = 5 * time.Second

// semanticPreviewTimeout is longer because semantic preview must call the
// configured embedding model. Keep it below the frontend request timeout with
// enough margin for response serialization.
const semanticPreviewTimeout = 25 * time.Second

// PreviewChunkingRequest is the body shape accepted by /chunker/preview.
// Text is checked manually below so we can return a friendlier error than
// gin's default "Field validation for 'Text' failed on the 'required' tag".
type PreviewChunkingRequest struct {
	Text             string                 `json:"text"`
	EmbeddingModelID string                 `json:"embedding_model_id"`
	ChunkingConfig   PreviewChunkingPayload `json:"chunking_config"`
}

// PreviewChunkingPayload mirrors the snake_case JSON the rest of the API
// uses for ChunkingConfig fields. We don't reuse types.ChunkingConfig
// directly because it carries a lot of unrelated fields (parser engine
// rules, parent-child sizes, etc.) that the preview path doesn't need.
type PreviewChunkingPayload struct {
	ChunkSize                    int      `json:"chunk_size"`
	ChunkOverlap                 int      `json:"chunk_overlap"`
	Separators                   []string `json:"separators"`
	Strategy                     string   `json:"strategy"`
	TokenLimit                   int      `json:"token_limit"`
	Languages                    []string `json:"languages"`
	SemanticBufferSize           int      `json:"semantic_buffer_size"`
	SemanticBreakpointPercentile int      `json:"semantic_breakpoint_percentile"`
}

// PreviewChunkResult describes one chunk emitted during preview.
type PreviewChunkResult struct {
	Seq              int                    `json:"seq"`
	Start            int                    `json:"start"`
	End              int                    `json:"end"`
	SizeChars        int                    `json:"size_chars"`
	SizeTokensApprox int                    `json:"size_tokens_approx"`
	ContextHeader    string                 `json:"context_header,omitempty"`
	Content          string                 `json:"content"`
	Structure        *PreviewChunkStructure `json:"structure,omitempty"`
}

type PreviewChunkStructure struct {
	TableCount   int                    `json:"table_count"`
	FormulaCount int                    `json:"formula_count"`
	CodeCount    int                    `json:"code_count"`
	ImageCount   int                    `json:"image_count"`
	Tables       []PreviewTableDetail   `json:"tables,omitempty"`
	Formulas     []PreviewFormulaDetail `json:"formulas,omitempty"`
	CodeBlocks   []PreviewCodeDetail    `json:"code_blocks,omitempty"`
	Images       []PreviewImageDetail   `json:"images,omitempty"`
}

type PreviewTableDetail struct {
	Kind    string   `json:"kind"`
	Columns []string `json:"columns,omitempty"`
	Rows    int      `json:"rows"`
	Summary string   `json:"summary,omitempty"`
	Start   int      `json:"start"`
	End     int      `json:"end"`
}

type PreviewFormulaDetail struct {
	Kind  string `json:"kind"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

type PreviewCodeDetail struct {
	Language string `json:"language,omitempty"`
	Start    int    `json:"start"`
	End      int    `json:"end"`
}

type PreviewImageDetail struct {
	URL         string `json:"url"`
	AltText     string `json:"alt_text,omitempty"`
	MarkdownRef string `json:"markdown_ref,omitempty"`
	Start       int    `json:"start"`
	End         int    `json:"end"`
}

// PreviewChunkingStats summarizes chunk-size distribution. Computed over
// the FULL chunk set, even when the response truncates to previewMaxChunks
// items, so avg/min/max/stddev reflect the real distribution, not just
// the first N.
type PreviewChunkingStats struct {
	Count       int `json:"count"`
	AvgChars    int `json:"avg_chars"`
	MinChars    int `json:"min_chars"`
	MaxChars    int `json:"max_chars"`
	StddevChars int `json:"stddev_chars"`
	// TruncatedTo, when set, is the original chunk count before the
	// response was truncated to previewMaxChunks for transport.
	TruncatedTo int `json:"truncated_to,omitempty"`
}

// PreviewChunkingResponse is the body returned by /chunker/preview.
type PreviewChunkingResponse struct {
	SelectedTier chunker.StrategyTier    `json:"selected_tier"`
	TierChain    []chunker.StrategyTier  `json:"tier_chain"`
	Rejected     []chunker.TierRejection `json:"rejected"`
	Profile      *chunker.DocProfile     `json:"profile"`
	Chunks       []PreviewChunkResult    `json:"chunks"`
	Stats        PreviewChunkingStats    `json:"stats"`
}

// PreviewChunkingWithModelService handles POST /chunker/preview. It runs the supplied text
// through the adaptive chunker and returns the chunks plus diagnostic
// information about which tier won. Read-only: no DB writes and no logging
// of the supplied text.
//
// PreviewChunking godoc
// @Summary      预览分块结果
// @Description  对提交的文本运行自适应分块器并返回分块预览，不写入数据库不生成 embedding。文本最大 64k 字符
// @Tags         分块
// @Accept       json
// @Produce      json
// @Param        request  body      handler.PreviewChunkingRequest   true  "{text, chunking_config}"
// @Success      200      {object}  handler.PreviewChunkingResponse  "分块结果"
// @Failure      400      {object}  map[string]interface{}           "请求参数错误"
// @Failure      413      {object}  map[string]interface{}           "文本超过预览限制"
// @Failure      504      {object}  map[string]interface{}           "分块超时"
// @Security     Bearer
// @Security     ApiKeyAuth
// @Router       /chunker/preview [post]
// PreviewChunkingWithModelService returns a preview handler that can resolve
// the current embedding model for semantic chunking previews.
func PreviewChunkingWithModelService(modelService interfaces.ModelService) gin.HandlerFunc {
	return func(c *gin.Context) {
		previewChunking(c, modelService)
	}
}

func previewChunking(c *gin.Context, modelService interfaces.ModelService) {
	var req PreviewChunkingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"success": false, "error": "invalid request body: " + err.Error()})
		return
	}

	if strings.TrimSpace(req.Text) == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "text is empty — paste a sample to preview chunking",
		})
		return
	}

	if utf8.RuneCountInString(req.Text) > previewMaxChars {
		c.JSON(http.StatusRequestEntityTooLarge, gin.H{
			"success": false,
			"error":   "text exceeds preview limit",
			"limit":   previewMaxChars,
		})
		return
	}

	cfg := chunker.SplitterConfig{
		ChunkSize:                    req.ChunkingConfig.ChunkSize,
		ChunkOverlap:                 req.ChunkingConfig.ChunkOverlap,
		Separators:                   req.ChunkingConfig.Separators,
		Strategy:                     req.ChunkingConfig.Strategy,
		TokenLimit:                   req.ChunkingConfig.TokenLimit,
		Languages:                    req.ChunkingConfig.Languages,
		SemanticBufferSize:           req.ChunkingConfig.SemanticBufferSize,
		SemanticBreakpointPercentile: req.ChunkingConfig.SemanticBreakpointPercentile,
	}

	if strings.EqualFold(strings.TrimSpace(cfg.Strategy), chunker.StrategySemantic) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), semanticPreviewTimeout)
		defer cancel()
		handleSemanticPreview(c, ctx, req, cfg, modelService)
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), previewTimeout)
	defer cancel()

	// Run the splitter on a goroutine so we can honor the request timeout.
	// The splitter is CPU-bound and doesn't accept a context — wrapping
	// here is the cheapest cancellation we can offer.
	type splitResult struct {
		chunks []chunker.Chunk
		diag   *chunker.Diagnostics
	}
	resCh := make(chan splitResult, 1)
	go func() {
		chunks, diag := chunker.SplitWithDiagnostics(req.Text, cfg)
		resCh <- splitResult{chunks: chunks, diag: diag}
	}()

	var sr splitResult
	select {
	case sr = <-resCh:
	case <-ctx.Done():
		c.JSON(http.StatusGatewayTimeout, gin.H{
			"success": false,
			"error":   "chunker preview timed out",
		})
		return
	}

	chunks, diag := sr.chunks, sr.diag

	// Diagnostics carries the profile when auto-strategy ran; for explicit
	// strategies the profile is nil and we materialize it here so the UI
	// can still show document stats. Avoids the previous double-pass.
	profile := diag.Profile
	if profile == nil {
		profile = chunker.ProfileDocument(req.Text)
	}

	logger.Debugf(ctx, "chunker preview: tier=%s chunks=%d", diag.SelectedTier, len(chunks))

	resp := buildPreviewResponse(chunks, diag, profile)
	c.JSON(http.StatusOK, gin.H{"success": true, "data": resp})
}

func handleSemanticPreview(
	c *gin.Context,
	ctx context.Context,
	req PreviewChunkingRequest,
	cfg chunker.SplitterConfig,
	modelService interfaces.ModelService,
) {
	modelID := strings.TrimSpace(req.EmbeddingModelID)
	if modelID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "semantic chunking preview requires an embedding model",
		})
		return
	}
	if modelService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "semantic chunking preview is not wired to a model service",
		})
		return
	}

	embedder, err := modelService.GetEmbeddingModel(ctx, modelID)
	if err != nil {
		status := http.StatusBadGateway
		if errors.Is(err, modelsvc.ErrModelNotFound) {
			status = http.StatusBadRequest
		}
		c.JSON(status, gin.H{
			"success": false,
			"error":   "failed to load embedding model for semantic preview: " + err.Error(),
		})
		return
	}

	type splitResult struct {
		chunks []chunker.Chunk
		err    error
	}
	resCh := make(chan splitResult, 1)
	go func() {
		chunks, err := chunker.SplitSemantic(ctx, req.Text, cfg, embedder)
		resCh <- splitResult{chunks: chunks, err: err}
	}()

	var sr splitResult
	select {
	case sr = <-resCh:
	case <-ctx.Done():
		c.JSON(http.StatusGatewayTimeout, gin.H{
			"success": false,
			"error":   "semantic chunker preview timed out",
		})
		return
	}
	if sr.err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "semantic chunking preview failed: " + sr.err.Error(),
		})
		return
	}

	profile := chunker.ProfileDocument(req.Text)
	resp := buildPreviewResponse(sr.chunks, &chunker.Diagnostics{
		SelectedTier: chunker.TierSemantic,
		TierChain:    []chunker.StrategyTier{chunker.TierSemantic},
		Profile:      profile,
	}, profile)
	c.JSON(http.StatusOK, gin.H{"success": true, "data": resp})
}

func buildPreviewResponse(
	chunks []chunker.Chunk,
	diag *chunker.Diagnostics,
	profile *chunker.DocProfile,
) PreviewChunkingResponse {
	lang := chunker.LangMixed
	if profile != nil && len(profile.DetectedLangs) > 0 {
		lang = profile.DetectedLangs[0]
	}

	// Compute rune lengths once per chunk; reused for stats and result
	// payload below. Avoids the previous triple-pass over each chunk's
	// content (stats + result + ApproxTokenCount each rune-counted).
	runeLens := make([]int, len(chunks))
	for i, ch := range chunks {
		runeLens[i] = utf8.RuneCountInString(ch.Content)
	}

	// Compute stats over the FULL chunk set first so the metrics stay
	// representative even when we trim the response to previewMaxChunks.
	totalCount := len(chunks)
	stats := computeChunkSizeStats(runeLens)
	if totalCount > previewMaxChunks {
		stats.TruncatedTo = totalCount
		chunks = chunks[:previewMaxChunks]
		runeLens = runeLens[:previewMaxChunks]
	}

	results := make([]PreviewChunkResult, 0, len(chunks))
	for i, ch := range chunks {
		results = append(results, PreviewChunkResult{
			Seq:              ch.Seq,
			Start:            ch.Start,
			End:              ch.End,
			SizeChars:        runeLens[i],
			SizeTokensApprox: chunker.ApproxTokenCountFromRuneLen(runeLens[i], lang),
			ContextHeader:    ch.ContextHeader,
			Content:          ch.Content,
			Structure:        buildPreviewChunkStructure(ch),
		})
	}

	if diag == nil {
		diag = &chunker.Diagnostics{SelectedTier: chunker.TierLegacy}
	}
	return PreviewChunkingResponse{
		SelectedTier: diag.SelectedTier,
		TierChain:    diag.TierChain,
		Rejected:     diag.Rejected,
		Profile:      profile,
		Chunks:       results,
		Stats:        stats,
	}
}

// computeChunkSizeStats summarizes count / avg / min / max / stddev from
// a pre-computed rune-length slice. Decoupling from chunker.Chunk lets
// the caller compute rune lengths once and reuse them for the response
// payload (avoids a second []rune allocation per chunk).
func computeChunkSizeStats(runeLens []int) PreviewChunkingStats {
	stats := PreviewChunkingStats{Count: len(runeLens)}
	if len(runeLens) == 0 {
		return stats
	}
	var sum, sumSq float64
	minLen, maxLen := math.MaxInt32, 0
	for _, l := range runeLens {
		sum += float64(l)
		sumSq += float64(l) * float64(l)
		if l < minLen {
			minLen = l
		}
		if l > maxLen {
			maxLen = l
		}
	}
	avg := sum / float64(len(runeLens))
	variance := sumSq/float64(len(runeLens)) - avg*avg
	if variance < 0 {
		// Float precision can push the variance slightly below zero on
		// near-uniform inputs; clamp so sqrt doesn't return NaN.
		variance = 0
	}
	stats.AvgChars = int(avg + 0.5)
	stats.MinChars = minLen
	stats.MaxChars = maxLen
	stats.StddevChars = int(math.Sqrt(variance) + 0.5)
	return stats
}

func buildPreviewChunkStructure(ch chunker.Chunk) *PreviewChunkStructure {
	structure := PreviewChunkStructure{
		TableCount:   len(ch.Tables),
		FormulaCount: len(ch.Formulas),
		CodeCount:    len(ch.CodeBlocks),
		ImageCount:   len(ch.Images),
	}
	if structure.TableCount == 0 &&
		structure.FormulaCount == 0 &&
		structure.CodeCount == 0 &&
		structure.ImageCount == 0 {
		return nil
	}
	if len(ch.Tables) > 0 {
		structure.Tables = make([]PreviewTableDetail, 0, len(ch.Tables))
		for idx, tbl := range ch.Tables {
			structure.Tables = append(structure.Tables, PreviewTableDetail{
				Kind:    tbl.Kind,
				Columns: tbl.Columns,
				Rows:    tbl.Rows,
				Summary: previewTableSummary(ch, idx),
				Start:   tbl.Start,
				End:     tbl.End,
			})
		}
	}
	if len(ch.Formulas) > 0 {
		structure.Formulas = make([]PreviewFormulaDetail, 0, len(ch.Formulas))
		for _, formula := range ch.Formulas {
			structure.Formulas = append(structure.Formulas, PreviewFormulaDetail{
				Kind:  formula.Kind,
				Start: formula.Start,
				End:   formula.End,
			})
		}
	}
	if len(ch.CodeBlocks) > 0 {
		structure.CodeBlocks = make([]PreviewCodeDetail, 0, len(ch.CodeBlocks))
		for _, code := range ch.CodeBlocks {
			structure.CodeBlocks = append(structure.CodeBlocks, PreviewCodeDetail{
				Language: code.Language,
				Start:    code.Start,
				End:      code.End,
			})
		}
	}
	if len(ch.Images) > 0 {
		structure.Images = make([]PreviewImageDetail, 0, len(ch.Images))
		for _, img := range ch.Images {
			structure.Images = append(structure.Images, PreviewImageDetail{
				URL:         img.OriginalRef,
				AltText:     img.AltText,
				MarkdownRef: previewContentSlice(ch.Content, img.Start, img.End),
				Start:       img.Start,
				End:         img.End,
			})
		}
	}
	return &structure
}

func previewContentSlice(content string, start, end int) string {
	// ImageRef offsets are byte offsets relative to the chunk content; see
	// chunker.ExtractImageRefs.
	if start < 0 || end < start || end > len(content) {
		return ""
	}
	return content[start:end]
}

func previewTableSummary(ch chunker.Chunk, idx int) string {
	if idx != 0 || len(ch.Metadata) == 0 {
		return ""
	}
	summary, _ := ch.Metadata["table_summary"].(string)
	return summary
}
