// Chunker debug / preview API. Single endpoint that runs the adaptive
// chunker over a sample text without touching DB. Semantic previews call
// the selected embedding model to compute breakpoints. Used by the KB editor
// debug panel.

import { post } from '../../utils/request'
import type {
  PreviewChunkingRequest,
  PreviewChunkingResponse
} from '../../types/chunker'

export function previewChunking(
  body: PreviewChunkingRequest
): Promise<{ success: boolean; data: PreviewChunkingResponse }> {
  return post<{
    success: boolean
    data: PreviewChunkingResponse
  }>('/api/v1/chunker/preview', body, { timeout: 60000 })
}
