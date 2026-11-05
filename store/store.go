package store

import (
	"context"
	"time"

	"github.com/google/uuid"
)

// Chunk represents a piece of code with its vector embedding
type Chunk struct {
	ID          string    `json:"id"`
	FilePath    string    `json:"file_path"`
	StartLine   int       `json:"start_line"`
	EndLine     int       `json:"end_line"`
	Content     string    `json:"content"`
	Vector      []float32 `json:"vector"`
	Hash        string    `json:"hash"`
	ContentHash string    `json:"content_hash"` // SHA256 of raw content (path-independent)
	UpdatedAt   time.Time `json:"updated_at"`
}

// Document represents a file with its chunks
type Document struct {
	Path     string    `json:"path"`
	Hash     string    `json:"hash"`
	ModTime  time.Time `json:"mod_time"`
	ChunkIDs []string  `json:"chunk_ids"`
}

// SearchResult represents a search match with its relevance score
type SearchResult struct {
	Chunk Chunk   `json:"chunk"`
	Score float32 `json:"score"`
}

// SearchOptions contains optional filters for vector search queries.
type SearchOptions struct {
	PathPrefix string
}

// IndexStats contains statistics about the index
type IndexStats struct {
	TotalFiles  int       `json:"total_files"`
	TotalChunks int       `json:"total_chunks"`
	IndexSize   int64     `json:"index_size"` // bytes
	LastUpdated time.Time `json:"last_updated"`
}

// FileStats contains statistics for a single file
type FileStats struct {
	Path       string    `json:"path"`
	ChunkCount int       `json:"chunk_count"`
	ModTime    time.Time `json:"mod_time"`
}

// VectorStore defines the interface for vector storage backends
type VectorStore interface {
	// SaveChunks stores multiple chunks atomically
	SaveChunks(ctx context.Context, chunks []Chunk) error

	// DeleteByFile removes all chunks for a given file path
	DeleteByFile(ctx context.Context, filePath string) error

	// DeleteChunksByIDs removes the chunks with the given IDs, leaving all
	// other points untouched. Used to delete only the stale positions of a
	// re-indexed file after the new points are upserted.
	DeleteChunksByIDs(ctx context.Context, chunkIDs []string) error

	// Search finds the most similar chunks to a query vector
	Search(ctx context.Context, queryVector []float32, limit int, opts SearchOptions) ([]SearchResult, error)

	// GetDocument retrieves document metadata by path
	GetDocument(ctx context.Context, filePath string) (*Document, error)

	// SaveDocument stores document metadata
	SaveDocument(ctx context.Context, doc Document) error

	// DeleteDocument removes document metadata
	DeleteDocument(ctx context.Context, filePath string) error

	// ListDocuments returns all indexed document paths
	ListDocuments(ctx context.Context) ([]string, error)

	// Load reads the store from persistent storage
	Load(ctx context.Context) error

	// Persist writes the store to persistent storage
	Persist(ctx context.Context) error

	// Close cleanly shuts down the store
	Close() error

	// GetStats returns index statistics
	GetStats(ctx context.Context) (*IndexStats, error)

	// ListFilesWithStats returns all files with their chunk counts
	ListFilesWithStats(ctx context.Context) ([]FileStats, error)

	// GetChunksForFile returns all chunks for a specific file
	GetChunksForFile(ctx context.Context, filePath string) ([]Chunk, error)

	// GetAllChunks returns all chunks in the store (used for text search)
	GetAllChunks(ctx context.Context) ([]Chunk, error)
}

// EmbeddingCache is an optional interface that VectorStore implementations can
// provide to enable content-addressed embedding deduplication. When a store
// implements this interface, the indexer will look up existing embeddings by
// content hash before calling the embedder, avoiding redundant API calls for
// identical content (e.g., across git worktrees).
//
// Reuse is scoped to the same file path as well as the hash: a stored
// vector encodes its file's identity ("File: <path>" header), so a renamed
// file correctly re-embeds instead of inheriting another path's vector.
type EmbeddingCache interface {
	// LookupByContentHash returns the embedding vector for the given content
	// hash stored under the given file path.
	// Returns (vector, true, nil) if found, (nil, false, nil) if not found.
	LookupByContentHash(ctx context.Context, contentHash, filePath string) ([]float32, bool, error)
}

// PointRef identifies one stored point for reconcile: its collection point
// ID and the file it belongs to. No vectors, no content.
type PointRef struct {
	PointID  string
	FilePath string
}

// BulkReconcileStore is implemented by stores that can enumerate every point
// ID cheaply (IDs and file paths only, no vectors or content) and delete
// points by ID. Startup reconcile uses it to scan the collection once
// instead of once per file.
type BulkReconcileStore interface {
	ListPointRefs(ctx context.Context) ([]PointRef, error)
	DeletePointIDs(ctx context.Context, pointIDs []string) error
}

// qdrantPointNamespace is the UUIDv5 namespace mapping chunk IDs to
// collection point IDs. Point IDs are deterministic in both versions, so
// reconcile compares point IDs — never the chunk_id payload — and legacy
// points without chunk_id still match.
var qdrantPointNamespace = uuid.MustParse("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

// PointIDForChunk maps a chunk ID to its deterministic collection point ID.
func PointIDForChunk(chunkID string) string {
	return uuid.NewSHA1(qdrantPointNamespace, []byte(chunkID)).String()
}
