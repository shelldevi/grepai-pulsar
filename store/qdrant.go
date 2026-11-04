package store

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
)

// sanitizeUTF8 ensures the string contains only valid UTF-8 characters.
// Invalid sequences are replaced with the Unicode replacement character.
func sanitizeUTF8(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	return strings.ToValidUTF8(s, "\uFFFD")
}

type QdrantStore struct {
	client         *qdrant.Client
	collectionName string
	dimensions     int
	apiKey         string
	// docs holds file-level document metadata (file hash, chunk IDs). Qdrant
	// point payloads carry chunk data only, so without this sidecar the
	// indexer cannot tell an unchanged file from a changed one. It is nil
	// unless the store was constructed with a document path — read-only
	// consumers (search, status, MCP) leave it unset.
	docs *qdrantDocStore
	// cacheDisabled forces cache lookups to miss for the lifetime of this
	// store. Set when the embedding fingerprint mismatches, so a config
	// change never mixes vectors from another model or chunking.
	cacheDisabled bool
}

func parseHost(endpoint string) string {
	host := strings.TrimPrefix(endpoint, "http://")
	host = strings.TrimPrefix(host, "https://")

	if idx := strings.Index(host, ":"); idx != -1 {
		host = host[:idx]
	}

	if idx := strings.Index(host, "/"); idx != -1 {
		host = host[:idx]
	}

	return host
}

func NewQdrantStore(ctx context.Context, endpoint string, port int, useTLS bool, collection, apiKey string, dimensions int, docPath ...string) (*QdrantStore, error) {
	return newQdrantStore(ctx, endpoint, port, useTLS, collection, apiKey, dimensions, true, docPath)
}

// NewQdrantStoreReadOnly connects without creating the collection or the
// content_hash field index, for read-only flows (e.g. scan --dry-run) that
// must not write anything. It fails if the collection does not exist.
func NewQdrantStoreReadOnly(ctx context.Context, endpoint string, port int, useTLS bool, collection, apiKey string, dimensions int, docPath ...string) (*QdrantStore, error) {
	return newQdrantStore(ctx, endpoint, port, useTLS, collection, apiKey, dimensions, false, docPath)
}

func newQdrantStore(ctx context.Context, endpoint string, port int, useTLS bool, collection, apiKey string, dimensions int, ensure bool, docPath []string) (*QdrantStore, error) {
	host := parseHost(endpoint)

	if port <= 0 {
		port = 6334
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   port,
		UseTLS: useTLS,
		APIKey: apiKey,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Qdrant client: %w", err)
	}

	store := &QdrantStore{
		client:         client,
		collectionName: collection,
		dimensions:     dimensions,
		apiKey:         apiKey,
	}

	created, err := store.ensureCollection(ctx, ensure)
	if err != nil {
		return nil, err
	}

	// Optional file-level document metadata. When set, GetDocument and
	// friends serve real metadata so unchanged files are skipped; when
	// unset, the previous stub behavior is kept for read-only consumers.
	if len(docPath) > 0 && docPath[0] != "" {
		docs := newQdrantDocStore(docPath[0])
		if err := docs.load(); err != nil {
			return nil, err
		}
		store.docs = docs

		// A new or emptied collection with a non-empty sidecar means the
		// collection was wiped, renamed or recreated: the sidecar describes
		// points that no longer exist. Discard it loudly; the next scan
		// backfills everything.
		if created || store.collectionPoints(ctx) == 0 {
			if n := len(docs.list()); n > 0 {
				log.Printf("Warning: Qdrant collection %s is new or empty but the document sidecar lists %d files; discarding sidecar and re-indexing from scratch", collection, n)
				docs.reset()
			}
		}
	}

	return store, nil
}

// collectionPoints returns the collection's point count, or -1 when it
// cannot be determined (treated as unknown, never as empty).
func (s *QdrantStore) collectionPoints(ctx context.Context) int64 {
	info, err := s.client.GetCollectionInfo(ctx, s.collectionName)
	if err != nil || info == nil || info.PointsCount == nil {
		return -1
	}
	return int64(*info.PointsCount)
}

// SetFingerprint records the embedding configuration for this run and
// compares it with the sidecar's. A mismatch discards the sidecar and
// disables cache lookups for the run, loudly, so a model/dimension/
// chunking/collection change never mixes vectors.
func (s *QdrantStore) SetFingerprint(fp CacheFingerprint) {
	if s.docs == nil {
		return
	}
	if s.docs.adoptFingerprint(fp) {
		return
	}
	old := s.docs.fingerprint()
	log.Printf("WARNING: embedding configuration changed (was %+v, now %+v): discarding Qdrant document sidecar and disabling embedding-cache reuse for this run", old, fp)
	s.docs.reset()
	s.docs.adoptFingerprint(fp)
	s.cacheDisabled = true
}

// ensureCollection verifies the collection exists. With ensure=true it is
// created (with the content_hash field index) when missing; with ensure=false
// (read-only flows) a missing collection is an error and nothing is written.
// Returns whether the collection was created by this call.
func (s *QdrantStore) ensureCollection(ctx context.Context, ensure bool) (bool, error) {
	exists, err := s.client.CollectionExists(ctx, s.collectionName)
	if err != nil {
		return false, fmt.Errorf("failed to check collection existence: %w", err)
	}

	created := false
	if !exists {
		if !ensure {
			return false, fmt.Errorf("collection %s does not exist (read-only mode, refusing to create it)", s.collectionName)
		}
		if s.dimensions <= 0 {
			return false, fmt.Errorf("dimensions must be positive, got: %d", s.dimensions)
		}
		err = s.client.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: s.collectionName,
			VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
				Size:     uint64(s.dimensions),
				Distance: qdrant.Distance_Cosine,
			}),
		})
		if err != nil {
			return false, fmt.Errorf("failed to create collection: %w", err)
		}
		created = true
	}

	// Create field index for content_hash to enable efficient lookups.
	// Skipped in read-only mode; error is otherwise intentionally ignored
	// because the index may already exist.
	if ensure {
		_, _ = s.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: s.collectionName,
			FieldName:      "content_hash",
			FieldType:      qdrant.PtrOf(qdrant.FieldType_FieldTypeKeyword),
		})
	}

	return created, nil
}

func sanitizeCollectionName(path string) string {
	return strings.ReplaceAll(path, "/", "_")
}

// pointVector extracts the dense vector from a retrieved point. Current
// Qdrant servers populate the deprecated flat Data field rather than the
// VectorOutput oneof, so both forms must be handled: reading only the oneof
// (GetDense) silently misses every cached vector and disables all
// content-hash reuse.
func pointVector(point *qdrant.RetrievedPoint) []float32 {
	if point == nil || point.Vectors == nil {
		return nil
	}
	vec := point.Vectors.GetVector()
	if vec == nil {
		return nil
	}
	if dense := vec.GetDense(); dense != nil {
		return dense.GetData()
	}
	return vec.GetData()
}

func SanitizeCollectionName(path string) string {
	return sanitizeCollectionName(path)
}

func (s *QdrantStore) getUUIDForChunk(chunkID string) uuid.UUID {
	namespace := uuid.MustParse("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
	return uuid.NewSHA1(namespace, []byte(chunkID))
}

func (s *QdrantStore) SaveChunks(ctx context.Context, chunks []Chunk) error {
	if len(chunks) == 0 {
		return nil
	}

	points := make([]*qdrant.PointStruct, 0, len(chunks))
	for _, chunk := range chunks {
		payload, err := s.buildChunkPayload(chunk)
		if err != nil {
			return fmt.Errorf("failed to build payload: %w", err)
		}

		pointID := s.getUUIDForChunk(chunk.ID)

		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewID(pointID.String()),
			Vectors: qdrant.NewVectors(chunk.Vector...),
			Payload: payload,
		})
	}

	_, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collectionName,
		Points:         points,
	})
	if err != nil {
		return fmt.Errorf("failed to upsert points: %w", err)
	}

	return nil
}

func (s *QdrantStore) buildChunkPayload(chunk Chunk) (map[string]*qdrant.Value, error) {
	payload := make(map[string]*qdrant.Value)

	filePathVal, err := qdrant.NewValue(chunk.FilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create file_path value: %w", err)
	}

	startLineVal, err := qdrant.NewValue(int64(chunk.StartLine))
	if err != nil {
		return nil, fmt.Errorf("failed to create start_line value: %w", err)
	}

	endLineVal, err := qdrant.NewValue(int64(chunk.EndLine))
	if err != nil {
		return nil, fmt.Errorf("failed to create end_line value: %w", err)
	}

	contentVal, err := qdrant.NewValue(sanitizeUTF8(chunk.Content))
	if err != nil {
		return nil, fmt.Errorf("failed to create content value: %w", err)
	}

	hashVal, err := qdrant.NewValue(chunk.Hash)
	if err != nil {
		return nil, fmt.Errorf("failed to create hash value: %w", err)
	}

	updatedAtVal, err := qdrant.NewValue(chunk.UpdatedAt.Format(time.RFC3339))
	if err != nil {
		return nil, fmt.Errorf("failed to create updated_at value: %w", err)
	}

	payload["file_path"] = filePathVal
	payload["start_line"] = startLineVal
	payload["end_line"] = endLineVal
	payload["content"] = contentVal
	payload["hash"] = hashVal
	payload["updated_at"] = updatedAtVal

	// The point ID is a one-way UUID of this value, so store it explicitly:
	// reconciliation and callers need the original chunk ID back.
	chunkIDVal, err := qdrant.NewValue(chunk.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to create chunk_id value: %w", err)
	}
	payload["chunk_id"] = chunkIDVal

	if chunk.ContentHash != "" {
		contentHashVal, err := qdrant.NewValue(chunk.ContentHash)
		if err != nil {
			return nil, fmt.Errorf("failed to create content_hash value: %w", err)
		}
		payload["content_hash"] = contentHashVal
	}

	return payload, nil
}

func (s *QdrantStore) DeleteByFile(ctx context.Context, filePath string) error {
	filter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewMatch("file_path", filePath),
		},
	}

	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Points:         qdrant.NewPointsSelectorFilter(filter),
	})
	if err != nil {
		return fmt.Errorf("failed to delete points: %w", err)
	}

	return nil
}

// DeleteChunksByIDs removes the points with the given chunk IDs, leaving all
// other points untouched. Chunk IDs map to point IDs deterministically
// (UUIDv5 of the chunk ID), so callers pass store.Chunk IDs directly.
func (s *QdrantStore) DeleteChunksByIDs(ctx context.Context, chunkIDs []string) error {
	if len(chunkIDs) == 0 {
		return nil
	}

	ids := make([]*qdrant.PointId, 0, len(chunkIDs))
	for _, chunkID := range chunkIDs {
		ids = append(ids, qdrant.NewID(s.getUUIDForChunk(chunkID).String()))
	}

	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collectionName,
		Points:         qdrant.NewPointsSelectorIDs(ids),
	})
	if err != nil {
		return fmt.Errorf("failed to delete points by IDs: %w", err)
	}

	return nil
}

func (s *QdrantStore) Search(ctx context.Context, queryVector []float32, limit int, opts SearchOptions) ([]SearchResult, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("limit must be positive, got: %d", limit)
	}

	// Fetch more results to account for filtering by path prefix
	fetchLimit := limit
	if opts.PathPrefix != "" {
		// Fetch 2x the limit to allow for filtering
		maxInt := int(^uint(0) >> 1)
		if limit > maxInt/2 {
			fetchLimit = maxInt
		} else {
			fetchLimit = limit * 2
		}
	}
	if fetchLimit < 0 {
		return nil, fmt.Errorf("invalid fetch limit: %d", fetchLimit)
	}
	fetchLimitU64 := uint64(fetchLimit)

	searchResult, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.collectionName,
		Query:          qdrant.NewQuery(queryVector...),
		Limit:          qdrant.PtrOf(fetchLimitU64),
		WithPayload:    qdrant.NewWithPayloadInclude("file_path", "start_line", "end_line", "content", "hash", "updated_at", "chunk_id"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search: %w", err)
	}

	results := make([]SearchResult, 0, len(searchResult))
	for _, point := range searchResult {
		chunk := s.parseChunkPayload(point.Payload)

		// Filter by path prefix if provided
		if opts.PathPrefix != "" && !strings.HasPrefix(chunk.FilePath, opts.PathPrefix) {
			continue
		}

		results = append(results, SearchResult{
			Chunk: *chunk,
			Score: point.Score,
		})

		// Stop once we have enough results
		if len(results) >= limit {
			break
		}
	}

	return results, nil
}

func (s *QdrantStore) parseChunkPayload(payload map[string]*qdrant.Value) *Chunk {
	chunk := &Chunk{}
	if val, ok := payload["chunk_id"]; ok {
		chunk.ID = val.GetStringValue()
	}
	if val, ok := payload["file_path"]; ok {
		chunk.FilePath = val.GetStringValue()
	}
	if val, ok := payload["start_line"]; ok {
		chunk.StartLine = int(val.GetIntegerValue())
	}
	if val, ok := payload["end_line"]; ok {
		chunk.EndLine = int(val.GetIntegerValue())
	}
	if val, ok := payload["content"]; ok {
		chunk.Content = val.GetStringValue()
	}
	if val, ok := payload["hash"]; ok {
		chunk.Hash = val.GetStringValue()
	}
	if val, ok := payload["updated_at"]; ok {
		t, err := time.Parse(time.RFC3339, val.GetStringValue())
		if err == nil {
			chunk.UpdatedAt = t
		}
	}
	if val, ok := payload["content_hash"]; ok {
		chunk.ContentHash = val.GetStringValue()
	}

	return chunk
}

func (s *QdrantStore) GetDocument(ctx context.Context, filePath string) (*Document, error) {
	if s.docs != nil {
		doc, ok := s.docs.get(filePath)
		if !ok {
			return nil, nil
		}
		return &doc, nil
	}

	filter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewMatch("file_path", filePath),
		},
	}

	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Filter:         filter,
		Limit:          qdrant.PtrOf(uint32(1)),
		WithPayload:    qdrant.NewWithPayloadInclude("chunk_ids"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get document: %w", err)
	}

	if len(scrollResult) == 0 {
		return nil, nil
	}

	doc := &Document{
		Path:     filePath,
		ChunkIDs: []string{},
	}

	return doc, nil
}

func (s *QdrantStore) SaveDocument(ctx context.Context, doc Document) error {
	if s.docs != nil {
		s.docs.set(doc)
		return nil
	}
	return nil
}

func (s *QdrantStore) DeleteDocument(ctx context.Context, filePath string) error {
	if s.docs != nil {
		s.docs.del(filePath)
		return nil
	}
	return nil
}

func (s *QdrantStore) ListDocuments(ctx context.Context) ([]string, error) {
	if s.docs != nil {
		return s.docs.list(), nil
	}

	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Limit:          qdrant.PtrOf(uint32(1000)),
		WithPayload:    qdrant.NewWithPayloadInclude("file_path"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list documents: %w", err)
	}

	pathsMap := make(map[string]bool)
	for _, point := range scrollResult {
		if val, ok := point.Payload["file_path"]; ok {
			pathsMap[val.GetStringValue()] = true
		}
	}

	paths := make([]string, 0, len(pathsMap))
	for path := range pathsMap {
		paths = append(paths, path)
	}

	return paths, nil
}

func (s *QdrantStore) Load(ctx context.Context) error {
	if s.docs != nil {
		return s.docs.load()
	}
	return nil
}

func (s *QdrantStore) Persist(ctx context.Context) error {
	if s.docs != nil {
		return s.docs.persist()
	}
	return nil
}

func (s *QdrantStore) Close() error {
	return nil
}

func (s *QdrantStore) GetStats(ctx context.Context) (*IndexStats, error) {
	collectionInfo, err := s.client.GetCollectionInfo(ctx, s.collectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection info: %w", err)
	}

	pointsCount := *collectionInfo.PointsCount
	if pointsCount > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("points count %d exceeds maximum int value", pointsCount)
	}

	stats := &IndexStats{
		TotalFiles:  0,
		TotalChunks: int(pointsCount),
		IndexSize:   0,
		LastUpdated: time.Now(),
	}

	return stats, nil
}

func (s *QdrantStore) ListFilesWithStats(ctx context.Context) ([]FileStats, error) {
	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Limit:          qdrant.PtrOf(uint32(10000)),
		WithPayload:    qdrant.NewWithPayloadInclude("file_path", "start_line", "end_line"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list files: %w", err)
	}

	fileStats := make(map[string]*FileStats)
	for _, point := range scrollResult {
		filePath := ""
		if val, ok := point.Payload["file_path"]; ok {
			filePath = val.GetStringValue()
		}

		if filePath == "" {
			continue
		}

		if _, exists := fileStats[filePath]; !exists {
			fileStats[filePath] = &FileStats{
				Path:       filePath,
				ChunkCount: 0,
				ModTime:    time.Now(),
			}
		}
		fileStats[filePath].ChunkCount++
	}

	result := make([]FileStats, 0, len(fileStats))
	for _, stat := range fileStats {
		result = append(result, *stat)
	}

	return result, nil
}

func (s *QdrantStore) GetChunksForFile(ctx context.Context, filePath string) ([]Chunk, error) {
	filter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewMatch("file_path", filePath),
		},
	}

	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Filter:         filter,
		Limit:          qdrant.PtrOf(uint32(10000)),
		WithPayload:    qdrant.NewWithPayloadInclude("file_path", "start_line", "end_line", "content", "hash", "updated_at", "chunk_id"),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get chunks: %w", err)
	}

	chunks := make([]Chunk, 0, len(scrollResult))
	for _, point := range scrollResult {
		chunk := s.parseChunkPayload(point.Payload)
		chunk.Vector = pointVector(point)
		chunks = append(chunks, *chunk)
	}

	return chunks, nil
}

func (s *QdrantStore) GetAllChunks(ctx context.Context) ([]Chunk, error) {
	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Limit:          qdrant.PtrOf(uint32(100000)),
		WithPayload:    qdrant.NewWithPayloadInclude("file_path", "start_line", "end_line", "content", "hash", "updated_at", "chunk_id"),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get all chunks: %w", err)
	}

	chunks := make([]Chunk, 0, len(scrollResult))
	for _, point := range scrollResult {
		chunk := s.parseChunkPayload(point.Payload)
		chunk.Vector = pointVector(point)
		chunks = append(chunks, *chunk)
	}

	return chunks, nil
}

// LookupByContentHash searches Qdrant for a point matching the content hash
// stored under the given file path. Both predicates are required: a vector
// encodes its file's identity, so cross-path reuse would attach the wrong
// file context to the new chunks.
func (s *QdrantStore) LookupByContentHash(ctx context.Context, contentHash, filePath string) ([]float32, bool, error) {
	if contentHash == "" {
		return nil, false, nil
	}
	if s.cacheDisabled {
		// Embedding configuration changed for this run: never reuse.
		return nil, false, nil
	}

	filter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewMatch("content_hash", contentHash),
			qdrant.NewMatch("file_path", filePath),
		},
	}

	scrollResult, err := s.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: s.collectionName,
		Filter:         filter,
		Limit:          qdrant.PtrOf(uint32(1)),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	if err != nil {
		return nil, false, fmt.Errorf("failed to lookup by content hash: %w", err)
	}

	if len(scrollResult) == 0 {
		return nil, false, nil
	}

	point := scrollResult[0]
	if vec := pointVector(point); vec != nil {
		return vec, true, nil
	}

	return nil, false, nil
}
