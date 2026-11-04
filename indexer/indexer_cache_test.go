package indexer

import (
	"context"
	"errors"
	"testing"

	"github.com/yoanbernabeu/grepai/store"
)

// cacheMockStore is a VectorStore + EmbeddingCache with programmable lookup
// results and call tracking, to verify lookup-before-delete ordering and
// fail-closed behavior.
type cacheMockStore struct {
	mockStore
	lookupVecs map[string][]float32
	lookupErr  error
	lookups    int
}

func (m *cacheMockStore) LookupByContentHash(ctx context.Context, contentHash, filePath string) ([]float32, bool, error) {
	m.lookups++
	if m.lookupErr != nil {
		return nil, false, m.lookupErr
	}
	vec, ok := m.lookupVecs[contentHash]
	return vec, ok, nil
}

func newCacheTestIndexer(st *cacheMockStore, emb *mockBatchEmbedder) *Indexer {
	return &Indexer{
		root:     "/test",
		store:    st,
		embedder: emb,
		chunker:  NewChunker(64, 8),
	}
}

func cachedVec(v float32) []float32 { return []float32{v, v, v} }

func totalBatchInputs(emb *mockBatchEmbedder) int {
	total := 0
	for _, b := range emb.batchInputs {
		total += len(b)
	}
	return total
}

func TestIndexFile_ReusesCachedChunks(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()

	// Seed the store as if a previous version of the file was indexed.
	oldContent := "package main\n\nfunc main() {\n\tprintln(\"v1\")\n}\n"
	oldInfos := NewChunker(64, 8).ChunkWithContext("file.go", oldContent)
	var oldIDs []string
	for i, ci := range oldInfos {
		st.lookupVecs[ci.ContentHash] = cachedVec(float32(i + 1))
		st.chunks[ci.ID] = store.Chunk{ID: ci.ID, FilePath: "file.go", Content: ci.Content, ContentHash: ci.ContentHash, Vector: cachedVec(1)}
		oldIDs = append(oldIDs, ci.ID)
	}
	st.documents["file.go"] = store.Document{Path: "file.go", Hash: "oldhash", ChunkIDs: oldIDs}

	// One-line change: most chunks keep their content hash.
	newContent := "package main\n\nfunc main() {\n\tprintln(\"v2\")\n}\n"
	file := FileInfo{Path: "file.go", Content: newContent, Hash: "newhash", ModTime: 2000}

	idx := newCacheTestIndexer(st, emb)
	n, err := idx.IndexFile(ctx, file)
	if err != nil {
		t.Fatalf("IndexFile failed: %v", err)
	}
	if n == 0 {
		t.Fatal("expected chunks to be indexed")
	}
	// Only the changed chunk(s) may have been embedded; the unchanged ones
	// must come from the cache.
	if got := totalBatchInputs(emb); got > 2 {
		t.Errorf("expected at most 2 freshly embedded chunks for a one-line edit, got %d", got)
	}
	// Old positions must be replaced via upsert-then-delete-stale: every
	// stored chunk must carry a current ID, with no stale leftovers.
	if !st.saveChunksCalled {
		t.Error("expected SaveChunks to store merged vectors")
	}
	infos := NewChunker(64, 8).ChunkWithContext("file.go", newContent)
	wantIDs := make(map[string]bool)
	for _, ci := range infos {
		wantIDs[ci.ID] = true
	}
	if len(st.chunks) != len(wantIDs) {
		t.Errorf("expected %d stored chunks with current IDs, got %d", len(wantIDs), len(st.chunks))
	}
	for id := range st.chunks {
		if !wantIDs[id] {
			t.Errorf("stale chunk left in store: %s", id)
		}
	}
}

func TestIndexFile_LookupErrorAbortsBeforeEmbed(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupErr: errors.New("qdrant down")}
	emb := newMockBatchEmbedder()
	idx := newCacheTestIndexer(st, emb)

	_, err := idx.IndexFile(ctx, FileInfo{Path: "file.go", Content: "package main\n", Hash: "h", ModTime: 1})
	if err == nil {
		t.Fatal("expected lookup failure to abort IndexFile")
	}
	if emb.embedCalled || emb.batchCalls > 0 {
		t.Error("embedder must not be called when the cache lookup fails")
	}
	if st.delByFileCalled {
		t.Error("DeleteByFile must not run when the lookup fails: the old index stays intact")
	}
	if st.saveChunksCalled {
		t.Error("SaveChunks must not run when the lookup fails")
	}
}

func TestIndexFilesBatched_PartialReuse(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()
	idx := newCacheTestIndexer(st, emb)

	content := "package main\n\n" +
		"func alpha() { println(\"alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha\") }\n\n" +
		"func beta() { println(\"beta beta beta beta beta beta beta beta beta beta beta beta beta beta\") }\n\n" +
		"func gamma() { println(\"gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma gamma\") }\n\n" +
		"func delta() { println(\"delta delta delta delta delta delta delta delta delta delta delta\") }\n"
	infos := NewChunker(64, 8).ChunkWithContext("file.go", content)
	if len(infos) < 3 {
		t.Fatalf("need at least 3 chunks for this test, got %d", len(infos))
	}
	// Pretend all but the first chunk are already indexed (e.g. under an
	// older revision of this file or in another file).
	for _, ci := range infos[1:] {
		st.lookupVecs[ci.ContentHash] = cachedVec(7)
	}

	n, c, err := idx.indexFilesBatched(ctx, []FileInfo{{Path: "file.go", Content: content, Hash: "h", ModTime: 1}}, emb, nil)
	if err != nil {
		t.Fatalf("indexFilesBatched failed: %v", err)
	}
	if n != 1 || c != len(infos) {
		t.Errorf("expected 1 file with %d chunks, got %d files %d chunks", len(infos), n, c)
	}
	if got := totalBatchInputs(emb); got != 1 {
		t.Errorf("expected exactly 1 freshly embedded chunk, got %d", got)
	}
	// All chunks must be stored, including the reused ones.
	if len(st.chunks) != len(infos) {
		t.Errorf("expected %d stored chunks, got %d", len(infos), len(st.chunks))
	}
}

func TestIndexFilesBatched_LookupErrorAbortsWithoutPay(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupErr: errors.New("connection refused")}
	emb := newMockBatchEmbedder()
	idx := newCacheTestIndexer(st, emb)

	_, _, err := idx.indexFilesBatched(ctx,
		[]FileInfo{{Path: "a.go", Content: "package a\n", Hash: "h1", ModTime: 1}},
		emb, nil)
	if err == nil {
		t.Fatal("expected lookup failure to abort the batch")
	}
	if emb.batchCalls > 0 {
		t.Error("EmbedBatches must not be called when a lookup fails")
	}
	if st.delByFileCalled || st.saveChunksCalled {
		t.Error("store must not be mutated when a lookup fails")
	}
}

func TestIndexFile_EmbedErrorKeepsOldChunks(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()
	emb.embedErr = errors.New("provider 500")
	idx := newCacheTestIndexer(st, emb)

	_, err := idx.IndexFile(ctx, FileInfo{Path: "file.go", Content: "package main\n", Hash: "h", ModTime: 1})
	if err == nil {
		t.Fatal("expected embed failure to propagate")
	}
	if st.delByFileCalled {
		t.Error("DeleteByFile must not run when embedding fails: old chunks stay indexed")
	}
}

func TestIndexFile_EmptyFileRemovesStaleState(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()
	idx := newCacheTestIndexer(st, emb)

	// Seed stale state as if a previous non-empty version was indexed.
	st.chunks["file.go_0"] = store.Chunk{ID: "file.go_0", FilePath: "file.go", Content: "old", Vector: cachedVec(1)}
	st.documents["file.go"] = store.Document{Path: "file.go", Hash: "oldhash", ChunkIDs: []string{"file.go_0"}}

	n, err := idx.IndexFile(ctx, FileInfo{Path: "file.go", Content: "   \n  \n", Hash: "emptyhash", ModTime: 2})
	if err != nil {
		t.Fatalf("IndexFile failed: %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 chunks for whitespace file, got %d", n)
	}
	if emb.embedCalled || emb.batchCalls > 0 {
		t.Error("nothing must be embedded for an empty file")
	}
	if len(st.chunks) != 0 {
		t.Errorf("expected stale chunks removed, got %d left", len(st.chunks))
	}
	if _, ok := st.documents["file.go"]; ok {
		t.Error("expected document removed for empty file")
	}
}

func TestIndexFilesBatched_EmptyFileRemovesStaleState(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()
	idx := newCacheTestIndexer(st, emb)

	st.chunks["gone.go_0"] = store.Chunk{ID: "gone.go_0", FilePath: "gone.go", Content: "old", Vector: cachedVec(1)}
	st.documents["gone.go"] = store.Document{Path: "gone.go", Hash: "oldhash", ChunkIDs: []string{"gone.go_0"}}

	n, c, err := idx.indexFilesBatched(ctx, []FileInfo{
		{Path: "gone.go", Content: "", Hash: "emptyhash", ModTime: 2},
		{Path: "new.go", Content: "package new\n", Hash: "h2", ModTime: 2},
	}, emb, nil)
	if err != nil {
		t.Fatalf("indexFilesBatched failed: %v", err)
	}
	if n != 2 {
		t.Errorf("expected both files processed, got %d", n)
	}
	if _, ok := st.documents["gone.go"]; ok {
		t.Error("expected document removed for emptied file")
	}
	for id := range st.chunks {
		if len(id) >= 7 && id[:7] == "gone.go" {
			t.Errorf("stale chunk left: %s", id)
		}
	}
	_ = c
}

func TestStaleChunkIDs(t *testing.T) {
	got := staleChunkIDs([]string{"a", "b", "c"}, []string{"b", "d"})
	if len(got) != 2 || got[0] != "a" || got[1] != "c" {
		t.Errorf("unexpected stale set: %v", got)
	}
	if len(staleChunkIDs(nil, []string{"x"})) != 0 {
		t.Error("expected no stale IDs without old IDs")
	}
}

func TestReconcileStoredChunks(t *testing.T) {
	ctx := context.Background()
	newIdx := func() (*Indexer, *cacheMockStore) {
		st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
		return &Indexer{root: "/test", store: st, chunker: NewChunker(64, 8)}, st
	}

	// Orphan points (in store, not in doc) are deleted; doc is kept.
	idx, st := newIdx()
	st.documents["a.go"] = store.Document{Path: "a.go", Hash: "h", ChunkIDs: []string{"a.go_0"}}
	st.chunks["a.go_0"] = store.Chunk{ID: "a.go_0", FilePath: "a.go"}
	st.chunks["a.go_9"] = store.Chunk{ID: "a.go_9", FilePath: "a.go"}
	dropped, err := idx.ReconcileStoredChunks(ctx)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	if dropped != 0 {
		t.Errorf("expected no dropped docs, got %d", dropped)
	}
	if _, ok := st.chunks["a.go_9"]; ok {
		t.Error("expected orphan chunk to be deleted")
	}
	if _, ok := st.chunks["a.go_0"]; !ok {
		t.Error("expected live chunk to survive")
	}

	// Documents listing missing points are dropped for re-indexing.
	idx, st = newIdx()
	st.documents["b.go"] = store.Document{Path: "b.go", Hash: "h", ChunkIDs: []string{"b.go_0", "b.go_1"}}
	st.chunks["b.go_0"] = store.Chunk{ID: "b.go_0", FilePath: "b.go"}
	dropped, err = idx.ReconcileStoredChunks(ctx)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	if dropped != 1 {
		t.Errorf("expected 1 dropped doc, got %d", dropped)
	}
	if _, ok := st.documents["b.go"]; ok {
		t.Error("expected missing-point document to be dropped")
	}

	// Clean state: no changes.
	idx, st = newIdx()
	st.documents["c.go"] = store.Document{Path: "c.go", Hash: "h", ChunkIDs: []string{"c.go_0"}}
	st.chunks["c.go_0"] = store.Chunk{ID: "c.go_0", FilePath: "c.go"}
	dropped, err = idx.ReconcileStoredChunks(ctx)
	if err != nil || dropped != 0 {
		t.Errorf("expected clean no-op, got dropped=%d err=%v", dropped, err)
	}
	if !st.delIDsCalled && len(st.delIDs) != 0 {
		t.Error("expected no deletions on clean state")
	}
}
