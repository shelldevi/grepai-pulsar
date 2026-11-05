package indexer

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/yoanbernabeu/grepai/embedder"
	"github.com/yoanbernabeu/grepai/store"
)

// TestIndexFileGroup_ContextLengthFallsBackToPerFile verifies R6: a
// context-length error from EmbedBatches falls back to the existing
// per-file path (IndexFile) for the group's files instead of aborting.
func TestIndexFileGroup_ContextLengthFallsBackToPerFile(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()
	emb.batchErr = &embedder.ContextLengthError{ChunkIndex: 0, Message: "too long"}
	idx := newCacheTestIndexer(st, emb)

	n, c, failed, err := idx.indexFilesBatched(ctx, []FileInfo{
		{Path: "a.go", Content: "package a\n", Hash: "h1", ModTime: 1},
		{Path: "b.go", Content: "package b\n", Hash: "h2", ModTime: 1},
	}, emb, nil)
	if err != nil {
		t.Fatalf("context-length error must fall back, not abort: %v", err)
	}
	if failed != 0 {
		t.Errorf("expected no failed files, got %d", failed)
	}
	if n != 2 || c != 2 {
		t.Errorf("expected 2 files / 2 chunks via fallback, got %d / %d", n, c)
	}
	if emb.batchCalls != 1 {
		t.Errorf("expected exactly 1 batch attempt, got %d", emb.batchCalls)
	}
	if !emb.embedCalled {
		t.Error("expected per-file EmbedBatch to be used by the fallback")
	}
	if len(st.chunks) != 2 {
		t.Errorf("expected 2 stored chunks, got %d", len(st.chunks))
	}
}

// TestIndexFileGroup_PersistentFailureCountsWithoutAbort verifies that files
// failing even the per-file fallback are counted, not fatal to the group.
func TestIndexFileGroup_PersistentFailureCountsWithoutAbort(t *testing.T) {
	ctx := context.Background()
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	emb := newMockBatchEmbedder()
	emb.batchErr = &embedder.ContextLengthError{ChunkIndex: 0, Message: "too long"}
	emb.embedErr = errors.New("store down")
	idx := newCacheTestIndexer(st, emb)

	n, _, failed, err := idx.indexFilesBatched(ctx, []FileInfo{
		{Path: "a.go", Content: "package a\n", Hash: "h1", ModTime: 1},
		{Path: "b.go", Content: "package b\n", Hash: "h2", ModTime: 1},
	}, emb, nil)
	if err != nil {
		t.Fatalf("fallback failures must not abort the group: %v", err)
	}
	if failed != 2 || n != 0 {
		t.Errorf("expected 0 indexed / 2 failed, got %d / %d", n, failed)
	}
}

// TestIndexAll_GoneGuardRefusesMajority verifies R5: removing more than half
// the known documents — even with a clean scan — is refused loudly.
func TestIndexAll_GoneGuardRefusesMajority(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	keep := []string{"keep0.go", "keep1.go", "keep2.go", "keep3.go"}
	for _, name := range keep {
		if err := os.WriteFile(filepath.Join(tmpDir, name), []byte("package main\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	for _, name := range keep {
		st.documents[name] = store.Document{Path: name, Hash: "same", ChunkIDs: []string{name + "_0"}}
	}
	for i := 0; i < 6; i++ {
		name := fmt.Sprintf("gone%d.go", i)
		st.documents[name] = store.Document{Path: name, Hash: "h", ChunkIDs: []string{name + "_0"}}
		st.chunks[name+"_0"] = store.Chunk{ID: name + "_0", FilePath: name}
	}
	emb := newMockBatchEmbedder()
	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatal(err)
	}
	idx := &Indexer{root: tmpDir, store: st, embedder: emb, chunker: NewChunker(64, 8), scanner: NewScanner(tmpDir, ignoreMatcher)}

	stats, err := idx.IndexAllWithBatchProgress(ctx, nil, nil)
	if err != nil {
		t.Fatalf("IndexAll failed: %v", err)
	}
	if stats.Incomplete {
		t.Error("clean scan must not be marked incomplete")
	}
	if stats.FilesRemoved != 0 {
		t.Errorf("guard must refuse majority removal, removed %d", stats.FilesRemoved)
	}
	if len(st.documents) != 10 {
		t.Errorf("no document may be dropped by a refused sweep, %d left", len(st.documents))
	}
}

// TestIndexAll_IncompleteScanDeletesNothing verifies R5: a read error marks
// the scan incomplete and no gone file is removed.
func TestIndexAll_IncompleteScanDeletesNothing(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmpDir, "good.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatal(err)
	}
	badPath := filepath.Join(tmpDir, "bad.go")
	if err := os.WriteFile(badPath, []byte("package main\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(badPath, 0000); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(badPath, 0644)

	st := &cacheMockStore{mockStore: *newMockStore(), lookupVecs: map[string][]float32{}}
	// good.go unchanged (hash of its content is unknown here; force skip by
	// absence of need: listing it keeps it out of gone).
	st.documents["good.go"] = store.Document{Path: "good.go", Hash: "match-me", ChunkIDs: []string{"good.go_0"}}
	for _, name := range []string{"gone1.go", "gone2.go", "gone3.go"} {
		st.documents[name] = store.Document{Path: name, Hash: "h", ChunkIDs: []string{name + "_0"}}
		st.chunks[name+"_0"] = store.Chunk{ID: name + "_0", FilePath: name}
	}
	emb := newMockBatchEmbedder()
	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatal(err)
	}
	idx := &Indexer{root: tmpDir, store: st, embedder: emb, chunker: NewChunker(64, 8), scanner: NewScanner(tmpDir, ignoreMatcher)}

	stats, err := idx.IndexAllWithBatchProgress(ctx, nil, nil)
	if err != nil {
		t.Fatalf("IndexAll failed: %v", err)
	}
	if !stats.Incomplete {
		t.Error("unreadable file must mark the scan incomplete")
	}
	if stats.FilesRemoved != 0 {
		t.Errorf("incomplete scan must delete 0 gone files, removed %d", stats.FilesRemoved)
	}
	for _, name := range []string{"gone1.go", "gone2.go", "gone3.go"} {
		if _, ok := st.documents[name]; !ok {
			t.Errorf("document %s must survive an incomplete scan", name)
		}
	}
}

// failSingleEmbedder fails the non-batch path only, to exercise the
// sequential fallback in IndexAllWithBatchProgress.
type failSingleEmbedder struct {
	mockEmbedder
}

func (f *failSingleEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return nil, errors.New("provider down")
}

// TestIndexAll_NonBatchFailureMarksIncomplete verifies R5: a per-file failure
// on the sequential path marks the scan incomplete and spares gone files.
func TestIndexAll_NonBatchFailureMarksIncomplete(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmpDir, "a.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatal(err)
	}
	st := newMockStore()
	for _, name := range []string{"gone1.go", "gone2.go"} {
		st.documents[name] = store.Document{Path: name, Hash: "h", ChunkIDs: []string{name + "_0"}}
	}
	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatal(err)
	}
	idx := &Indexer{root: tmpDir, store: st, embedder: &failSingleEmbedder{}, chunker: NewChunker(64, 8), scanner: NewScanner(tmpDir, ignoreMatcher)}

	stats, err := idx.IndexAllWithBatchProgress(ctx, nil, nil)
	if err != nil {
		t.Fatalf("IndexAll failed: %v", err)
	}
	if !stats.Incomplete || stats.FailedFiles != 1 {
		t.Errorf("expected incomplete with 1 failed file, got %+v", stats)
	}
	if stats.FilesRemoved != 0 {
		t.Errorf("incomplete scan must delete 0 gone files, removed %d", stats.FilesRemoved)
	}
	if len(st.documents) != 2 {
		t.Errorf("gone documents must survive, %d left", len(st.documents))
	}
}

// TestScanMetadata_CountsWalkErrors verifies the metadata pass counts
// unreadable directories instead of silently skipping them.
func TestScanMetadata_CountsWalkErrors(t *testing.T) {
	tmpDir := t.TempDir()
	sub := filepath.Join(tmpDir, "locked")
	if err := os.Mkdir(sub, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "x.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(sub, 0000); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(sub, 0755)

	ignoreMatcher, err := NewIgnoreMatcher(tmpDir, []string{}, "")
	if err != nil {
		t.Fatal(err)
	}
	scanner := NewScanner(tmpDir, ignoreMatcher)
	_, _, scanErrs, err := scanner.ScanMetadata()
	if err != nil {
		t.Fatalf("ScanMetadata failed: %v", err)
	}
	if scanErrs == 0 {
		t.Error("unreadable directory must be counted as a scan error")
	}
}
