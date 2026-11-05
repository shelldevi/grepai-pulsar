package indexer

import (
	"context"
	"errors"
	"testing"

	"github.com/yoanbernabeu/grepai/store"
)

// bulkMockStore is a mockStore with BulkReconcileStore support and a
// programmable scroll result, to verify the bulk reconcile path and its
// fail-closed behavior on scroll errors.
type bulkMockStore struct {
	mockStore
	refs        []store.PointRef
	scrollErr   error
	delPointIDs [][]string
}

func (m *bulkMockStore) ListPointRefs(ctx context.Context) ([]store.PointRef, error) {
	if m.scrollErr != nil {
		return nil, m.scrollErr
	}
	return m.refs, nil
}

func (m *bulkMockStore) DeletePointIDs(ctx context.Context, pointIDs []string) error {
	m.delPointIDs = append(m.delPointIDs, pointIDs)
	for _, pid := range pointIDs {
		for id, ch := range m.chunks {
			if store.PointIDForChunk(ch.ID) == pid {
				delete(m.chunks, id)
			}
		}
	}
	return nil
}

func newBulkTestIndexer(st *bulkMockStore) *Indexer {
	return &Indexer{root: "/test", store: st, chunker: NewChunker(64, 8)}
}

func TestReconcileBulk_ScrollErrorRemovesNothing(t *testing.T) {
	ctx := context.Background()
	st := &bulkMockStore{mockStore: *newMockStore(), scrollErr: errors.New("scroll boom")}
	st.documents["a.go"] = store.Document{Path: "a.go", Hash: "h", ChunkIDs: []string{"a.go_0"}}
	st.chunks["a.go_0"] = store.Chunk{ID: "a.go_0", FilePath: "a.go"}
	idx := newBulkTestIndexer(st)

	dropped, err := idx.ReconcileStoredChunks(ctx)
	if err == nil {
		t.Fatal("expected scroll error to propagate")
	}
	if dropped != 0 {
		t.Errorf("expected no dropped docs on scroll error, got %d", dropped)
	}
	if _, ok := st.documents["a.go"]; !ok {
		t.Error("document must be kept when the scroll fails")
	}
	if _, ok := st.chunks["a.go_0"]; !ok {
		t.Error("points must be kept when the scroll fails")
	}
	if idx.pendingSweep != nil {
		t.Error("no sweep may be armed when the scroll fails")
	}
	if swept, serr := idx.SweepUnexpectedPoints(ctx); swept != 0 || serr != nil {
		t.Errorf("sweep after failed scroll must be a no-op, got swept=%d err=%v", swept, serr)
	}
	if len(st.delPointIDs) != 0 {
		t.Error("no point deletion may happen when the scroll fails")
	}
}

func TestReconcileBulk_DropsDivergentDocAndSweepsOrphans(t *testing.T) {
	ctx := context.Background()
	st := &bulkMockStore{mockStore: *newMockStore()}
	// a.go intact; b.go shrunk (tail b.go_1 orphaned); gone.go deleted.
	st.documents["a.go"] = store.Document{Path: "a.go", Hash: "h", ChunkIDs: []string{"a.go_0"}}
	st.documents["b.go"] = store.Document{Path: "b.go", Hash: "h", ChunkIDs: []string{"b.go_0", "b.go_1"}}
	st.documents["c.go"] = store.Document{Path: "c.go", Hash: "h", ChunkIDs: []string{"c.go_0"}}
	st.documents["d.go"] = store.Document{Path: "d.go", Hash: "h", ChunkIDs: []string{"d.go_0"}}
	st.chunks["a.go_0"] = store.Chunk{ID: "a.go_0", FilePath: "a.go"}
	st.chunks["b.go_0"] = store.Chunk{ID: "b.go_0", FilePath: "b.go"}
	st.chunks["c.go_0"] = store.Chunk{ID: "c.go_0", FilePath: "c.go"}
	st.chunks["d.go_0"] = store.Chunk{ID: "d.go_0", FilePath: "d.go"}
	st.refs = []store.PointRef{
		{PointID: store.PointIDForChunk("a.go_0"), FilePath: "a.go"},
		{PointID: store.PointIDForChunk("b.go_0"), FilePath: "b.go"},
		{PointID: store.PointIDForChunk("c.go_0"), FilePath: "c.go"},
		{PointID: store.PointIDForChunk("d.go_0"), FilePath: "d.go"},
		{PointID: store.PointIDForChunk("gone.go_0"), FilePath: "gone.go"},
	}
	idx := newBulkTestIndexer(st)

	dropped, err := idx.ReconcileStoredChunks(ctx)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	if dropped != 1 {
		t.Errorf("expected b.go dropped (missing b.go_1), got %d", dropped)
	}
	if _, ok := st.documents["b.go"]; ok {
		t.Error("b.go must be dropped for re-indexing")
	}
	if _, ok := st.documents["a.go"]; !ok {
		t.Error("intact a.go must be kept")
	}

	swept, err := idx.SweepUnexpectedPoints(ctx)
	if err != nil {
		t.Fatalf("sweep failed: %v", err)
	}
	// gone.go_0 belongs to no document; b.go_0 was adopted... no:
	// b.go was dropped, so b.go_0 is unexpected too unless re-adopted.
	// Nothing re-adopted it here, so both are swept.
	if swept != 2 {
		t.Errorf("expected 2 swept points, got %d", swept)
	}
	if len(st.delPointIDs) != 1 || len(st.delPointIDs[0]) != 2 {
		t.Errorf("expected one point-ID deletion of 2 IDs, got %v", st.delPointIDs)
	}
	if _, ok := st.chunks["a.go_0"]; !ok {
		t.Error("a.go_0 must survive the sweep")
	}
}

func TestReconcileBulk_GuardBlocksMajoritySweep(t *testing.T) {
	ctx := context.Background()
	st := &bulkMockStore{mockStore: *newMockStore()}
	st.documents["a.go"] = store.Document{Path: "a.go", Hash: "h", ChunkIDs: []string{"a.go_0"}}
	st.chunks["a.go_0"] = store.Chunk{ID: "a.go_0", FilePath: "a.go"}
	st.refs = []store.PointRef{
		{PointID: store.PointIDForChunk("a.go_0"), FilePath: "a.go"},
		{PointID: store.PointIDForChunk("junk_0"), FilePath: "junk.go"},
		{PointID: store.PointIDForChunk("junk_1"), FilePath: "junk.go"},
		{PointID: store.PointIDForChunk("junk_2"), FilePath: "junk.go"},
	}
	idx := newBulkTestIndexer(st)

	if _, err := idx.ReconcileStoredChunks(ctx); err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	swept, err := idx.SweepUnexpectedPoints(ctx)
	if err != nil {
		t.Fatalf("sweep failed: %v", err)
	}
	if swept != 0 {
		t.Errorf("guard must block a 3-of-4 sweep, got %d", swept)
	}
	if len(st.delPointIDs) != 0 {
		t.Error("guarded sweep must delete nothing")
	}
}
