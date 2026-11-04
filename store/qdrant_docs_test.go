package store

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
)

func TestQdrantDocStore_RoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qdrant_docs.gob")
	s := newQdrantDocStore(path)

	if _, ok := s.get("a.go"); ok {
		t.Fatal("expected miss on empty store")
	}

	doc := Document{Path: "a.go", Hash: "abc", ModTime: time.Now().Truncate(time.Second), ChunkIDs: []string{"a.go_0", "a.go_1"}}
	s.set(doc)
	if got, ok := s.get("a.go"); !ok || got.Hash != "abc" || len(got.ChunkIDs) != 2 {
		t.Fatalf("get after set failed: %+v %v", got, ok)
	}

	if paths := s.list(); len(paths) != 1 || paths[0] != "a.go" {
		t.Fatalf("unexpected list: %v", paths)
	}

	if err := s.persist(); err != nil {
		t.Fatalf("persist failed: %v", err)
	}

	// A fresh instance must load what the first one wrote.
	s2 := newQdrantDocStore(path)
	if err := s2.load(); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if got, ok := s2.get("a.go"); !ok || got.Hash != "abc" {
		t.Fatalf("load round-trip failed: %+v %v", got, ok)
	}

	s2.del("a.go")
	if _, ok := s2.get("a.go"); ok {
		t.Fatal("expected miss after delete")
	}
	if err := s2.persist(); err != nil {
		t.Fatalf("persist after delete failed: %v", err)
	}
	s3 := newQdrantDocStore(path)
	if err := s3.load(); err != nil {
		t.Fatalf("reload failed: %v", err)
	}
	if len(s3.list()) != 0 {
		t.Fatalf("expected empty store after delete+persist, got %v", s3.list())
	}
}

func TestQdrantDocStore_LoadMissingIsEmpty(t *testing.T) {
	s := newQdrantDocStore(filepath.Join(t.TempDir(), "does-not-exist.gob"))
	if err := s.load(); err != nil {
		t.Fatalf("load of missing file must not fail: %v", err)
	}
	if len(s.list()) != 0 {
		t.Fatalf("expected empty store, got %v", s.list())
	}
	// Persist with no changes must not create a file.
	if err := s.persist(); err != nil {
		t.Fatalf("clean persist must not fail: %v", err)
	}
}

func TestPointVector_FlatAndOneofForms(t *testing.T) {
	// Current servers return the deprecated flat Data field, not the oneof.
	flat := &qdrant.RetrievedPoint{
		Vectors: &qdrant.VectorsOutput{
			VectorsOptions: &qdrant.VectorsOutput_Vector{
				Vector: &qdrant.VectorOutput{Data: []float32{0.1, 0.2}},
			},
		},
	}
	if got := pointVector(flat); len(got) != 2 || got[0] != 0.1 {
		t.Errorf("flat form not extracted: %v", got)
	}

	// The oneof form must keep working.
	oneof := &qdrant.RetrievedPoint{
		Vectors: &qdrant.VectorsOutput{
			VectorsOptions: &qdrant.VectorsOutput_Vector{
				Vector: &qdrant.VectorOutput{
					Vector: &qdrant.VectorOutput_Dense{
						Dense: &qdrant.DenseVector{Data: []float32{0.3, 0.4}},
					},
				},
			},
		},
	}
	if got := pointVector(oneof); len(got) != 2 || got[0] != 0.3 {
		t.Errorf("oneof form not extracted: %v", got)
	}

	if pointVector(nil) != nil {
		t.Error("nil point must yield nil vector")
	}
	if pointVector(&qdrant.RetrievedPoint{}) != nil {
		t.Error("point without vectors must yield nil vector")
	}
}

func TestQdrantDocStore_Fingerprint(t *testing.T) {
	s := newQdrantDocStore(filepath.Join(t.TempDir(), "qdrant_docs.gob"))
	fp := CacheFingerprint{Model: "m", Dimensions: 64, ChunkSize: 512, ChunkOverlap: 50, Collection: "c"}

	// First adopt (no fingerprint yet) succeeds and marks dirty.
	if !s.adoptFingerprint(fp) {
		t.Fatal("expected first adopt to succeed")
	}
	// Same fingerprint adopts cleanly.
	if !s.adoptFingerprint(fp) {
		t.Fatal("expected same fingerprint to adopt")
	}
	// Mismatch refuses.
	other := fp
	other.Model = "other"
	if s.adoptFingerprint(other) {
		t.Fatal("expected mismatch to refuse adoption")
	}
	// Reset clears everything including the fingerprint.
	s.reset()
	if len(s.list()) != 0 {
		t.Fatal("expected reset to clear docs")
	}
	if !s.adoptFingerprint(other) {
		t.Fatal("expected adopt after reset to succeed")
	}

	// Fingerprint survives a persist/load round-trip.
	if err := s.persist(); err != nil {
		t.Fatalf("persist failed: %v", err)
	}
	s2 := newQdrantDocStore(s.path)
	if err := s2.load(); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if !s2.adoptFingerprint(other) {
		t.Fatal("expected loaded fingerprint to match")
	}
	if s2.adoptFingerprint(fp) {
		t.Fatal("expected loaded fingerprint to reject a different one")
	}
}

func TestQdrantDocStore_CorruptRebuilds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qdrant_docs.gob")
	if err := os.WriteFile(path, []byte("not a gob file{{{}}}"), 0600); err != nil {
		t.Fatalf("failed to write corrupt file: %v", err)
	}
	s := newQdrantDocStore(path)
	if err := s.load(); err != nil {
		t.Fatalf("corrupt sidecar must not fail startup: %v", err)
	}
	if len(s.list()) != 0 {
		t.Fatalf("corrupt sidecar must rebuild empty, got %v", s.list())
	}
}

func TestQdrantStore_SetFingerprintDisablesCacheOnMismatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qdrant_docs.gob")
	qs := &QdrantStore{docs: newQdrantDocStore(path)}
	fp := CacheFingerprint{Model: "m", Dimensions: 64, ChunkSize: 512, ChunkOverlap: 50, Collection: "c"}
	qs.SetFingerprint(fp)
	if qs.cacheDisabled {
		t.Fatal("first fingerprint must not disable the cache")
	}
	// Same config: still enabled.
	qs.SetFingerprint(fp)
	if qs.cacheDisabled {
		t.Fatal("same fingerprint must not disable the cache")
	}
	// Changed model: discard + disable, loudly (log goes to stderr).
	changed := fp
	changed.Model = "m2"
	qs.SetFingerprint(changed)
	if !qs.cacheDisabled {
		t.Fatal("mismatched fingerprint must disable cache lookups")
	}
	if n := len(qs.docs.list()); n != 0 {
		t.Fatalf("mismatch must discard sidecar docs, got %d", n)
	}
	if _, found, err := qs.LookupByContentHash(context.Background(), "abc", "f.go"); err != nil || found {
		t.Fatalf("disabled cache must miss cleanly, got %v %v", found, err)
	}
}
