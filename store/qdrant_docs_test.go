package store

import (
	"path/filepath"
	"testing"
	"time"
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
