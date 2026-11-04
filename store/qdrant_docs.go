package store

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// qdrantDocStore is a small GOB-persisted sidecar holding per-file document
// metadata for the Qdrant backend.
//
// Qdrant point payloads carry chunk data only (file path, lines, content,
// hashes) — there is no place for file-level metadata such as the file hash,
// mod time, or the chunk-ID list. QdrantStore.SaveDocument used to be a no-op
// and GetDocument a stub, so the indexer could never skip an unchanged file:
// every scan re-queued every file and every watch event re-embedded the
// whole file. This sidecar restores file-level skip gates for Qdrant without
// touching point payloads or search results.
//
// It is only active when the store is constructed with a document path (see
// NewQdrantStore). Read-only consumers (search, status, MCP) construct the
// store without a path and keep the previous behavior.
type qdrantDocStore struct {
	mu    sync.Mutex
	path  string
	docs  map[string]Document
	dirty bool
}

func newQdrantDocStore(path string) *qdrantDocStore {
	return &qdrantDocStore{
		path: path,
		docs: make(map[string]Document),
	}
}

// load reads persisted metadata. A missing file is not an error — it means
// this backend has not been indexed with document tracking yet (e.g. an
// index written before this sidecar existed). The first scan backfills it.
func (s *qdrantDocStore) load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	f, err := os.Open(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to open qdrant document metadata %s: %w", s.path, err)
	}
	defer f.Close()

	docs := make(map[string]Document)
	if err := gob.NewDecoder(f).Decode(&docs); err != nil {
		return fmt.Errorf("failed to decode qdrant document metadata %s: %w", s.path, err)
	}
	s.docs = docs
	s.dirty = false
	return nil
}

func (s *qdrantDocStore) get(filePath string) (Document, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	doc, ok := s.docs[filePath]
	return doc, ok
}

func (s *qdrantDocStore) set(doc Document) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.docs[doc.Path] = doc
	s.dirty = true
}

func (s *qdrantDocStore) del(filePath string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.docs[filePath]; ok {
		delete(s.docs, filePath)
		s.dirty = true
	}
}

func (s *qdrantDocStore) list() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	paths := make([]string, 0, len(s.docs))
	for path := range s.docs {
		paths = append(paths, path)
	}
	return paths
}

// persist writes the metadata atomically (temp file + rename). It is a no-op
// unless the store changed since the last persist, so per-file SaveDocument
// calls stay cheap on large initial scans.
func (s *qdrantDocStore) persist() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.dirty {
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("failed to create qdrant document metadata dir: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.path), "qdrant_docs-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp qdrant document metadata file: %w", err)
	}
	tmpName := tmp.Name()
	if err := gob.NewEncoder(tmp).Encode(s.docs); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("failed to encode qdrant document metadata: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("failed to close temp qdrant document metadata file: %w", err)
	}
	if err := os.Rename(tmpName, s.path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("failed to replace qdrant document metadata %s: %w", s.path, err)
	}

	s.dirty = false
	return nil
}
