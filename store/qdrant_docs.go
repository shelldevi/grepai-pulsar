package store

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
)

// CacheFingerprint identifies the embedding configuration a document
// sidecar was written with. Vectors are only reusable for the same model,
// dimensions, chunking and collection; mixing configurations would attach
// wrong vectors to chunks.
type CacheFingerprint struct {
	Model        string `json:"model"`
	Dimensions   int    `json:"dimensions"`
	ChunkSize    int    `json:"chunk_size"`
	ChunkOverlap int    `json:"chunk_overlap"`
	Collection   string `json:"collection"`
}

// qdrantDocFile is the persisted sidecar shape.
type qdrantDocFile struct {
	Docs           map[string]Document
	Fingerprint    CacheFingerprint
	HasFingerprint bool
}

func (f CacheFingerprint) equal(o CacheFingerprint) bool {
	return f == o
}

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
	mu             sync.Mutex
	path           string
	docs           map[string]Document
	fp             CacheFingerprint
	hasFingerprint bool
	dirty          bool
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
// A corrupt or empty file is also not fatal: it is logged and rebuilt from
// scratch (the next scan re-queues everything and chunk lookups refill it).
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

	var stored qdrantDocFile
	if err := gob.NewDecoder(f).Decode(&stored); err != nil {
		log.Printf("Warning: qdrant document metadata %s is corrupt or from an older format (%v); rebuilding from scratch", s.path, err)
		s.docs = make(map[string]Document)
		s.fp = CacheFingerprint{}
		s.hasFingerprint = false
		s.dirty = true
		return nil
	}
	if stored.Docs == nil {
		stored.Docs = make(map[string]Document)
	}
	s.docs = stored.Docs
	s.fp = stored.Fingerprint
	s.hasFingerprint = stored.HasFingerprint
	s.dirty = false
	return nil
}

// adoptFingerprint records the current embedding configuration. The first
// run with a sidecar that has none adopts it silently and marks the store
// dirty so it is persisted. A mismatch returns false: the caller must
// discard the sidecar and disable cache lookups for the run.
func (s *qdrantDocStore) adoptFingerprint(fp CacheFingerprint) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.hasFingerprint {
		s.fp = fp
		s.hasFingerprint = true
		s.dirty = true
		return true
	}
	return s.fp.equal(fp)
}

// reset discards all tracked documents and the fingerprint.
func (s *qdrantDocStore) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.docs = make(map[string]Document)
	s.fp = CacheFingerprint{}
	s.hasFingerprint = false
	s.dirty = true
}

// fingerprint returns the recorded embedding configuration.
func (s *qdrantDocStore) fingerprint() CacheFingerprint {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.fp
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

// persist writes the metadata atomically (temp file + fsync + rename +
// directory fsync). It is a no-op unless the store changed since the last
// persist, so per-file SaveDocument calls stay cheap on large initial scans.
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
	stored := qdrantDocFile{
		Docs:           s.docs,
		Fingerprint:    s.fp,
		HasFingerprint: s.hasFingerprint,
	}
	if err := gob.NewEncoder(tmp).Encode(stored); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("failed to encode qdrant document metadata: %w", err)
	}
	// Flush file contents to stable storage before the rename so a crash
	// cannot leave a truncated sidecar behind.
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("failed to fsync temp qdrant document metadata file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("failed to close temp qdrant document metadata file: %w", err)
	}
	if err := os.Rename(tmpName, s.path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("failed to replace qdrant document metadata %s: %w", s.path, err)
	}
	// Fsync the directory so the rename itself is durable.
	if dir, err := os.Open(filepath.Dir(s.path)); err == nil {
		_ = dir.Sync()
		dir.Close()
	}

	s.dirty = false
	return nil
}
