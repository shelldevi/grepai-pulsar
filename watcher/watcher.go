package watcher

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/yoanbernabeu/grepai/indexer"
)

type EventType int

const (
	EventCreate EventType = iota
	EventModify
	EventDelete
	EventRename
)

type FileEvent struct {
	Type EventType
	Path string
}

type Watcher struct {
	root       string
	watcher    *fsnotify.Watcher
	ignore     *indexer.IgnoreMatcher
	debounceMs int
	maxWaitMs  int
	events     chan FileEvent
	done       chan struct{}

	// Debouncing state
	pending    map[string]FileEvent
	pendingMu  sync.Mutex
	timer      *time.Timer
	windowStart time.Time
}

func NewWatcher(root string, ignore *indexer.IgnoreMatcher, debounceMs int) (*Watcher, error) {
	fsw, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}

	return &Watcher{
		root:       root,
		watcher:    fsw,
		ignore:     ignore,
		debounceMs: debounceMs,
		events:     make(chan FileEvent, 100),
		done:       make(chan struct{}),
		pending:    make(map[string]FileEvent),
	}, nil
}

// SetMaxWaitMs bounds how long a batch may be held while events keep
// arriving. The debounce timer alone resets on every event, so a
// continuously active repo (or a very long debounce such as 30 minutes)
// would starve indexing indefinitely. When set (>0), the first event that
// arrives at least maxWaitMs after the oldest pending event flushes the
// batch immediately. Zero (the default) keeps the previous behavior.
func (w *Watcher) SetMaxWaitMs(maxWaitMs int) {
	w.pendingMu.Lock()
	defer w.pendingMu.Unlock()
	w.maxWaitMs = maxWaitMs
}

func (w *Watcher) Start(ctx context.Context) error {
	// Add root directory and all subdirectories
	if err := w.addRecursive(w.root); err != nil {
		return err
	}

	// Start event processing
	go w.processEvents(ctx)

	return nil
}

func (w *Watcher) Events() <-chan FileEvent {
	return w.events
}

func (w *Watcher) Close() error {
	close(w.done)
	return w.watcher.Close()
}

func (w *Watcher) addRecursive(root string) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip inaccessible paths
		}

		relPath, err := filepath.Rel(w.root, path)
		if err != nil {
			return nil
		}

		// Handle directories: use ShouldSkipDir to respect .grepaiignore negations
		if info.IsDir() {
			if w.ignore.ShouldSkipDir(relPath) {
				return filepath.SkipDir
			}
			// Directory is not skipped; watch it if not individually ignored
			if !w.ignore.ShouldIgnore(relPath) {
				if err := w.watcher.Add(path); err != nil {
					log.Printf("Failed to watch %s: %v", path, err)
				}
			}
			return nil
		}

		// Skip ignored files
		if w.ignore.ShouldIgnore(relPath) {
			return nil
		}

		return nil
	})
}

func (w *Watcher) processEvents(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-w.done:
			return
		case event, ok := <-w.watcher.Events:
			if !ok {
				return
			}
			w.handleEvent(event)
		case err, ok := <-w.watcher.Errors:
			if !ok {
				return
			}
			log.Printf("Watcher error: %v", err)
		}
	}
}

func (w *Watcher) handleEvent(event fsnotify.Event) {
	relPath, err := filepath.Rel(w.root, event.Name)
	if err != nil {
		return
	}

	// Ignore hidden files and ignored paths
	if strings.HasPrefix(filepath.Base(relPath), ".") {
		return
	}
	if w.ignore.ShouldIgnore(relPath) {
		return
	}

	// Check if it's a supported file
	ext := strings.ToLower(filepath.Ext(event.Name))
	if !indexer.SupportedExtensions[ext] {
		// Check if it's a directory (for watching new directories)
		info, err := os.Stat(event.Name)
		if err != nil || !info.IsDir() {
			return
		}

		// New directory created, add to watcher
		if event.Has(fsnotify.Create) {
			if err := w.addRecursive(event.Name); err != nil {
				log.Printf("Failed to add new directory %s: %v", event.Name, err)
			}
		}
		return
	}

	var evType EventType
	switch {
	case event.Has(fsnotify.Create):
		evType = EventCreate
	case event.Has(fsnotify.Write):
		evType = EventModify
	case event.Has(fsnotify.Remove):
		evType = EventDelete
	case event.Has(fsnotify.Rename):
		evType = EventRename
	default:
		return
	}

	w.debounceEvent(FileEvent{
		Type: evType,
		Path: relPath,
	})
}

func (w *Watcher) debounceEvent(event FileEvent) {
	w.pendingMu.Lock()

	// A new window starts when nothing is pending.
	if len(w.pending) == 0 {
		w.windowStart = time.Now()
	}

	// Merge events: delete > create/modify
	existing, exists := w.pending[event.Path]
	if exists && existing.Type == EventDelete && event.Type != EventDelete {
		// Keep delete if file was deleted then recreated quickly
		// This will be handled as delete + create
	} else {
		w.pending[event.Path] = event
	}

	// Max-wait bound: flush now if the oldest pending event has waited
	// long enough, no matter how active the repo is. This is checked on
	// event arrival, so the overshoot is at most the inter-arrival gap.
	flushNow := w.maxWaitMs > 0 && !w.windowStart.IsZero() &&
		time.Since(w.windowStart) >= time.Duration(w.maxWaitMs)*time.Millisecond

	// Reset timer
	if w.timer != nil {
		w.timer.Stop()
	}
	if !flushNow {
		w.timer = time.AfterFunc(time.Duration(w.debounceMs)*time.Millisecond, w.flush)
	}
	w.pendingMu.Unlock()

	if flushNow {
		w.flush()
	}
}

func (w *Watcher) flush() {
	w.pendingMu.Lock()
	events := make([]FileEvent, 0, len(w.pending))
	for _, event := range w.pending {
		events = append(events, event)
	}
	w.pending = make(map[string]FileEvent)
	w.windowStart = time.Time{}
	if w.timer != nil {
		w.timer.Stop()
		w.timer = nil
	}
	w.pendingMu.Unlock()

	// Blocking send: a burst larger than the channel buffer applies
	// backpressure instead of dropping files. Dropping here would leave
	// files unindexed until the next restart, which with a long debounce
	// could be far away.
	for _, event := range events {
		select {
		case w.events <- event:
		case <-w.done:
			return
		}
	}
}

func (e EventType) String() string {
	switch e {
	case EventCreate:
		return "CREATE"
	case EventModify:
		return "MODIFY"
	case EventDelete:
		return "DELETE"
	case EventRename:
		return "RENAME"
	default:
		return "UNKNOWN"
	}
}
