package watcher

import (
	"strconv"
	"testing"
	"time"
)

func itoa(i int) string { return strconv.Itoa(i) }

func newTestWatcher(debounceMs int) *Watcher {
	return &Watcher{
		debounceMs: debounceMs,
		events:     make(chan FileEvent, 100),
		done:       make(chan struct{}),
		pending:    make(map[string]FileEvent),
	}
}

func drainEvents(w *Watcher) []FileEvent {
	var out []FileEvent
	for {
		select {
		case ev := <-w.events:
			out = append(out, ev)
		default:
			return out
		}
	}
}

// Rapid saves of one file must coalesce into a single index operation.
func TestDebounce_CoalescesRapidEvents(t *testing.T) {
	w := newTestWatcher(80)
	defer close(w.done)

	for i := 0; i < 10; i++ {
		w.debounceEvent(FileEvent{Type: EventModify, Path: "a.go"})
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(200 * time.Millisecond)

	if got := drainEvents(w); len(got) != 1 {
		t.Fatalf("expected 1 coalesced event, got %d", len(got))
	}
}

// The window resets on every event: with steady activity there is exactly
// one delivery, after the last event — not one per window.
func TestDebounce_GlobalReset(t *testing.T) {
	w := newTestWatcher(120)
	defer close(w.done)

	for i := 0; i < 5; i++ {
		w.debounceEvent(FileEvent{Type: EventModify, Path: "a.go"})
		time.Sleep(40 * time.Millisecond)
	}
	// 200ms of activity with a 120ms debounce: nothing may be delivered yet.
	if got := drainEvents(w); len(got) != 0 {
		t.Fatalf("expected no delivery during continuous activity, got %d", len(got))
	}
	time.Sleep(250 * time.Millisecond)
	if got := drainEvents(w); len(got) != 1 {
		t.Fatalf("expected 1 delivery after activity stops, got %d", len(got))
	}
}

// max_wait_ms bounds the hold time even under continuous activity.
func TestDebounce_MaxWaitBoundsStarvation(t *testing.T) {
	w := newTestWatcher(10000)
	w.SetMaxWaitMs(150)
	defer close(w.done)

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		w.debounceEvent(FileEvent{Type: EventModify, Path: "a.go"})
		time.Sleep(30 * time.Millisecond)
	}
	// A 10s debounce alone would deliver nothing during this window.
	if got := drainEvents(w); len(got) < 2 {
		t.Fatalf("expected several max-wait deliveries during continuous activity, got %d", len(got))
	}
}

// A burst larger than the channel buffer must not lose files.
func TestDebounce_MassFlushLosesNothing(t *testing.T) {
	w := newTestWatcher(100000)
	defer close(w.done)

	const n = 150
	for i := 0; i < n; i++ {
		w.debounceEvent(FileEvent{Type: EventCreate, Path: "file" + itoa(i) + ".go"})
	}
	// Distinct paths: some share a name here; count via pending instead.
	w.pendingMu.Lock()
	pending := len(w.pending)
	w.pendingMu.Unlock()

	done := make(chan []FileEvent, 1)
	go func() {
		var out []FileEvent
		for len(out) < pending {
			select {
			case ev := <-w.events:
				out = append(out, ev)
			case <-time.After(5 * time.Second):
				done <- out
				return
			}
		}
		done <- out
	}()
	w.flush()
	if got := <-done; len(got) != pending {
		t.Fatalf("expected all %d pending events delivered, got %d", pending, len(got))
	}
}
