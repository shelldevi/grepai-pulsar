package watcher

import (
	"strconv"
	"testing"
	"time"
)

func itoa(i int) string { return strconv.Itoa(i) }

func newTestWatcher(debounceMs int) *Watcher {
	w := &Watcher{
		debounceMs: debounceMs,
		events:     make(chan FileEvent, 100),
		done:       make(chan struct{}),
		pending:    make(map[string]FileEvent),
		kick:       make(chan struct{}, 1),
	}
	go w.dispatch()
	return w
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
	defer w.Close()

	for i := 0; i < 10; i++ {
		w.debounceEvent(FileEvent{Type: EventModify, Path: "a.go"})
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(300 * time.Millisecond)

	if got := drainEvents(w); len(got) != 1 {
		t.Fatalf("expected 1 coalesced event, got %d", len(got))
	}
}

// The window resets on every event: with steady activity there is exactly
// one delivery, after the last event — not one per window.
func TestDebounce_GlobalReset(t *testing.T) {
	w := newTestWatcher(120)
	defer w.Close()

	for i := 0; i < 5; i++ {
		w.debounceEvent(FileEvent{Type: EventModify, Path: "a.go"})
		time.Sleep(40 * time.Millisecond)
	}
	// 200ms of activity with a 120ms debounce: nothing may be delivered yet.
	if got := drainEvents(w); len(got) != 0 {
		t.Fatalf("expected no delivery during continuous activity, got %d", len(got))
	}
	time.Sleep(300 * time.Millisecond)
	if got := drainEvents(w); len(got) != 1 {
		t.Fatalf("expected 1 delivery after activity stops, got %d", len(got))
	}
}

// max_wait_ms bounds the hold time even under continuous activity.
func TestDebounce_MaxWaitBoundsStarvation(t *testing.T) {
	w := newTestWatcher(10000)
	w.SetMaxWaitMs(150)
	defer w.Close()

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

// The max-wait timer fires even when the burst stops before the bound:
// delivery happens at the deadline, not a full debounce later.
func TestDebounce_MaxWaitFiresAfterBurstStops(t *testing.T) {
	w := newTestWatcher(10000)
	w.SetMaxWaitMs(300)
	defer w.Close()

	start := time.Now()
	for i := 0; i < 5; i++ {
		w.debounceEvent(FileEvent{Type: EventModify, Path: "a.go"})
		time.Sleep(20 * time.Millisecond)
	}
	// Burst lasted ~100ms, then silence. The 10s debounce alone would wait
	// ~10s; the bound must deliver at ~300ms from the window start.
	var got []FileEvent
	deadline := time.Now().Add(3 * time.Second)
	for len(got) == 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
		got = drainEvents(w)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 max-wait delivery, got %d", len(got))
	}
	if dt := time.Since(start); dt < 250*time.Millisecond || dt > 2*time.Second {
		t.Fatalf("delivery at %v after burst start, want ~300ms (max wait), not ~10s (debounce)", dt)
	}
}

// A burst larger than the channel buffer must not lose files.
func TestDebounce_MassFlushLosesNothing(t *testing.T) {
	w := newTestWatcher(50)
	defer w.Close()

	const n = 150
	for i := 0; i < n; i++ {
		w.debounceEvent(FileEvent{Type: EventCreate, Path: "file" + itoa(i) + ".go"})
	}
	// The dispatcher backpressures past the channel buffer; wait for all.
	var got []FileEvent
	deadline := time.Now().Add(5 * time.Second)
	for len(got) < n && time.Now().Before(deadline) {
		select {
		case ev := <-w.events:
			got = append(got, ev)
		case <-time.After(20 * time.Millisecond):
		}
	}
	if len(got) != n {
		t.Fatalf("expected all %d events delivered, got %d", n, len(got))
	}
}
