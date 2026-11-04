// Offline missing-chunk estimator. Walks a tree with the exact scanner +
// chunker code paths (same ignore rules, same size/binary/minified skips),
// WITHOUT any store, Qdrant, embedding or writes. Used to rank directories
// and extensions by chunk count and estimated tokens before tuning ignore.
//
// Usage:
//   IGNORE_JSON='["**/.git/**",...]' CHUNK_SIZE=3072 CHUNK_OVERLAP=192 \
//     go run ./offline-estimate /path/to/root
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/yoanbernabeu/grepai/embedder"
	"github.com/yoanbernabeu/grepai/indexer"
)

type agg struct {
	files  int
	chunks int
	tokens int
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: offline-estimate <root>")
		os.Exit(2)
	}
	root := os.Args[1]

	var ignore []string
	if err := json.Unmarshal([]byte(os.Getenv("IGNORE_JSON")), &ignore); err != nil {
		fmt.Fprintln(os.Stderr, "bad IGNORE_JSON:", err)
		os.Exit(2)
	}
	size, _ := strconv.Atoi(os.Getenv("CHUNK_SIZE"))
	overlap, _ := strconv.Atoi(os.Getenv("CHUNK_OVERLAP"))

	matcher, err := indexer.NewIgnoreMatcher(root, ignore, "")
	if err != nil {
		fmt.Fprintln(os.Stderr, "ignore matcher:", err)
		os.Exit(2)
	}
	scanner := indexer.NewScanner(root, matcher)
	chunker := indexer.NewChunker(size, overlap)

	metas, skipped, err := scanner.ScanMetadata()
	if err != nil {
		fmt.Fprintln(os.Stderr, "scan:", err)
		os.Exit(2)
	}

	byDir := map[string]*agg{}
	byExt := map[string]*agg{}
	junk := map[string]*agg{}
	bySub := map[string]*agg{}
	type fileRow struct {
		path   string
		chunks int
		tokens int
	}
	var topFiles []fileRow
	totalFiles, totalChunks, totalTokens := 0, 0, 0
	var tooLarge int

	for _, meta := range metas {
		fi, err := scanner.ScanFile(meta.Path)
		if err != nil || fi == nil {
			continue
		}
		infos := chunker.ChunkWithContext(fi.Path, fi.Content)
		if len(infos) == 0 {
			continue
		}
		toks := 0
		for _, ci := range infos {
			toks += embedder.EstimateTokens(ci.Content)
		}
		totalFiles++
		totalChunks += len(infos)
		totalTokens += toks

		top := strings.SplitN(meta.Path, string(filepath.Separator), 2)[0]
		if _, ok := byDir[top]; !ok {
			byDir[top] = &agg{}
		}
		byDir[top].files++
		byDir[top].chunks += len(infos)
		byDir[top].tokens += toks

		ext := strings.ToLower(filepath.Ext(meta.Path))
		if ext == "" {
			ext = "(none)"
		}
		if _, ok := byExt[ext]; !ok {
			byExt[ext] = &agg{}
		}
		byExt[ext].files++
		byExt[ext].chunks += len(infos)
		byExt[ext].tokens += toks

		base := filepath.Base(meta.Path)
		var kind string
		lowerPath := strings.ToLower(meta.Path)
		switch {
		case strings.Contains(base, "?"):
			kind = "query-string names (?/&/=)"
		case strings.HasSuffix(base, ".bak") || strings.Contains(base, ".bak."):
			kind = "*.bak* backups"
		case strings.Contains(lowerPath, ".bak-") || strings.Contains(lowerPath, ".bak/") || strings.Contains(lowerPath, "/backup") || strings.Contains(lowerPath, "backup/"):
			kind = "backup dirs (.bak-*, backup*)"
		case strings.Contains(lowerPath, "__macosx"):
			kind = "__MACOSX doubles"
		case strings.Contains(lowerPath, "/temp") || strings.Contains(lowerPath, "/tmp/") || strings.HasPrefix(base, "temp") || strings.HasPrefix(base, "tmp"):
			kind = "temp/tmp files"
		case strings.Contains(lowerPath, ".ai-tasks/") || strings.Contains(lowerPath, ".worktrees/"):
			kind = "agent worktrees/runs"
		}
		if kind != "" {
			if _, ok := junk[kind]; !ok {
				junk[kind] = &agg{}
			}
			junk[kind].files++
			junk[kind].chunks += len(infos)
			junk[kind].tokens += toks
		}

		// Level-2 breakdown for the dominant top dir.
		if top == "pkgs-private" {
			rest := strings.SplitN(meta.Path, string(filepath.Separator), 3)
			l2 := top
			if len(rest) == 3 {
				l2 = top + "/" + strings.SplitN(rest[1]+"/"+rest[2], string(filepath.Separator), 2)[0]
			} else if len(rest) == 2 {
				l2 = top + "/" + rest[1]
			}
			if _, ok := bySub[l2]; !ok {
				bySub[l2] = &agg{}
			}
			bySub[l2].files++
			bySub[l2].chunks += len(infos)
			bySub[l2].tokens += toks
		}

		// Top individual files.
		topFiles = append(topFiles, fileRow{meta.Path, len(infos), toks})
	}

	fmt.Printf("ROOT %s\nFILES_INDEXABLE %d CHUNKS %d EST_TOKENS %d (~%.1fM)\nSKIPPED_ENTRIES %d\n",
		root, totalFiles, totalChunks, totalTokens, float64(totalTokens)/1e6, len(skipped))

	printTop := func(title string, m map[string]*agg) {
		type row struct {
			k string
			a *agg
		}
		var rows []row
		for k, a := range m {
			rows = append(rows, row{k, a})
		}
		sort.Slice(rows, func(i, j int) bool { return rows[i].a.tokens > rows[j].a.tokens })
		fmt.Printf("\n== %s (by est tokens) ==\n", title)
		n := 25
		if len(rows) < n {
			n = len(rows)
		}
		for _, r := range rows[:n] {
			fmt.Printf("%-50s files=%6d chunks=%7d tokens~=%9d\n", r.k, r.a.files, r.a.chunks, r.a.tokens)
		}
	}
	printTop("TOP DIRS", byDir)
	printTop("PKGS-PRIVATE L2", bySub)
	printTop("EXTENSIONS", byExt)
	printTop("JUNK CLUSTERS", junk)

	sort.Slice(topFiles, func(i, j int) bool { return topFiles[i].tokens > topFiles[j].tokens })
	fmt.Printf("\n== TOP 30 FILES (by est tokens) ==\n")
	n := 30
	if len(topFiles) < n {
		n = len(topFiles)
	}
	for _, r := range topFiles[:n] {
		fmt.Printf("%-70s chunks=%5d tokens~=%8d\n", r.path, r.chunks, r.tokens)
	}
	_ = tooLarge
}
