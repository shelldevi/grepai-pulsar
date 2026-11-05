package indexer

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

const (
	// defaultMaxFileSize is the built-in upper bound for indexed files.
	// It applies until the operator sets indexer.max_file_size in config.
	defaultMaxFileSize = 1 * 1024 * 1024 // 1 MB
)

// MinifiedPatterns lists patterns for minified files to skip by default
var MinifiedPatterns = []string{
	".min.js",
	".min.css",
	".bundle.js",
	".bundle.css",
}

// isMinifiedFile checks if a file is a minified file based on naming patterns
func isMinifiedFile(path string) bool {
	lowerPath := strings.ToLower(path)
	for _, pattern := range MinifiedPatterns {
		if strings.HasSuffix(lowerPath, pattern) {
			return true
		}
	}
	return false
}

// SupportedExtensions lists file extensions to index
var SupportedExtensions = map[string]bool{
	".go":     true,
	".js":     true,
	".ts":     true,
	".jsx":    true,
	".tsx":    true,
	".py":     true,
	".rb":     true,
	".java":   true,
	".c":      true,
	".cpp":    true,
	".cc":     true,
	".h":      true,
	".hpp":    true,
	".cs":     true,
	".php":    true,
	".rs":     true,
	".swift":  true,
	".kt":     true,
	".scala":  true,
	".vue":    true,
	".svelte": true,
	".html":   true,
	".css":    true,
	".scss":   true,
	".less":   true,
	".sql":    true,
	".sh":     true,
	".bash":   true,
	".zsh":    true,
	".yaml":   true,
	".yml":    true,
	".json":   true,
	".xml":    true,
	".md":     true,
	".txt":    true,
	".toml":   true,
	".ini":    true,
	".cfg":    true,
	".conf":   true,
	".env":    true,
	".lua":    true,
	".r":      true,
	".R":      true,
	".dart":   true,
	".ex":     true,
	".exs":    true,
	".erl":    true,
	".clj":    true,
	".hs":     true,
	".ml":     true,
	".fs":     true,
	".elm":    true,
	".nim":    true,
	".zig":    true,
	".proto":  true,
	".tf":     true,
	".hcl":    true,
	".pas":    true, // Pascal source file
	".dpr":    true, // Delphi project file
}

type FileInfo struct {
	Path    string
	Size    int64
	ModTime int64
	Hash    string
	Content string
}

type FileMeta struct {
	Path    string
	Size    int64
	ModTime int64
}

// maxScanErrPaths caps the failing-path list: the count is exact, the list
// is a sample for the operator (dry-run prints up to this many).
const maxScanErrPaths = 20

// ScanOutcome is the result of the metadata pass: files found, ordinary
// skips, and the walk/stat errors that mark a scan incomplete. ErrPaths
// names up to maxScanErrPaths failing paths in walk order; ErrCount is the
// exact total.
type ScanOutcome struct {
	Files    []FileMeta
	Skipped  []string
	ErrCount int
	ErrPaths []string
}

type Scanner struct {
	root        string
	ignore      *IgnoreMatcher
	maxFileSize int64
}

func NewScanner(root string, ignore *IgnoreMatcher) *Scanner {
	return &Scanner{
		root:        root,
		ignore:      ignore,
		maxFileSize: defaultMaxFileSize,
	}
}

// SetMaxFileSize overrides the built-in file size limit (in bytes).
// Values <= 0 keep the built-in default, so an unset config key changes
// nothing. Called by the CLI from indexer.max_file_size.
func (s *Scanner) SetMaxFileSize(size int64) {
	if size > 0 {
		s.maxFileSize = size
	}
}

// MaxFileSize returns the effective per-file size limit in bytes.
func (s *Scanner) MaxFileSize() int64 {
	return s.maxFileSize
}

// logSkipSize reports a file skipped for exceeding the size limit. Skips
// otherwise stay silent, so an operator-set limit needs this to be visible.
func (s *Scanner) logSkipSize(relPath string, size int64) {
	log.Printf("Skipping %s: size %d bytes exceeds max file size %d bytes", relPath, size, s.maxFileSize)
}

// ScanMetadata scans indexable files and returns only file metadata.
// It avoids reading file contents and hash computation for a faster first pass.
// ScanMetadata scans indexable files and returns only file metadata.
// It avoids reading file contents and hash computation for a faster first pass.
// Walk/stat errors are counted and up to maxScanErrPaths of them named, so
// callers can treat the result as partial: deletions based on absence must
// be skipped for such a scan. Ordinary skips (ignored, unsupported,
// minified, too large) are not errors. A file that vanished mid-scan is
// absence, not an error.
func (s *Scanner) ScanMetadata() (ScanOutcome, error) {
	out := ScanOutcome{}

	addErr := func(path string) {
		out.ErrCount++
		if len(out.ErrPaths) < maxScanErrPaths {
			if rel, rerr := filepath.Rel(s.root, path); rerr == nil {
				path = rel
			}
			out.ErrPaths = append(out.ErrPaths, path)
		}
	}

	err := filepath.WalkDir(s.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// A file that vanished mid-scan is absence, not an error:
			// agents edit constantly, and a deleted file must land in
			// the gone set (subject to the usual guards), not mark the
			// whole scan incomplete. Permission and I/O errors still
			// count: the scan did not see everything.
			if os.IsNotExist(err) {
				return nil
			}
			addErr(path)
			return nil // Skip files we can't access
		}

		relPath, err := filepath.Rel(s.root, path)
		if err != nil {
			addErr(path)
			return nil
		}

		// Handle directories: use ShouldSkipDir to respect .grepaiignore negations
		if d.IsDir() {
			if s.ignore.ShouldSkipDir(relPath) {
				return filepath.SkipDir
			}
			return nil // Descend into the directory
		}

		// Skip ignored files
		if s.ignore.ShouldIgnore(relPath) {
			return nil
		}

		// Check extension
		ext := strings.ToLower(filepath.Ext(path))
		if !SupportedExtensions[ext] {
			return nil
		}

		// Skip minified files
		if isMinifiedFile(relPath) {
			out.Skipped = append(out.Skipped, relPath+" (minified)")
			return nil
		}

		info, err := d.Info()
		if err != nil {
			// Vanished mid-scan: absence, not an error (see above).
			if os.IsNotExist(err) {
				return nil
			}
			addErr(path)
			return nil
		}

		// Skip large files
		if info.Size() > s.maxFileSize {
			s.logSkipSize(relPath, info.Size())
			out.Skipped = append(out.Skipped, relPath+" (too large)")
			return nil
		}

		out.Files = append(out.Files, FileMeta{
			Path:    relPath,
			Size:    info.Size(),
			ModTime: info.ModTime().Unix(),
		})

		return nil
	})

	return out, err
}

func (s *Scanner) Scan() ([]FileInfo, []string, error) {
	var files []FileInfo
	var skipped []string

	err := filepath.WalkDir(s.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // Skip files we can't access
		}

		relPath, err := filepath.Rel(s.root, path)
		if err != nil {
			return nil
		}

		// Handle directories: use ShouldSkipDir to respect .grepaiignore negations
		if d.IsDir() {
			if s.ignore.ShouldSkipDir(relPath) {
				return filepath.SkipDir
			}
			return nil // Descend into the directory
		}

		// Skip ignored files
		if s.ignore.ShouldIgnore(relPath) {
			return nil
		}

		// Check extension
		ext := strings.ToLower(filepath.Ext(path))
		if !SupportedExtensions[ext] {
			return nil
		}

		// Skip minified files
		if isMinifiedFile(relPath) {
			skipped = append(skipped, relPath+" (minified)")
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}

		// Skip large files
		if info.Size() > s.maxFileSize {
			s.logSkipSize(relPath, info.Size())
			skipped = append(skipped, relPath+" (too large)")
			return nil
		}

		// Read file content
		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		// Skip binary files
		if !utf8.Valid(content) || containsNull(content) {
			return nil
		}

		// Calculate hash
		hash := sha256.Sum256(content)

		files = append(files, FileInfo{
			Path:    relPath,
			Size:    info.Size(),
			ModTime: info.ModTime().Unix(),
			Hash:    hex.EncodeToString(hash[:]),
			Content: string(content),
		})

		return nil
	})

	return files, skipped, err
}

func (s *Scanner) ScanFile(relPath string) (*FileInfo, error) {
	absPath := filepath.Join(s.root, relPath)

	// Skip minified files
	if isMinifiedFile(relPath) {
		return nil, nil
	}

	info, err := os.Stat(absPath)
	if err != nil {
		// Deleted between the metadata pass and the read: absence, not
		// an error. The caller leaves the file in the gone set.
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	if info.Size() > s.maxFileSize {
		s.logSkipSize(relPath, info.Size())
		return nil, nil // Skip large file
	}

	content, err := os.ReadFile(absPath)
	if err != nil {
		// Deleted between stat and read: absence, not an error.
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	if !utf8.Valid(content) || containsNull(content) {
		return nil, nil // Skip binary files
	}

	hash := sha256.Sum256(content)

	return &FileInfo{
		Path:    relPath,
		Size:    info.Size(),
		ModTime: info.ModTime().Unix(),
		Hash:    hex.EncodeToString(hash[:]),
		Content: string(content),
	}, nil
}

func containsNull(data []byte) bool {
	for _, b := range data {
		if b == 0 {
			return true
		}
	}
	return false
}

func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
