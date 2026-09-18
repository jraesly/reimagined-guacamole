// Package media measures media model weights without estimating runtime activations.
package media

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

type Tensor struct {
	Name, DType string
	Shape       []uint64
	Bytes       uint64
}

type Header struct {
	Metadata map[string]string
	Tensors  []Tensor
}

// ReadHeader reads only the length prefix and JSON header, never tensor data.
func ReadHeader(path string) (*Header, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var prefix [8]byte
	if _, err = io.ReadFull(f, prefix[:]); err != nil {
		return nil, fmt.Errorf("%s: truncated safetensors prefix: %w", path, err)
	}
	n := binary.LittleEndian.Uint64(prefix[:])
	if n == 0 || n >= 100<<20 {
		return nil, fmt.Errorf("%s: invalid header length %d", path, n)
	}
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if uint64(st.Size()-8) < n {
		return nil, fmt.Errorf("%s: truncated safetensors header", path)
	}
	raw := make([]byte, n)
	if _, err = io.ReadFull(f, raw); err != nil {
		return nil, err
	}
	var entries map[string]json.RawMessage
	if err = json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if entries == nil {
		return nil, fmt.Errorf("%s: header must be an object", path)
	}
	h := &Header{Metadata: map[string]string{}}
	var total uint64
	for name, entry := range entries {
		if name == "__metadata__" {
			if err = json.Unmarshal(entry, &h.Metadata); err != nil {
				return nil, fmt.Errorf("%s: metadata: %w", path, err)
			}
			continue
		}
		var t struct {
			DType   string   `json:"dtype"`
			Shape   []uint64 `json:"shape"`
			Offsets []uint64 `json:"data_offsets"`
		}
		if err = json.Unmarshal(entry, &t); err != nil {
			return nil, fmt.Errorf("%s: tensor %q: %w", path, name, err)
		}
		if t.DType == "" || t.Shape == nil || len(t.Offsets) != 2 || t.Offsets[1] < t.Offsets[0] {
			return nil, fmt.Errorf("%s: tensor %q: invalid dtype, shape or offsets", path, name)
		}
		b := t.Offsets[1] - t.Offsets[0]
		if b > math.MaxUint64-total {
			return nil, fmt.Errorf("%s: tensor byte sum overflow", path)
		}
		total += b
		h.Tensors = append(h.Tensors, Tensor{name, t.DType, t.Shape, b})
	}
	sort.Slice(h.Tensors, func(i, j int) bool { return h.Tensors[i].Name < h.Tensors[j].Name })
	return h, nil
}

func (h *Header) TotalBytes() uint64 {
	var n uint64
	for _, t := range h.Tensors {
		n += t.Bytes
	}
	return n
}
func (h *Header) ByDType() map[string]uint64 {
	out := map[string]uint64{}
	for _, t := range h.Tensors {
		out[t.DType] += t.Bytes
	}
	return out
}

// ReadShardedDir merges the shards referenced by model.safetensors.index.json.
// Each shard is read once; index paths must stay within the repository.
func ReadShardedDir(dir string) (*Header, error) {
	return readIndex(dir, filepath.Join(dir, "model.safetensors.index.json"))
}

func readIndex(dir, path string) (*Header, error) {
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := readJSON(path, &index); err != nil {
		return nil, err
	}
	if len(index.WeightMap) == 0 {
		return nil, fmt.Errorf("%s: empty weight_map", path)
	}
	unique := map[string]bool{}
	for _, name := range index.WeightMap {
		if !filepath.IsLocal(name) || filepath.Ext(name) != ".safetensors" {
			return nil, fmt.Errorf("%s: invalid shard path %q", path, name)
		}
		unique[filepath.Clean(name)] = true
	}
	var paths []string
	for name := range unique {
		paths = append(paths, filepath.Join(dir, name))
	}
	sort.Strings(paths)
	return readHeaders(paths)
}

func readHeaders(paths []string) (*Header, error) {
	result := &Header{Metadata: map[string]string{}}
	var total uint64
	for _, path := range paths {
		h, err := ReadHeader(path)
		if err != nil {
			return nil, err
		}
		n := h.TotalBytes()
		if n > math.MaxUint64-total {
			return nil, fmt.Errorf("%s: shard byte sum overflow", path)
		}
		total += n
		result.Tensors = append(result.Tensors, h.Tensors...)
		for k, v := range h.Metadata {
			result.Metadata[k] = v
		}
	}
	return result, nil
}

func directoryHeader(dir string) (*Header, error) {
	indexes, err := filepath.Glob(filepath.Join(dir, "*.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	if len(indexes) > 1 {
		return nil, fmt.Errorf("%s: multiple weight indexes; choose one model variant", dir)
	}
	if len(indexes) == 1 {
		return readIndex(dir, indexes[0])
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".safetensors" {
			paths = append(paths, filepath.Join(dir, e.Name()))
		}
	}
	return readHeaders(paths)
}

func readJSON(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// Configs and indexes are bounded too; they do not contain tensor data.
	raw, err := io.ReadAll(io.LimitReader(f, 100<<20))
	if err != nil {
		return err
	}
	if len(raw) >= 100<<20 {
		return fmt.Errorf("%s: JSON exceeds size limit", path)
	}
	if err = json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}
