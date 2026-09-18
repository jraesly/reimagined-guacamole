// Package gguf reads the header of a GGUF file: magic, version, tensor
// descriptors and metadata key/values. It never reads tensor data, so a
// multi-gigabyte model costs a few kilobytes of I/O.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

const magic = 0x46554747 // "GGUF" little-endian

// Type is a GGUF metadata value type.
type Type uint32

const (
	TypeUint8 Type = iota
	TypeInt8
	TypeUint16
	TypeInt16
	TypeUint32
	TypeInt32
	TypeFloat32
	TypeBool
	TypeString
	TypeArray
	TypeUint64
	TypeInt64
	TypeFloat64
)

// Limits that a well-formed model file never approaches; they bound memory
// use when a corrupt or hostile file declares absurd counts.
const (
	maxString      = 1 << 20
	maxArrayLen    = 1 << 24
	maxTensorCount = 1 << 20
	maxKVCount     = 1 << 16
	maxDims        = 8
)

// TensorInfo describes one tensor from the header.
type TensorInfo struct {
	Name string
	Dims []uint64
	Type uint32
}

// Elements is the number of scalar elements in the tensor.
func (t TensorInfo) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// Header is the parsed GGUF header.
type Header struct {
	Version     uint32
	TensorCount uint64
	Metadata    map[string]any
	Tensors     []TensorInfo
}

// ReadHeader parses the header of the GGUF file at path.
func ReadHeader(path string) (*Header, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h, err := Parse(bufio.NewReaderSize(f, 1<<16))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return h, nil
}

// Parse reads a GGUF header from r.
func Parse(r io.Reader) (*Header, error) {
	p := &parser{r: r}
	if m := p.u32(); m != magic {
		if p.err != nil {
			return nil, p.err
		}
		return nil, fmt.Errorf("not a GGUF file (magic %#x)", m)
	}
	h := &Header{Version: p.u32()}
	if h.Version != 2 && h.Version != 3 {
		return nil, fmt.Errorf("unsupported GGUF version %d", h.Version)
	}
	h.TensorCount = p.u64()
	kvCount := p.u64()
	if p.err != nil {
		return nil, p.err
	}
	if h.TensorCount > maxTensorCount || kvCount > maxKVCount {
		return nil, fmt.Errorf("implausible counts: tensors=%d kv=%d", h.TensorCount, kvCount)
	}
	h.Metadata = make(map[string]any, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key := p.str()
		typ := Type(p.u32())
		val := p.value(typ)
		if p.err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, p.err)
		}
		h.Metadata[key] = val
	}
	h.Tensors = make([]TensorInfo, 0, h.TensorCount)
	for i := uint64(0); i < h.TensorCount; i++ {
		t := TensorInfo{Name: p.str()}
		nd := p.u32()
		if p.err != nil {
			return nil, p.err
		}
		if nd > maxDims {
			return nil, fmt.Errorf("tensor %q: %d dimensions", t.Name, nd)
		}
		t.Dims = make([]uint64, nd)
		for d := range t.Dims {
			t.Dims[d] = p.u64()
		}
		t.Type = p.u32()
		p.u64() // data offset, unused
		if p.err != nil {
			return nil, fmt.Errorf("tensor %q: %w", t.Name, p.err)
		}
		h.Tensors = append(h.Tensors, t)
	}
	return h, nil
}

// String returns a string metadata value.
func (h *Header) String(key string) (string, bool) {
	s, ok := h.Metadata[key].(string)
	return s, ok
}

// Uint returns an integer metadata value of any width as uint64.
func (h *Header) Uint(key string) (uint64, bool) {
	return toUint(h.Metadata[key])
}

// Float returns a float metadata value.
func (h *Header) Float(key string) (float64, bool) {
	switch v := h.Metadata[key].(type) {
	case float32:
		return float64(v), true
	case float64:
		return v, true
	}
	return 0, false
}

// UintSlice returns an integer array value. A scalar integer is returned as
// a one-element slice so callers can treat per-layer and global keys alike.
func (h *Header) UintSlice(key string) ([]uint64, bool) {
	v, ok := h.Metadata[key]
	if !ok {
		return nil, false
	}
	if arr, ok := v.([]any); ok {
		out := make([]uint64, 0, len(arr))
		for _, e := range arr {
			u, ok := toUint(e)
			if !ok {
				return nil, false
			}
			out = append(out, u)
		}
		return out, true
	}
	if u, ok := toUint(v); ok {
		return []uint64{u}, true
	}
	return nil, false
}

// Arch returns general.architecture.
func (h *Header) Arch() string {
	s, _ := h.String("general.architecture")
	return s
}

func toUint(v any) (uint64, bool) {
	switch x := v.(type) {
	case uint8:
		return uint64(x), true
	case int8:
		return uint64(x), x >= 0
	case uint16:
		return uint64(x), true
	case int16:
		return uint64(x), x >= 0
	case uint32:
		return uint64(x), true
	case int32:
		return uint64(x), x >= 0
	case uint64:
		return x, true
	case int64:
		return uint64(x), x >= 0
	}
	return 0, false
}

type parser struct {
	r   io.Reader
	err error
}

func (p *parser) read(buf []byte) {
	if p.err != nil {
		return
	}
	if _, err := io.ReadFull(p.r, buf); err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
			p.err = errors.New("truncated header")
			return
		}
		p.err = err
	}
}

func (p *parser) u8() uint8 {
	var b [1]byte
	p.read(b[:])
	return b[0]
}

func (p *parser) u16() uint16 {
	var b [2]byte
	p.read(b[:])
	return binary.LittleEndian.Uint16(b[:])
}

func (p *parser) u32() uint32 {
	var b [4]byte
	p.read(b[:])
	return binary.LittleEndian.Uint32(b[:])
}

func (p *parser) u64() uint64 {
	var b [8]byte
	p.read(b[:])
	return binary.LittleEndian.Uint64(b[:])
}

func (p *parser) str() string {
	n := p.u64()
	if p.err != nil {
		return ""
	}
	if n > maxString {
		p.err = fmt.Errorf("string length %d exceeds limit", n)
		return ""
	}
	b := make([]byte, n)
	p.read(b)
	return string(b)
}

func (p *parser) value(t Type) any {
	switch t {
	case TypeUint8:
		return p.u8()
	case TypeInt8:
		return int8(p.u8())
	case TypeUint16:
		return p.u16()
	case TypeInt16:
		return int16(p.u16())
	case TypeUint32:
		return p.u32()
	case TypeInt32:
		return int32(p.u32())
	case TypeFloat32:
		return math.Float32frombits(p.u32())
	case TypeBool:
		return p.u8() != 0
	case TypeString:
		return p.str()
	case TypeUint64:
		return p.u64()
	case TypeInt64:
		return int64(p.u64())
	case TypeFloat64:
		return math.Float64frombits(p.u64())
	case TypeArray:
		et := Type(p.u32())
		n := p.u64()
		if p.err != nil {
			return nil
		}
		if n > maxArrayLen {
			p.err = fmt.Errorf("array length %d exceeds limit", n)
			return nil
		}
		arr := make([]any, 0, min(n, 1024))
		for i := uint64(0); i < n && p.err == nil; i++ {
			arr = append(arr, p.value(et))
		}
		return arr
	}
	p.err = fmt.Errorf("unknown value type %d", t)
	return nil
}
