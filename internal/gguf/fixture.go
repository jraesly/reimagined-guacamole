package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"sort"
)

// Fixture describes a synthetic GGUF header for tests. Only the header is
// produced; tensor data is omitted, which is exactly what ReadHeader needs.
type Fixture struct {
	Version  uint32
	Metadata map[string]any
	Tensors  []TensorInfo
}

// Bytes encodes the fixture as a GGUF header. Metadata keys are written in
// sorted order so output is deterministic. Supported value types: uint8,
// uint16, uint32, uint64, int32, int64, float32, float64, bool, string, and
// slices of those (encoded as arrays).
func (f Fixture) Bytes() []byte {
	var b bytes.Buffer
	w := writer{&b}
	version := f.Version
	if version == 0 {
		version = 3
	}
	w.u32(magic)
	w.u32(version)
	w.u64(uint64(len(f.Tensors)))
	w.u64(uint64(len(f.Metadata)))
	keys := make([]string, 0, len(f.Metadata))
	for k := range f.Metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		w.str(k)
		w.value(f.Metadata[k])
	}
	for _, t := range f.Tensors {
		w.str(t.Name)
		w.u32(uint32(len(t.Dims)))
		for _, d := range t.Dims {
			w.u64(d)
		}
		w.u32(t.Type)
		w.u64(0)
	}
	return b.Bytes()
}

type writer struct{ b *bytes.Buffer }

func (w writer) u8(v uint8)   { w.b.WriteByte(v) }
func (w writer) u16(v uint16) { _ = binary.Write(w.b, binary.LittleEndian, v) }
func (w writer) u32(v uint32) { _ = binary.Write(w.b, binary.LittleEndian, v) }
func (w writer) u64(v uint64) { _ = binary.Write(w.b, binary.LittleEndian, v) }
func (w writer) str(s string) {
	w.u64(uint64(len(s)))
	w.b.WriteString(s)
}

func (w writer) value(v any) {
	switch x := v.(type) {
	case uint8:
		w.u32(uint32(TypeUint8))
		w.u8(x)
	case uint16:
		w.u32(uint32(TypeUint16))
		w.u16(x)
	case uint32:
		w.u32(uint32(TypeUint32))
		w.u32(x)
	case int32:
		w.u32(uint32(TypeInt32))
		w.u32(uint32(x))
	case uint64:
		w.u32(uint32(TypeUint64))
		w.u64(x)
	case int64:
		w.u32(uint32(TypeInt64))
		w.u64(uint64(x))
	case float32:
		w.u32(uint32(TypeFloat32))
		w.u32(math.Float32bits(x))
	case float64:
		w.u32(uint32(TypeFloat64))
		w.u64(math.Float64bits(x))
	case bool:
		w.u32(uint32(TypeBool))
		if x {
			w.u8(1)
		} else {
			w.u8(0)
		}
	case string:
		w.u32(uint32(TypeString))
		w.str(x)
	case []uint32:
		w.u32(uint32(TypeArray))
		w.u32(uint32(TypeUint32))
		w.u64(uint64(len(x)))
		for _, e := range x {
			w.u32(e)
		}
	case []string:
		w.u32(uint32(TypeArray))
		w.u32(uint32(TypeString))
		w.u64(uint64(len(x)))
		for _, e := range x {
			w.str(e)
		}
	default:
		panic("gguf fixture: unsupported value type")
	}
}
