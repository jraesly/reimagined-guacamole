package gguf

import "math"

// ggml tensor types: elements per block and bytes per block. Block-quantized
// types store a scale (and sometimes a minimum) per block; the K-quants use
// 256-element super-blocks. Sizes follow ggml's block structs.
var typeSizes = map[uint32]struct {
	name       string
	blockElems uint64
	blockBytes uint64
}{
	0:  {"F32", 1, 4},
	1:  {"F16", 1, 2},
	2:  {"Q4_0", 32, 18},
	3:  {"Q4_1", 32, 20},
	6:  {"Q5_0", 32, 22},
	7:  {"Q5_1", 32, 24},
	8:  {"Q8_0", 32, 34},
	9:  {"Q8_1", 32, 36},
	10: {"Q2_K", 256, 84},
	11: {"Q3_K", 256, 110},
	12: {"Q4_K", 256, 144},
	13: {"Q5_K", 256, 176},
	14: {"Q6_K", 256, 210},
	15: {"Q8_K", 256, 292},
	16: {"IQ2_XXS", 256, 66},
	17: {"IQ2_XS", 256, 74},
	18: {"IQ3_XXS", 256, 98},
	19: {"IQ1_S", 256, 50},
	20: {"IQ4_NL", 32, 18},
	21: {"IQ3_S", 256, 110},
	22: {"IQ2_S", 256, 82},
	23: {"IQ4_XS", 256, 136},
	24: {"I8", 1, 1},
	25: {"I16", 1, 2},
	26: {"I32", 1, 4},
	27: {"I64", 1, 8},
	28: {"F64", 1, 8},
	29: {"IQ1_M", 256, 56},
	30: {"BF16", 1, 2},
	34: {"TQ1_0", 256, 54},
	35: {"TQ2_0", 256, 66},
	39: {"MXFP4", 32, 17},
}

// TypeName returns the ggml type name for a tensor type id.
func TypeName(t uint32) (string, bool) {
	s, ok := typeSizes[t]
	return s.name, ok
}

// Bytes is the tensor's storage size on disk, when its type is known. ggml
// quantizes along the first dimension, so each row is padded to whole
// blocks independently.
func (t TensorInfo) Bytes() (uint64, bool) {
	s, ok := typeSizes[t.Type]
	if !ok || len(t.Dims) == 0 || t.Elements() == 0 {
		return 0, false
	}
	rowBytes := (t.Dims[0] + s.blockElems - 1) / s.blockElems * s.blockBytes
	rows := uint64(1)
	for _, d := range t.Dims[1:] {
		rows *= d
	}
	if rowBytes != 0 && rows > math.MaxUint64/rowBytes {
		return 0, false
	}
	return rows * rowBytes, true
}
