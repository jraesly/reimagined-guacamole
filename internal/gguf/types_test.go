package gguf

import "testing"

func TestTensorBytes(t *testing.T) {
	cases := []struct {
		name string
		typ  uint32
		dims []uint64
		want uint64
	}{
		{"f32", 0, []uint64{10, 10}, 400},
		{"f16", 1, []uint64{1024}, 2048},
		{"q8_0 exact blocks", 8, []uint64{64}, 68},
		{"q8_0 partial block rounds up", 8, []uint64{33}, 68},
		{"q4_k super-block", 12, []uint64{256, 2}, 288},
		{"q6_k", 14, []uint64{512}, 420},
		{"mxfp4", 39, []uint64{32}, 17},
	}
	for _, c := range cases {
		got, ok := TensorInfo{Dims: c.dims, Type: c.typ}.Bytes()
		if !ok || got != c.want {
			t.Errorf("%s: bytes = %d,%v want %d", c.name, got, ok, c.want)
		}
	}
	if _, ok := (TensorInfo{Dims: []uint64{1}, Type: 4}).Bytes(); ok {
		t.Error("removed type id 4 must be unknown")
	}
	// Rows pad independently: two rows of 33 elements in q8_0 are 2 × 68, not
	// ceil(66/32) × 34 = 102.
	if got, _ := (TensorInfo{Dims: []uint64{33, 2}, Type: 8}).Bytes(); got != 136 {
		t.Errorf("per-row padding: got %d, want 136", got)
	}
	huge := TensorInfo{Dims: []uint64{1 << 40, 1 << 40}, Type: 0}
	if huge.Elements() != 0 {
		t.Error("overflowing element product must report 0")
	}
	if _, ok := huge.Bytes(); ok {
		t.Error("overflowing tensor must not report bytes")
	}
	if n, ok := TypeName(12); !ok || n != "Q4_K" {
		t.Errorf("TypeName(12) = %q,%v", n, ok)
	}
}
