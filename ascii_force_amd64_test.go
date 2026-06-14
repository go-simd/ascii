//go:build amd64

package ascii

import (
	"bytes"
	"testing"
)

// forceLower / forceUpper drive a chosen kernel (SSE or AVX2) directly over
// whole blocks and finish the tail with the scalar fold, so both amd64 paths are
// covered even when the runtime CPU (or Rosetta) would not dispatch to one.
func forceLower(src []byte, avx2 bool) []byte {
	dst := make([]byte, len(src))
	n := len(src)
	done := 0
	if avx2 && n >= 32 {
		b := n / 32
		lowerBlocksAVX2(dst, src, b)
		done = b * 32
	} else if n >= 16 {
		b := n / 16
		lowerBlocksSSE(dst, src, b)
		done = b * 16
	}
	for i := done; i < n; i++ {
		c := src[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst[i] = c
	}
	return dst
}

func forceUpper(src []byte, avx2 bool) []byte {
	dst := make([]byte, len(src))
	n := len(src)
	done := 0
	if avx2 && n >= 32 {
		b := n / 32
		upperBlocksAVX2(dst, src, b)
		done = b * 32
	} else if n >= 16 {
		b := n / 16
		upperBlocksSSE(dst, src, b)
		done = b * 16
	}
	for i := done; i < n; i++ {
		c := src[i]
		if 'a' <= c && c <= 'z' {
			c -= 'a' - 'A'
		}
		dst[i] = c
	}
	return dst
}

func forceFold(a, b []byte, avx2 bool) bool {
	n := len(a)
	done := 0
	if avx2 && n >= 32 {
		blocks := n / 32
		if equalFoldBlocksAVX2(a, b, blocks) == 0 {
			return false
		}
		done = blocks * 32
	} else if n >= 16 {
		blocks := n / 16
		if equalFoldBlocksSSE(a, b, blocks) == 0 {
			return false
		}
		done = blocks * 16
	}
	for i := done; i < n; i++ {
		if asciiLower(a[i]) != asciiLower(b[i]) {
			return false
		}
	}
	return true
}

// TestForceKernelsAMD64 validates the SSE and AVX2 kernels against the standard
// library. AVX2 is only exercised when the CPU supports it (the instructions
// would #UD otherwise).
func TestForceKernelsAMD64(t *testing.T) {
	for _, avx2 := range []bool{false, true} {
		if avx2 && !hasAVX2 {
			continue
		}
		for _, n := range sizes {
			src := randASCII(n, int64(n)*23+7)
			if got, want := forceLower(src, avx2), bytes.ToLower(src); !bytes.Equal(got, want) {
				t.Fatalf("lower avx2=%v n=%d:\n got=%q\nwant=%q", avx2, n, got, want)
			}
			if got, want := forceUpper(src, avx2), bytes.ToUpper(src); !bytes.Equal(got, want) {
				t.Fatalf("upper avx2=%v n=%d:\n got=%q\nwant=%q", avx2, n, got, want)
			}
			// EqualFold: case-flipped equal, plus a corrupted unequal.
			flip := append([]byte(nil), src...)
			for i := range flip {
				if 'a' <= flip[i] && flip[i] <= 'z' {
					flip[i] -= 0x20
				} else if 'A' <= flip[i] && flip[i] <= 'Z' {
					flip[i] += 0x20
				}
			}
			if forceFold(src, flip, avx2) != bytes.EqualFold(src, flip) {
				t.Fatalf("fold-equal avx2=%v n=%d", avx2, n)
			}
			if n > 0 {
				bad := append([]byte(nil), src...)
				bad[n/2] ^= 0x01
				if forceFold(src, bad, avx2) != bytes.EqualFold(src, bad) {
					t.Fatalf("fold-unequal avx2=%v n=%d", avx2, n)
				}
			}
		}
	}
}

// TestDispatchBranchesAMD64 drives every branch of the amd64 dispatchers through
// the public API. On a native AVX2 box hasAVX2 is true, so the SSE and
// scalar-only branches are reached by forcing hasAVX2 low (restored via defer).
func TestDispatchBranchesAMD64(t *testing.T) {
	check := func(n int) {
		src := randASCII(n, int64(n)*29+11)
		if got, want := ToUpper(src), bytes.ToUpper(src); !bytes.Equal(got, want) {
			t.Fatalf("ToUpper hasAVX2=%v n=%d", hasAVX2, n)
		}
		if got, want := ToLower(src), bytes.ToLower(src); !bytes.Equal(got, want) {
			t.Fatalf("ToLower hasAVX2=%v n=%d", hasAVX2, n)
		}
		flip := bytes.ToUpper(src)
		if EqualFold(src, flip) != bytes.EqualFold(src, flip) {
			t.Fatalf("EqualFold hasAVX2=%v n=%d", hasAVX2, n)
		}
		if n > 0 {
			bad := append([]byte(nil), src...)
			bad[n/2] ^= 0x01
			if EqualFold(src, bad) != bytes.EqualFold(src, bad) {
				t.Fatalf("EqualFold-bad hasAVX2=%v n=%d", hasAVX2, n)
			}
		}
	}
	ns := []int{0, 8, 15, 16, 31, 32, 33, 63, 64, 65, 128}
	for _, n := range ns { // real CPU flag: AVX2 path when available
		check(n)
	}
	saved := hasAVX2
	defer func() { hasAVX2 = saved }()
	hasAVX2 = false // SSE path + scalar-only return
	for _, n := range ns {
		check(n)
	}
}
