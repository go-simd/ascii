//go:build ppc64le

package ascii

import (
	"bytes"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDispatchPPC64LE drives ToLower, ToUpper and EqualFold down both ppc64le
// branches — the scalar fallback (each *SIMD entry returns 0) and the VSX
// kernels — by toggling hasVSX, restoring it with defer, and comparing against
// the bytes standard library. The kernels load/store blocks with
// LXVB16X/STXVB16X, ISA-3.0 (POWER9) instructions that raise SIGILL on POWER8,
// so the kernel-forcing branch runs only when the host is actually POWER9+
// (mirroring the amd64 force tests). The scalar-fallback branch is always
// exercised. The power9-targeted QEMU CI job and the native POWER9/POWER10 farm
// runs cover the kernel branch.
func TestDispatchPPC64LE(t *testing.T) {
	saved := hasVSX
	defer func() { hasVSX = saved }()

	check := func(label string) {
		// Sizes span below/at/above the 16-byte block stride so the kernel block,
		// the threshold short-circuit and the scalar tail are all exercised.
		for _, n := range []int{0, 1, 15, 16, 17, 31, 32, 100, 256, 4096} {
			src := randASCII(n, int64(n)*3+1)
			if got, want := ToLower(src), bytes.ToLower(src); !bytes.Equal(got, want) {
				t.Fatalf("%s ToLower n=%d: got %q want %q", label, n, got, want)
			}
			if got, want := ToUpper(src), bytes.ToUpper(src); !bytes.Equal(got, want) {
				t.Fatalf("%s ToUpper n=%d: got %q want %q", label, n, got, want)
			}
			// EqualFold: equal under fold (random case flips) and a mismatch.
			b := make([]byte, n)
			copy(b, src)
			for i := range b {
				if 'a' <= b[i] && b[i] <= 'z' && i%2 == 0 {
					b[i] -= 'a' - 'A'
				}
			}
			if !EqualFold(src, b) {
				t.Fatalf("%s EqualFold n=%d: case-flipped copy should fold-equal", label, n)
			}
			if n > 0 {
				b[0] ^= 0x01 // force a non-letter mismatch
				if EqualFold(src, b) != bytes.EqualFold(src, b) {
					t.Fatalf("%s EqualFold mismatch n=%d disagrees with stdlib", label, n)
				}
			}
		}
	}

	// Scalar fallback: always safe, exercised on every ppc64le host.
	hasVSX = false
	check("fallback")

	// VSX kernel: only force it on when the CPU is POWER9+, otherwise the
	// LXVB16X/STXVB16X in the kernels would SIGILL (e.g. on a POWER8 farm node).
	if !cpu.PPC64.IsPOWER9 {
		t.Log("CPU is pre-POWER9; VSX kernel branch not exercised on this host")
		return
	}
	hasVSX = true
	check("vsx")
}
