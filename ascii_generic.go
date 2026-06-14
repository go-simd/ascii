//go:build !amd64 && !arm64 && !riscv64 && !loong64 && !ppc64le && !s390x

package ascii

// On architectures without a SIMD kernel the scalar tails in ascii.go do all the
// work: each "SIMD" entry point reports that it processed nothing, so the
// caller's loop folds the entire slice.

func lowerSIMD(dst, src []byte) int { return 0 }
func upperSIMD(dst, src []byte) int { return 0 }

// equalFoldSIMD returns 0 (no whole blocks consumed), never a mismatch, so the
// scalar tail compares every byte.
func equalFoldSIMD(a, b []byte) int { return 0 }
