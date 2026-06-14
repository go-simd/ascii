package ascii

import (
	"bytes"
	"math/rand"
	"strings"
	"testing"
)

// sizes spans empty, sub-stride, exact-stride and stride+tail lengths so both
// the SIMD block loop and the scalar tail are exercised on every arch (16- and
// 32-byte strides).
var sizes = []int{0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 127, 128, 255, 256, 257}

func randASCII(n int, seed int64) []byte {
	r := rand.New(rand.NewSource(seed))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.Intn(0x80)) // 0x00..0x7F
	}
	return b
}

func randBytes(n int, seed int64) []byte {
	r := rand.New(rand.NewSource(seed))
	b := make([]byte, n)
	r.Read(b)
	return b
}

func TestToUpper(t *testing.T) {
	for _, n := range sizes {
		for _, in := range [][]byte{randASCII(n, int64(n)*3+1), randBytes(n, int64(n)*5+2)} {
			if got, want := ToUpper(in), bytes.ToUpper(in); !bytes.Equal(got, want) {
				t.Fatalf("ToUpper n=%d:\n got=%q\nwant=%q", n, got, want)
			}
		}
	}
}

func TestToLower(t *testing.T) {
	for _, n := range sizes {
		for _, in := range [][]byte{randASCII(n, int64(n)*7+3), randBytes(n, int64(n)*11+4)} {
			if got, want := ToLower(in), bytes.ToLower(in); !bytes.Equal(got, want) {
				t.Fatalf("ToLower n=%d:\n got=%q\nwant=%q", n, got, want)
			}
		}
	}
}

func TestAllASCIIBytes(t *testing.T) {
	// Every single ASCII byte must fold exactly like the standard library.
	all := make([]byte, 0x80)
	for i := range all {
		all[i] = byte(i)
	}
	if got, want := ToUpper(all), bytes.ToUpper(all); !bytes.Equal(got, want) {
		t.Fatalf("ToUpper all-ascii:\n got=%q\nwant=%q", got, want)
	}
	if got, want := ToLower(all), bytes.ToLower(all); !bytes.Equal(got, want) {
		t.Fatalf("ToLower all-ascii:\n got=%q\nwant=%q", got, want)
	}
}

func TestStringWrappers(t *testing.T) {
	cases := []string{"", "Hello, World!", "ALLCAPS", "nocaps", "MiXeD123", "héllo WÖRLD", strings.Repeat("aB", 200)}
	for _, s := range cases {
		if got, want := ToUpperString(s), strings.ToUpper(s); got != want {
			t.Fatalf("ToUpperString(%q)=%q want %q", s, got, want)
		}
		if got, want := ToLowerString(s), strings.ToLower(s); got != want {
			t.Fatalf("ToLowerString(%q)=%q want %q", s, got, want)
		}
	}
}

func TestEqualFold(t *testing.T) {
	for _, n := range sizes {
		base := randASCII(n, int64(n)*13+5)
		// equal under fold: flip case of a random subset
		other := append([]byte(nil), base...)
		r := rand.New(rand.NewSource(int64(n) * 17))
		for i := range other {
			if c := other[i]; r.Intn(2) == 0 {
				if 'a' <= c && c <= 'z' {
					other[i] = c - 0x20
				} else if 'A' <= c && c <= 'Z' {
					other[i] = c + 0x20
				}
			}
		}
		if got, want := EqualFold(base, other), bytes.EqualFold(base, other); got != want {
			t.Fatalf("EqualFold equal-case n=%d: got=%v want=%v", n, got, want)
		}
		// definitely-unequal: corrupt one byte (non-letter delta)
		if n > 0 {
			bad := append([]byte(nil), base...)
			idx := r.Intn(n)
			bad[idx] ^= 0x01 // toggle a low bit -> changes a non-case-bit
			if got, want := EqualFold(base, bad), bytes.EqualFold(base, bad); got != want {
				t.Fatalf("EqualFold corrupted n=%d idx=%d: got=%v want=%v", n, idx, got, want)
			}
		}
	}
}

func TestEqualFoldEdge(t *testing.T) {
	cases := []struct{ a, b string }{
		{"", ""},
		{"Go", "GO"},
		{"Go", "go"},
		{"Go", "g0"},
		{"abc", "abcd"},                          // different length
		{"abc", "ab"},                            // different length
		{"héllo", "HÉLLO"},                       // non-ascii -> delegated
		{"AbCdEfGhIjKlMnOp", "aBcDeFgHiJkLmNoP"}, // exactly 16
		{strings.Repeat("Aa", 33), strings.Repeat("aA", 33)},
	}
	for _, c := range cases {
		if got, want := EqualFold([]byte(c.a), []byte(c.b)), bytes.EqualFold([]byte(c.a), []byte(c.b)); got != want {
			t.Fatalf("EqualFold(%q,%q)=%v want %v", c.a, c.b, got, want)
		}
	}
}

func FuzzToUpper(f *testing.F) {
	f.Add([]byte("Hello"))
	f.Add([]byte("héllo WÖRLD"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, b []byte) {
		if got, want := ToUpper(b), bytes.ToUpper(b); !bytes.Equal(got, want) {
			t.Fatalf("ToUpper(%q)=%q want %q", b, got, want)
		}
	})
}

func FuzzToLower(f *testing.F) {
	f.Add([]byte("HELLO"))
	f.Add([]byte("HÉLLO wörld"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, b []byte) {
		if got, want := ToLower(b), bytes.ToLower(b); !bytes.Equal(got, want) {
			t.Fatalf("ToLower(%q)=%q want %q", b, got, want)
		}
	})
}

func FuzzEqualFold(f *testing.F) {
	f.Add([]byte("Go"), []byte("GO"))
	f.Add([]byte("héllo"), []byte("HÉLLO"))
	f.Add([]byte{}, []byte{})
	f.Fuzz(func(t *testing.T, a, b []byte) {
		if got, want := EqualFold(a, b), bytes.EqualFold(a, b); got != want {
			t.Fatalf("EqualFold(%q,%q)=%v want %v", a, b, got, want)
		}
	})
}

func BenchmarkToUpper(b *testing.B) {
	src := randASCII(4096, 1)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ToUpper(src)
	}
}

func BenchmarkToUpperStd(b *testing.B) {
	src := randASCII(4096, 1)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bytes.ToUpper(src)
	}
}

func BenchmarkToLower(b *testing.B) {
	src := randASCII(4096, 1)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ToLower(src)
	}
}

func BenchmarkEqualFold(b *testing.B) {
	a := randASCII(4096, 1)
	c := append([]byte(nil), a...)
	b.SetBytes(int64(len(a)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EqualFold(a, c)
	}
}

func BenchmarkEqualFoldStd(b *testing.B) {
	a := randASCII(4096, 1)
	c := append([]byte(nil), a...)
	b.SetBytes(int64(len(a)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bytes.EqualFold(a, c)
	}
}
