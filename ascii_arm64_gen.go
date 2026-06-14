//go:build ignore

// Command gen produces ascii_arm64.s with go-asmgen: NEON kernels (16-byte
// stride) for ASCII case folding.
//
// The Go arm64 assembler exposes only VCMEQ/VCMTST for NEON vector compares (no
// unsigned VCMHS), so the letter-range test is built from per-byte subtraction
// and the sign bit instead of a compare — multiply-free, table-free, and using
// only VSUB/VAND/VEOR/VUSHR/VADD. For ToLower (range 'A'..'Z'):
//
//	lo   = (b - 'A')      & 0x80   // 0x80 iff b <  'A'
//	hi   = (b - ('Z'+1))  & 0x80   // 0x80 iff b <= 'Z'
//	inr  = hi & (lo ^ 0x80)        // 0x80 iff 'A' <= b <= 'Z'
//	out  = b + (inr >> 2)          // >>2 turns 0x80 into 0x20
//
// ToUpper is symmetric over 'a'..'z' with a subtraction. (All inputs are < 0x80,
// guaranteed by the wrapper, so the byte subtraction's sign bit is exactly the
// range predicate.) EqualFold folds both inputs to lower case the same way and
// ORs the per-byte XOR of the two folded blocks; a zero accumulator means equal.
//
// Run: GOWORK=off go run ascii_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func repByte(x byte, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = x
	}
	return b
}

func caseSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("src"), abi.Scalar("n", abi.Int64)},
		nil,
	)
}

func foldSig() abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b"), abi.Scalar("n", abi.Int64)},
		[]abi.Arg{abi.Scalar("ret", abi.Int64)},
	)
}

func main() {
	f := emit.NewFile("arm64")

	loA := f.Data("loA", repByte('A', 16))   // 0x41
	hiZ := f.Data("hiZ", repByte('Z'+1, 16)) // 0x5B
	loa := f.Data("loa", repByte('a', 16))   // 0x61
	hiz := f.Data("hiz", repByte('z'+1, 16)) // 0x7B
	c80 := f.Data("c80", repByte(0x80, 16))  // sign bit

	// loadConsts: V20=lo, V21=hi, V22=0x80.
	loadConsts := func(b *arm64.Builder, lo, hi string) *arm64.Builder {
		return b.
			Raw("MOVD $%s+0(SB), R5", lo).Raw("VLD1 (R5), [V20.B16]").
			Raw("MOVD $%s+0(SB), R5", hi).Raw("VLD1 (R5), [V21.B16]").
			Raw("MOVD $%s+0(SB), R5", c80).Raw("VLD1 (R5), [V22.B16]")
	}

	// rangeDelta: from src bytes in Vsrc, compute the 0x20 delta vector into Vdst
	// using V20=lo, V21=hi, V22=0x80; scratch V1.
	rangeDelta := func(b *arm64.Builder, Vsrc, Vdst string) *arm64.Builder {
		return b.
			Raw("VSUB V20.B16, %s.B16, V1.B16", Vsrc).       // b - lo
			Raw("VAND V22.B16, V1.B16, V1.B16").             // (b-lo) & 0x80
			Raw("VEOR V22.B16, V1.B16, V1.B16").             // ^0x80 -> 0x80 iff b>=lo
			Raw("VSUB V21.B16, %s.B16, %s.B16", Vsrc, Vdst). // b - hi
			Raw("VAND V22.B16, %s.B16, %s.B16", Vdst, Vdst). // (b-hi)&0x80 -> 0x80 iff b<=hi-1
			Raw("VAND V1.B16, %s.B16, %s.B16", Vdst, Vdst).  // inr = 0x80 iff in range
			Raw("VUSHR $2, %s.B16, %s.B16", Vdst, Vdst)      // 0x80>>2 = 0x20
	}

	caseFunc := func(name, lo, hi string, add bool) *emit.Function {
		op := "VSUB"
		if add {
			op = "VADD"
		}
		g := arm64.NewFunc(name, caseSig(), 0)
		g.LoadArg("dst_base", "R0").LoadArg("src_base", "R1").LoadArg("n", "R2")
		loadConsts(g, lo, hi)
		g.Raw("CBZ R2, done").
			Label("loop").
			Raw("VLD1 (R1), [V0.B16]")
		rangeDelta(g, "V0", "V2") // V2 = delta
		g.Raw("%s V2.B16, V0.B16, V0.B16", op).
			Raw("VST1 [V0.B16], (R0)").
			Raw("ADD $16, R1, R1").Raw("ADD $16, R0, R0").
			Raw("SUB $1, R2, R2").Raw("CBNZ R2, loop").
			Label("done").Ret()
		return g.Func()
	}
	f.Add(caseFunc("lowerBlocks", loA, hiZ, true))
	f.Add(caseFunc("upperBlocks", loa, hiz, false))

	// foldLower folds Vsrc to lower in place (delta into V2, scratch V1).
	foldLower := func(b *arm64.Builder, Vsrc string) *arm64.Builder {
		rangeDelta(b, Vsrc, "V2")
		return b.Raw("VADD V2.B16, %s.B16, %s.B16", Vsrc, Vsrc)
	}
	ef := arm64.NewFunc("equalFoldBlocks", foldSig(), 0)
	ef.LoadArg("a_base", "R0").LoadArg("b_base", "R1").LoadArg("n", "R2")
	loadConsts(ef, loA, hiZ)
	ef.Raw("VEOR V23.B16, V23.B16, V23.B16"). // accumulated diff
							Raw("CBZ R2, fdone").
							Label("floop").
							Raw("VLD1 (R0), [V3.B16]")
	foldLower(ef, "V3")
	ef.Raw("VLD1 (R1), [V4.B16]")
	foldLower(ef, "V4")
	ef.Raw("VEOR V4.B16, V3.B16, V3.B16").
		Raw("VORR V3.B16, V23.B16, V23.B16").
		Raw("ADD $16, R0, R0").Raw("ADD $16, R1, R1").
		Raw("SUB $1, R2, R2").Raw("CBNZ R2, floop").
		Label("fdone").
		Raw("VMOV V23.D[0], R3").
		Raw("VMOV V23.D[1], R4").
		Raw("ORR R4, R3, R3").
		Raw("MOVD $0, R5").
		Raw("CBNZ R3, fret").
		Raw("MOVD $1, R5").
		Label("fret").
		StoreRet("R5", "ret").Ret()
	f.Add(ef.Func())

	if err := os.WriteFile("ascii_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote ascii_arm64.s")
}
