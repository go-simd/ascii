//go:build ignore

// Command gen produces ascii_ppc64le.s with go-asmgen: VSX/AltiVec kernels
// (16-byte stride) for ASCII case folding on ppc64le (POWER8+).
//
// VSX is the ppc64le baseline, so no runtime feature check is needed. The
// letter-range test uses the multiply-free sign-bit method (per-byte subtract;
// the result's top bit is the predicate because every input byte is < 0x80).
// For ToLower (range 'A'..'Z'):
//
//	lo   = (b - 'A')      & 0x80   // 0x80 iff b <  'A'   (VSUBUBM, VAND)
//	hi   = (b - ('Z'+1))  & 0x80   // 0x80 iff b <= 'Z'
//	inr  = hi & (lo ^ 0x80)        // 0x80 iff 'A' <= b <= 'Z'  (VXOR, VAND)
//	out  = b + (inr >> 2)          // VSRB by 2: 0x80 -> 0x20  (VADDUBM)
//
// ToUpper is symmetric over 'a'..'z' with VSUBUBM. Constants are 16-byte splats
// loaded with LXVB16X (which, with the VSX register aliasing VS(32+k) == Vk,
// lands in the VMX register the byte ops use). EqualFold folds both inputs to
// lower case, ORs the per-byte XOR of the two folded blocks, then moves the two
// 64-bit halves to GPRs (MFVSRD/MFVSRLD) and ORs them — zero means equal.
//
// Endianness: LXVB16X/STXVB16X move bytes in natural memory order, and the fold
// is per-byte, so the kernel is endian-neutral; the position-sensitive Fuzz* and
// table tests are the gate. Run: GOWORK=off go run ascii_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
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
	f := emit.NewFile("ppc64le")

	loA := f.Data("loA_p", repByte('A', 16))
	hiZ := f.Data("hiZ_p", repByte('Z'+1, 16))
	loa := f.Data("loa_p", repByte('a', 16))
	hiz := f.Data("hiz_p", repByte('z'+1, 16))
	c80 := f.Data("c80_p", repByte(0x80, 16))
	c02 := f.Data("c02_p", repByte(2, 16))

	// loadConsts: V20=lo, V21=hi, V22=0x80, V23=2. (VS52..VS55 alias V20..V23.)
	loadConsts := func(b *ppc64.Builder, lo, hi string) *ppc64.Builder {
		return b.
			Raw("MOVD $%s+0(SB), R8", lo).Raw("LXVB16X (R0)(R8), VS52").  // V20
			Raw("MOVD $%s+0(SB), R8", hi).Raw("LXVB16X (R0)(R8), VS53").  // V21
			Raw("MOVD $%s+0(SB), R8", c80).Raw("LXVB16X (R0)(R8), VS54"). // V22
			Raw("MOVD $%s+0(SB), R8", c02).Raw("LXVB16X (R0)(R8), VS55")  // V23
	}

	// rangeDelta: src bytes in Vsrc -> 0x20 delta into Vdst, scratch V1.
	rangeDelta := func(b *ppc64.Builder, Vsrc, Vdst string) *ppc64.Builder {
		// VSUBUBM A, B, T computes T = A - B, so the byte source goes first.
		return b.
			Raw("VSUBUBM %s, V20, V1", Vsrc).       // b - lo
			Raw("VAND V1, V22, V1").                // & 0x80
			Raw("VXOR V1, V22, V1").                // ^0x80 -> 0x80 iff b>=lo
			Raw("VSUBUBM %s, V21, %s", Vsrc, Vdst). // b - hi
			Raw("VAND %s, V22, %s", Vdst, Vdst).    // & 0x80 -> 0x80 iff b<=hi-1
			Raw("VAND %s, V1, %s", Vdst, Vdst).     // inr
			Raw("VSRB %s, V23, %s", Vdst, Vdst)     // inr>>2 -> 0x20 (VSRB data,count,dst)
	}

	mkCase := func(name, lo, hi string, add bool) *emit.Function {
		// VADDUBM A,B,T -> T=A+B (commutative); VSUBUBM A,B,T -> T=A-B, so for
		// ToUpper the byte (V0) must be the minuend: V0 = V0 - delta.
		applyOp := "VSUBUBM V0, V2, V0"
		if add {
			applyOp = "VADDUBM V2, V0, V0"
		}
		g := ppc64.NewFunc(name, caseSig(), 0)
		g.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("n", "R5")
		loadConsts(g, lo, hi)
		g.Raw("CMP R5, $0").Raw("BEQ done").
			Raw("MOVD $0, R6"). // src/dst byte offset
			Label("loop").
			Raw("LXVB16X (R6)(R4), VS32") // V0 = 16 src bytes
		rangeDelta(g, "V0", "V2")
		g.Raw(applyOp). // out = b +/- delta
				Raw("STXVB16X VS32, (R6)(R3)").
				Raw("ADD $16, R6").
				Raw("ADD $-1, R5").
				Raw("CMP R5, $0").Raw("BNE loop").
				Label("done").Ret()
		return g.Func()
	}
	f.Add(mkCase("lowerBlocks", loA, hiZ, true))
	f.Add(mkCase("upperBlocks", loa, hiz, false))

	foldLower := func(b *ppc64.Builder, Vsrc string) *ppc64.Builder {
		rangeDelta(b, Vsrc, "V2")
		return b.Raw("VADDUBM V2, %s, %s", Vsrc, Vsrc)
	}
	ef := ppc64.NewFunc("equalFoldBlocks", foldSig(), 0)
	ef.LoadArg("a_base", "R3").LoadArg("b_base", "R4").LoadArg("n", "R5")
	loadConsts(ef, loA, hiZ)
	ef.Raw("VXOR V24, V24, V24"). // V24 = accumulator (VS56)
					Raw("CMP R5, $0").Raw("BEQ fdone").
					Raw("MOVD $0, R6").
					Label("floop").
					Raw("LXVB16X (R6)(R3), VS35") // V3 = a bytes
	foldLower(ef, "V3")
	ef.Raw("LXVB16X (R6)(R4), VS36") // V4 = b bytes
	foldLower(ef, "V4")
	ef.Raw("VXOR V3, V4, V3"). // diff
					Raw("VOR V3, V24, V24"). // accumulate
					Raw("ADD $16, R6").
					Raw("ADD $-1, R5").
					Raw("CMP R5, $0").Raw("BNE floop").
					Label("fdone").
		// Reduce the 16-byte accumulator to a scalar: OR its two doublewords.
		// MFVSRD reads one doubleword directly; the other is rotated in with
		// VSLDOI $8 (the matchlen-proven extraction, avoiding MFVSRLD).
		Raw("MFVSRD VS56, R7"). // first doubleword of V24
		Raw("VSLDOI $8, V24, V24, V25").
		Raw("MFVSRD VS57, R8"). // second doubleword (V25 == VS57)
		Raw("OR R7, R8, R7").
		Raw("MOVD $1, R9").
		Raw("CMP R7, $0").Raw("BEQ fret"). // no diff -> 1
		Raw("MOVD $0, R9").
		Label("fret").
		StoreRet("R9", "ret").Ret()
	f.Add(ef.Func())

	if err := os.WriteFile("ascii_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote ascii_ppc64le.s")
}
