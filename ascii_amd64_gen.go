//go:build ignore

// Command gen produces ascii_amd64.s with go-asmgen: SSE2 (16-byte) and AVX2
// (32-byte) kernels for ASCII case folding.
//
// The wrapper guarantees every input byte is < 0x80 before calling these
// kernels, so signed byte compares (PCMPGTB) coincide with unsigned ones over
// the whole ASCII range. ToLower adds 0x20 to bytes in 'A'..'Z'; ToUpper
// subtracts 0x20 from bytes in 'a'..'z'; the range mask is built with no table
// and no multiply:
//
//	gt   = (b > lo-1)          // PCMPGTB b, splat(lo-1)
//	lt   = (b < hi+1)          // PCMPGTB splat(hi+1), b
//	mask = gt & lt             // 0xFF inside the letter range, else 0
//	out  = b +/- (mask & 0x20)
//
// EqualFold folds both inputs the same way and ORs the per-byte XOR of the two
// folded blocks across the loop; (V)PTEST at the end reports any difference and
// SETEQ returns 1 (equal) or 0 (some byte differed).
//
// Run: GOWORK=off go run ascii_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
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
	f := emit.NewFile("amd64")

	loA := f.Data("loA", repByte('A'-1, 16)) // 0x40
	hiZ := f.Data("hiZ", repByte('Z'+1, 16)) // 0x5B
	loa := f.Data("loa", repByte('a'-1, 16)) // 0x60
	hiz := f.Data("hiz", repByte('z'+1, 16)) // 0x7B
	c20 := f.Data("c20", repByte(0x20, 16))  // 0x20
	loAb := f.Data("loAb", repByte('A'-1, 32))
	hiZb := f.Data("hiZb", repByte('Z'+1, 32))
	loab := f.Data("loab", repByte('a'-1, 32))
	hizb := f.Data("hizb", repByte('z'+1, 32))
	c20b := f.Data("c20b", repByte(0x20, 32))

	// ---- SSE case kernel (16-byte stride). add=true -> ToLower, else ToUpper.
	sseCase := func(name, loC, hiC string, add bool) *emit.Function {
		op := "PSUBB"
		if add {
			op = "PADDB"
		}
		s := amd64.NewFunc(name, caseSig(), 0)
		s.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("n", "CX").
			Raw("MOVOU %s+0(SB), X5", loC).
			Raw("MOVOU %s+0(SB), X6", hiC).
			Raw("MOVOU %s+0(SB), X7", c20).
			Raw("TESTQ CX, CX").Raw("JZ done").
			Label("loop").
			Raw("MOVOU (SI), X0").
			Raw("MOVO X0, X1").Raw("PCMPGTB X5, X1"). // X1 = (b > lo-1)
			Raw("MOVO X6, X2").Raw("PCMPGTB X0, X2"). // X2 = (hi+1 > b)
			Raw("PAND X2, X1").                       // mask = gt & lt
			Raw("PAND X7, X1").                       // mask & 0x20
			Raw("%s X1, X0", op).                     // b +/- delta
			Raw("MOVOU X0, (DI)").
			Raw("ADDQ $16, SI").Raw("ADDQ $16, DI").Raw("DECQ CX").Raw("JNZ loop").
			Label("done").Ret()
		return s.Func()
	}
	f.Add(sseCase("lowerBlocksSSE", loA, hiZ, true))
	f.Add(sseCase("upperBlocksSSE", loa, hiz, false))

	// ---- AVX2 case kernel (32-byte stride).
	avxCase := func(name, loC, hiC string, add bool) *emit.Function {
		op := "VPSUBB"
		if add {
			op = "VPADDB"
		}
		v := amd64.NewFunc(name, caseSig(), 0)
		v.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("n", "CX").
			Raw("VMOVDQU %s+0(SB), Y5", loC).
			Raw("VMOVDQU %s+0(SB), Y6", hiC).
			Raw("VMOVDQU %s+0(SB), Y7", c20b). // 32-byte 0x20 splat
			Raw("TESTQ CX, CX").Raw("JZ vdone").
			Label("vloop").
			Raw("VMOVDQU (SI), Y0").
			Raw("VPCMPGTB Y5, Y0, Y1"). // (b > lo-1)
			Raw("VPCMPGTB Y0, Y6, Y2"). // (hi+1 > b)
			Raw("VPAND Y2, Y1, Y1").    // mask
			Raw("VPAND Y7, Y1, Y1").    // mask & 0x20
			Raw("%s Y1, Y0, Y0", op).   // b +/- delta
			Raw("VMOVDQU Y0, (DI)").
			Raw("ADDQ $32, SI").Raw("ADDQ $32, DI").Raw("DECQ CX").Raw("JNZ vloop").
			Label("vdone").Raw("VZEROUPPER").Ret()
		return v.Func()
	}
	f.Add(avxCase("lowerBlocksAVX2", loAb, hiZb, true))
	f.Add(avxCase("upperBlocksAVX2", loab, hizb, false))

	// ---- SSE EqualFold (16-byte stride): fold both to lower, accumulate XOR.
	// foldLower folds the 16 bytes at memSrc to lower case into dstReg, using
	// X5=lo-1, X6=hi+1, X7=0x20 and X3 as scratch.
	foldLowerSSE := func(b *amd64.Builder, memSrc, dstReg string) *amd64.Builder {
		return b.
			Raw("MOVOU %s, %s", memSrc, dstReg).
			Raw("MOVO %s, X3", dstReg).Raw("PCMPGTB X5, X3"). // gt
			Raw("MOVO X6, X4").Raw("PCMPGTB %s, X4", dstReg). // lt
			Raw("PAND X4, X3").Raw("PAND X7, X3").            // delta
			Raw("PADDB X3, %s", dstReg)                       // lower
	}
	sf := amd64.NewFunc("equalFoldBlocksSSE", foldSig(), 0)
	sf.LoadArg("a_base", "DI").LoadArg("b_base", "SI").LoadArg("n", "CX").
		Raw("MOVOU %s+0(SB), X5", loA).
		Raw("MOVOU %s+0(SB), X6", hiZ).
		Raw("MOVOU %s+0(SB), X7", c20).
		Raw("PXOR X8, X8"). // accumulated difference
		Raw("TESTQ CX, CX").Raw("JZ fdone")
	sf.Label("floop")
	foldLowerSSE(sf, "(DI)", "X0")
	foldLowerSSE(sf, "(SI)", "X1")
	sf.Raw("PXOR X1, X0").Raw("POR X0, X8").
		Raw("ADDQ $16, DI").Raw("ADDQ $16, SI").Raw("DECQ CX").Raw("JNZ floop").
		Label("fdone").
		Raw("XORQ AX, AX").Raw("PTEST X8, X8").Raw("SETEQ AL").
		StoreRet("AX", "ret").Ret()
	f.Add(sf.Func())

	// ---- AVX2 EqualFold (32-byte stride).
	foldLowerAVX := func(b *amd64.Builder, memSrc, dstReg string) *amd64.Builder {
		return b.
			Raw("VMOVDQU %s, %s", memSrc, dstReg).
			Raw("VPCMPGTB Y5, %s, Y3", dstReg). // gt
			Raw("VPCMPGTB %s, Y6, Y4", dstReg). // lt
			Raw("VPAND Y4, Y3, Y3").Raw("VPAND Y7, Y3, Y3").
			Raw("VPADDB Y3, %s, %s", dstReg, dstReg)
	}
	vf := amd64.NewFunc("equalFoldBlocksAVX2", foldSig(), 0)
	vf.LoadArg("a_base", "DI").LoadArg("b_base", "SI").LoadArg("n", "CX").
		Raw("VMOVDQU %s+0(SB), Y5", loAb).
		Raw("VMOVDQU %s+0(SB), Y6", hiZb).
		Raw("VMOVDQU %s+0(SB), Y7", c20b).
		Raw("VPXOR Y8, Y8, Y8").
		Raw("TESTQ CX, CX").Raw("JZ vfdone")
	vf.Label("vfloop")
	foldLowerAVX(vf, "(DI)", "Y0")
	foldLowerAVX(vf, "(SI)", "Y1")
	vf.Raw("VPXOR Y1, Y0, Y0").Raw("VPOR Y0, Y8, Y8").
		Raw("ADDQ $32, DI").Raw("ADDQ $32, SI").Raw("DECQ CX").Raw("JNZ vfloop").
		Label("vfdone").
		Raw("XORQ AX, AX").Raw("VPTEST Y8, Y8").Raw("SETEQ AL").Raw("VZEROUPPER").
		StoreRet("AX", "ret").Ret()
	f.Add(vf.Func())

	if err := os.WriteFile("ascii_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote ascii_amd64.s")
}
