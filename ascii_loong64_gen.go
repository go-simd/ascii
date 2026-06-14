//go:build ignore

// Command gen produces ascii_loong64.s with go-asmgen: LSX kernels (16-byte
// stride) for ASCII case folding.
//
// The letter-range test uses the multiply-free sign-bit method (per-byte
// subtraction, the result's top bit is the range predicate because every input
// byte is < 0x80). For ToLower (range 'A'..'Z'):
//
//	lo   = (b - 'A')      & 0x80   // 0x80 iff b <  'A'
//	hi   = (b - ('Z'+1))  & 0x80   // 0x80 iff b <= 'Z'
//	inr  = hi & (lo ^ 0x80)        // 0x80 iff 'A' <= b <= 'Z'
//	out  = b + (inr >> 2)          // VSRLB $2: 0x80 -> 0x20
//
// ToUpper is symmetric over 'a'..'z' with VSUBB. EqualFold folds both inputs to
// lower case and ORs the per-byte XOR of the two folded blocks; the two 64-bit
// halves are moved to GPRs and OR-ed — zero means equal.
//
// Run: GOWORK=off go run ascii_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

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
	f := emit.NewFile("loong64")

	// loadConsts splats lo,hi,0x80 into V20,V21,V22 from immediates in GPRs.
	loadConsts := func(b *loong64.Builder, lo, hi int) *loong64.Builder {
		return b.
			Raw("MOVV $%d, R7", lo).Raw("VMOVQ R7, V20.B16").
			Raw("MOVV $%d, R7", hi).Raw("VMOVQ R7, V21.B16").
			Raw("MOVV $0x80, R7").Raw("VMOVQ R7, V22.B16")
	}

	// rangeDelta: src bytes in Vsrc -> 0x20 delta into Vdst, scratch V1.
	rangeDelta := func(b *loong64.Builder, Vsrc, Vdst string) *loong64.Builder {
		return b.
			Raw("VSUBB V20, %s, V1", Vsrc).       // b - lo
			Raw("VANDV V22, V1, V1").             // & 0x80
			Raw("VXORV V22, V1, V1").             // ^0x80 -> 0x80 iff b>=lo
			Raw("VSUBB V21, %s, %s", Vsrc, Vdst). // b - hi
			Raw("VANDV V22, %s, %s", Vdst, Vdst). // & 0x80 -> 0x80 iff b<=hi-1
			Raw("VANDV V1, %s, %s", Vdst, Vdst).  // inr
			Raw("VSRLB $2, %s, %s", Vdst, Vdst)   // 0x80>>2 = 0x20
	}

	caseFunc := func(name string, lo, hi int, add bool) *emit.Function {
		op := "VSUBB"
		if add {
			op = "VADDB"
		}
		g := loong64.NewFunc(name, caseSig(), 0)
		g.LoadArg("dst_base", "R4").LoadArg("src_base", "R5").LoadArg("n", "R6")
		loadConsts(g, lo, hi)
		g.Raw("BEQ R6, R0, done").
			Label("loop").
			Raw("VMOVQ (R5), V0")
		rangeDelta(g, "V0", "V2")
		g.Raw("%s V2, V0, V0", op).
			Raw("VMOVQ V0, (R4)").
			Raw("ADDV $16, R5, R5").Raw("ADDV $16, R4, R4").
			Raw("ADDV $-1, R6, R6").Raw("BNE R6, R0, loop").
			Label("done").Ret()
		return g.Func()
	}
	f.Add(caseFunc("lowerBlocks", 'A', 'Z'+1, true))
	f.Add(caseFunc("upperBlocks", 'a', 'z'+1, false))

	foldLower := func(b *loong64.Builder, Vsrc string) *loong64.Builder {
		rangeDelta(b, Vsrc, "V2")
		return b.Raw("VADDB V2, %s, %s", Vsrc, Vsrc)
	}
	ef := loong64.NewFunc("equalFoldBlocks", foldSig(), 0)
	ef.LoadArg("a_base", "R4").LoadArg("b_base", "R5").LoadArg("n", "R6")
	loadConsts(ef, 'A', 'Z'+1)
	ef.Raw("VXORV V23, V23, V23"). // accumulated diff
					Raw("BEQ R6, R0, fdone").
					Label("floop").
					Raw("VMOVQ (R4), V3")
	foldLower(ef, "V3")
	ef.Raw("VMOVQ (R5), V4")
	foldLower(ef, "V4")
	ef.Raw("VXORV V4, V3, V3").
		Raw("VORV V3, V23, V23").
		Raw("ADDV $16, R4, R4").Raw("ADDV $16, R5, R5").
		Raw("ADDV $-1, R6, R6").Raw("BNE R6, R0, floop").
		Label("fdone").
		Raw("VMOVQ V23.V[0], R8").
		Raw("VMOVQ V23.V[1], R9").
		Raw("OR R9, R8, R8").
		Raw("MOVV $0, R10").
		Raw("BNE R8, R0, fret").
		Raw("MOVV $1, R10").
		Label("fret").
		StoreRet("R10", "ret").Ret()
	f.Add(ef.Func())

	if err := os.WriteFile("ascii_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote ascii_loong64.s")
}
