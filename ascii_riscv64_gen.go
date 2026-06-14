//go:build ignore

// Command gen produces ascii_riscv64.s with go-asmgen: RVV kernels (16-byte
// stride) for ASCII case folding.
//
// RVV has unsigned vector compares that produce mask registers and a masked
// vector add, so the letter-range fold is a compare + masked add (no multiply,
// no table). For ToLower (range 'A'..'Z'):
//
//	V0 = (b > 'A'-1)               // VMSGTUVX
//	v1 = (b <= 'Z')               // VMSLEUVX
//	V0 = V0 & v1                   // VMANDMM -> in-range mask
//	b  = b + 0x20 under mask V0    // VADDVX ..., V0, b
//
// ToUpper is symmetric over 'a'..'z' with VSUBVX. EqualFold folds both inputs to
// lower case, ORs the per-byte XOR of the two folded blocks, then VMSNE against
// zero + VCPOPM counts any differing byte — zero means equal.
//
// Run: GOWORK=off go run ascii_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
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
	f := emit.NewFile("riscv64")

	// mkCase builds a ToLower (add) / ToUpper (sub) kernel: read src (X6), store
	// to dst (X5).
	mkCase := func(name string, lo, hi int, add bool) *emit.Function {
		op := "VSUBVX"
		if add {
			op = "VADDVX"
		}
		g := riscv64.NewFunc(name, caseSig(), 0)
		g.LoadArg("dst_base", "X5").LoadArg("src_base", "X6").LoadArg("n", "X7").
			Raw("VSETVLI $16, E8, M1, TA, MA, X8").
			Raw("MOV $%d, X14", lo-1).
			Raw("MOV $%d, X15", hi).
			Raw("MOV $0x20, X16").
			Raw("BEQ X7, X0, done").
			Label("loop").
			Raw("VLE8V (X6), V1").
			Raw("VMSGTUVX X14, V1, V0").
			Raw("VMSLEUVX X15, V1, V2").
			Raw("VMANDMM V2, V0, V0").
			Raw("%s X16, V1, V0, V1", op).
			Raw("VSE8V V1, (X5)").
			Raw("ADD $16, X6, X6").Raw("ADD $16, X5, X5").
			Raw("ADD $-1, X7, X7").Raw("BNE X7, X0, loop").
			Label("done").Ret()
		return g.Func()
	}
	f.Add(mkCase("lowerBlocks", 'A', 'Z', true))
	f.Add(mkCase("upperBlocks", 'a', 'z', false))

	// EqualFold: fold both to lower, accumulate OR of XORs in V8.
	// foldLower folds the bytes in Vsrc to lower in place using X14='A'-1,
	// X15='Z', X16=0x20; mask in V0, scratch V2.
	foldLower := func(b *riscv64.Builder, Vsrc string) *riscv64.Builder {
		return b.
			Raw("VMSGTUVX X14, %s, V0", Vsrc).
			Raw("VMSLEUVX X15, %s, V2", Vsrc).
			Raw("VMANDMM V2, V0, V0").
			Raw("VADDVX X16, %s, V0, %s", Vsrc, Vsrc)
	}
	ef := riscv64.NewFunc("equalFoldBlocks", foldSig(), 0)
	ef.LoadArg("a_base", "X5").LoadArg("b_base", "X6").LoadArg("n", "X7").
		Raw("VSETVLI $16, E8, M1, TA, MA, X8").
		Raw("MOV $%d, X14", 'A'-1).
		Raw("MOV $%d, X15", 'Z').
		Raw("MOV $0x20, X16").
		Raw("VXORVV V8, V8, V8"). // V8 = 0 (accumulator)
		Raw("BEQ X7, X0, fdone").
		Label("floop").
		Raw("VLE8V (X5), V3")
	foldLower(ef, "V3")
	ef.Raw("VLE8V (X6), V4")
	foldLower(ef, "V4")
	ef.Raw("VXORVV V4, V3, V3"). // diff = la ^ lb
					Raw("VORVV V3, V8, V8"). // accumulate
					Raw("ADD $16, X5, X5").Raw("ADD $16, X6, X6").
					Raw("ADD $-1, X7, X7").Raw("BNE X7, X0, floop").
					Label("fdone").
					Raw("VMSNEVI $0, V8, V0"). // V0 = mask(acc != 0)
					Raw("VCPOPM V0, X10").     // count of differing bytes
					Raw("MOV $1, X11").
					Raw("BEQ X10, X0, fret"). // none -> ret 1
					Raw("MOV $0, X11").
					Label("fret").
					StoreRet("X11", "ret").Ret()
	f.Add(ef.Func())

	if err := os.WriteFile("ascii_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote ascii_riscv64.s")
}
