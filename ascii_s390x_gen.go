//go:build ignore

// Command gen produces ascii_s390x.s with go-asmgen: vector-facility kernels
// (16-byte stride) for ASCII case folding on s390x (IBM Z, z13+, BIG-ENDIAN).
//
// The vector facility is the s390x baseline, so no runtime feature check is
// needed. The letter-range test uses the multiply-free sign-bit method (per-byte
// subtract; the top bit of the result is the range predicate because every input
// byte is < 0x80). For ToLower (range 'A'..'Z'):
//
//	lo   = (b - 'A')      & 0x80   // 0x80 iff b <  'A'   (VSB, VN)
//	hi   = (b - ('Z'+1))  & 0x80   // 0x80 iff b <= 'Z'
//	inr  = hi & (lo ^ 0x80)        // 0x80 iff 'A' <= b <= 'Z'  (VX, VN)
//	out  = b + (inr >> 2)          // VESRLB $2: 0x80 -> 0x20   (VAB)
//
// ToUpper is symmetric over 'a'..'z' with VSB. Constants are built in-register
// with VREPIB (replicate immediate byte) — no data section needed.
//
// BIG-ENDIAN NOTE: every operation here is per-byte and VL/VST move bytes in
// natural memory order on s390x (lane 0 == first memory byte), so the kernel is
// endian-neutral; the position-sensitive Fuzz* and table tests are the gate.
// EqualFold folds both inputs to lower case, ORs the per-byte XOR of the two
// folded blocks, then extracts the two 64-bit halves with VLGVG and ORs them —
// zero means equal. Run: GOWORK=off go run ascii_s390x_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/s390x"
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
	f := emit.NewFile("s390x")

	// loadConsts: V20=lo, V21=hi, V22=0x80 via replicate-immediate-byte.
	loadConsts := func(b *s390x.Builder, lo, hi int) *s390x.Builder {
		return b.
			Raw("VREPIB $%d, V20", lo).
			Raw("VREPIB $%d, V21", hi).
			Raw("VREPIB $128, V22")
	}

	// rangeDelta: src bytes in Vsrc -> 0x20 delta into Vdst, scratch V1.
	rangeDelta := func(b *s390x.Builder, Vsrc, Vdst string) *s390x.Builder {
		return b.
			Raw("VSB V20, %s, V1", Vsrc).        // b - lo
			Raw("VN V22, V1, V1").               // & 0x80
			Raw("VX V22, V1, V1").               // ^0x80 -> 0x80 iff b>=lo
			Raw("VSB V21, %s, %s", Vsrc, Vdst).  // b - hi
			Raw("VN V22, %s, %s", Vdst, Vdst).   // & 0x80 -> 0x80 iff b<=hi-1
			Raw("VN V1, %s, %s", Vdst, Vdst).    // inr
			Raw("VESRLB $2, %s, %s", Vdst, Vdst) // >>2 -> 0x20
	}

	mkCase := func(name string, lo, hi int, add bool) *emit.Function {
		op := "VSB"
		if add {
			op = "VAB"
		}
		g := s390x.NewFunc(name, caseSig(), 0)
		g.LoadArg("dst_base", "R1").LoadArg("src_base", "R2").LoadArg("n", "R3")
		loadConsts(g, lo, hi)
		g.Raw("CMPBEQ R3, $0, done").
			Label("loop").
			Raw("VL (R2), V0") // V0 = 16 src bytes
		rangeDelta(g, "V0", "V2")
		g.Raw("%s V2, V0, V0", op). // out = b +/- delta
						Raw("VST V0, (R1)").
						Raw("ADD $16, R2").
						Raw("ADD $16, R1").
						Raw("ADD $-1, R3").
						Raw("CMPBNE R3, $0, loop").
						Label("done").Ret()
		return g.Func()
	}
	f.Add(mkCase("lowerBlocks", 'A', 'Z'+1, true))
	f.Add(mkCase("upperBlocks", 'a', 'z'+1, false))

	foldLower := func(b *s390x.Builder, Vsrc string) *s390x.Builder {
		rangeDelta(b, Vsrc, "V2")
		return b.Raw("VAB V2, %s, %s", Vsrc, Vsrc)
	}
	ef := s390x.NewFunc("equalFoldBlocks", foldSig(), 0)
	ef.LoadArg("a_base", "R1").LoadArg("b_base", "R2").LoadArg("n", "R3")
	loadConsts(ef, 'A', 'Z'+1)
	ef.Raw("VZERO V24"). // accumulator
				Raw("CMPBEQ R3, $0, fdone").
				Label("floop").
				Raw("VL (R1), V3")
	foldLower(ef, "V3")
	ef.Raw("VL (R2), V4")
	foldLower(ef, "V4")
	ef.Raw("VX V4, V3, V3"). // diff
					Raw("VO V3, V24, V24"). // accumulate
					Raw("ADD $16, R1").
					Raw("ADD $16, R2").
					Raw("ADD $-1, R3").
					Raw("CMPBNE R3, $0, floop").
					Label("fdone").
					Raw("VLGVG $0, V24, R5"). // high doubleword
					Raw("VLGVG $1, V24, R6"). // low doubleword
					Raw("OR R6, R5").
					Raw("MOVD $1, R7").
					Raw("CMPBEQ R5, $0, fret"). // no diff -> 1
					Raw("MOVD $0, R7").
					Label("fret").
					StoreRet("R7", "ret").Ret()
	f.Add(ef.Func())

	if err := os.WriteFile("ascii_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote ascii_s390x.s")
}
