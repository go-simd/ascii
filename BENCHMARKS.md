# Performance parity — go-simd/ascii vs stdlib

**Reference:** the `bytes` package stdlib (`bytes.ToUpper` / `bytes.ToLower` /
`bytes.EqualFold`) — the portable scalar baseline go-simd/ascii accelerates with
a SIMD case-fold/compare kernel (amd64 AVX2/SSE, arm64 NEON, plus ppc64le/s390x).
Input: 4 KiB of random printable ASCII (seed 1), single core. `b.SetBytes(len)`
so `go test` reports MB/s.

## amd64 (AVX2, GitHub Actions x86_64 runner — ratios valid, absolute ns/op CI-noisy)

**Methodology.** GitHub Actions `ubuntu-latest` runner, **AMD EPYC 9V74** (`avx2`
present, **no `avx512*`** — confirmed from `/proc/cpuinfo`), `GOAMD64` baseline,
Go stable, single core. `-count=6`, **min-of-6** (best run per benchmark). The
runner is shared, so absolute throughput is noisy; the **ratios vs the `bytes`
stdlib** are measured back-to-back on the *same* CPU and are valid. Reproduce via
`gh workflow run bench-amd64.yml`.

| op | go-simd (MB/s) | stdlib `bytes` | ×stdlib | verdict |
|----|---------------:|---------------:|--------:|---------|
| ToUpper   | 3685 |  685 (ToUpper) | 5.38× | wins |
| ToLower   | 3376 |  685 (ToUpper proxy¹) | ~4.93× | wins |
| EqualFold | 4016 | 1879 (EqualFold) | 2.14× | wins |

¹ `bytes.ToLower` has no dedicated benchmark in the harness; `bytes.ToUpper` is
the same-cost scalar proxy (both are byte-wise case maps), so the ToLower ratio
is reported against it.

* The AVX2 case-fold/compare kernel **beats the `bytes` stdlib 2.1–5.4×** on
  amd64. ToUpper/ToLower allocate one output buffer (1 alloc/op, unavoidable for
  the non-in-place API); EqualFold is allocation-free.

### Notes
* Output is byte-identical to the `bytes` package on every input (100% coverage,
  fuzz-clean across the case-fold and comparison paths).
* arm64 (M4 Max NEON) numbers are not yet captured in this file; the amd64 AVX2
  column above is the GitHub Actions measurement. Different hardware/ISA rows are
  not directly comparable in absolute terms.
