// bench.cpp — EdgeZoo Phase 3: INT8 vs FP32 matrix multiply scaling benchmark
//
// Build (no CMake required):
//   clang++ -std=c++17 -O2 -march=native -o bench bench.cpp && ./bench

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// ─── Quantization parameters ──────────────────────────────────────────────────
//
// Inputs A and B are quantized with scale 0.02 (symmetric, zero-point = 0).
// S_C is computed per run from the actual FP32 output range so the INT8
// output grid covers the real accumulation range — not a fixed constant.
//
// Why not fix S_C? Because the output magnitude of a matmul scales with K:
//   E[|C[i][j]|] ~ K * s_A * s_B * E[|q|^2]
// At K=128 the range is ~±300; at K=1024 it is ~±2400. A single fixed S_C
// cannot cover both without clipping (too small) or wasting resolution
// (too large). Computing S_C from the FP32 result makes the error comparison
// meaningful at every size.

static constexpr float S_A = 0.02f;
static constexpr float S_B = 0.02f;

// ─── FP32 matrix multiply ─────────────────────────────────────────────────────

void matmul_fp32(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float*       __restrict__ C,
                 int N, int K)
{
    std::memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k) {
            const float a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                C[i * N + j] += a * B[k * N + j];
        }
}

// ─── INT8 matrix multiply with INT32 accumulation ────────────────────────────
//
// Why INT32 accumulation?
//   INT8 × INT8 → up to 16-bit product.
//   Summing K products → up to 16 + log2(K) bits.
//   For K=1024: 26 bits. INT32 guarantees no overflow for any K.
//
// Requantization happens after the inner loop, not inside it.

void matmul_int8(const int8_t* __restrict__ A,
                 const int8_t* __restrict__ B,
                 int8_t*       __restrict__ C,
                 int N, int K, float requant_scale)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<int32_t>(A[i * K + k])
                     * static_cast<int32_t>(B[k * N + j]);
            const float scaled = static_cast<float>(acc) * requant_scale;
            C[i * N + j] = static_cast<int8_t>(
                std::clamp(static_cast<int>(std::round(scaled)), -128, 127));
        }
}

// ─── Timing harness ───────────────────────────────────────────────────────────

template<typename Fn>
double bench(Fn fn, int n_reps)
{
    fn();  // warm-up, excluded from timing
    std::vector<double> t(n_reps);
    for (int r = 0; r < n_reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        t[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return std::accumulate(t.begin(), t.end(), 0.0) / n_reps;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

void fill_float(float* data, int n, float range, unsigned seed)
{
    std::srand(seed);
    for (int i = 0; i < n; ++i)
        data[i] = range * (2.f * static_cast<float>(std::rand()) /
                           static_cast<float>(RAND_MAX) - 1.f);
}

void quantize(const float* src, int8_t* dst, int n, float scale)
{
    for (int i = 0; i < n; ++i)
        dst[i] = static_cast<int8_t>(
            std::clamp(static_cast<int>(std::round(src[i] / scale)), -128, 127));
}

// S_C derived from the actual FP32 output range.
float compute_sc(const float* C, int n)
{
    float mx = 0.f;
    for (int i = 0; i < n; ++i) mx = std::max(mx, std::abs(C[i]));
    return mx / 127.f;
}

double max_abs_error(const float* C_fp, const int8_t* C_i8, int n, float sc)
{
    double err = 0.0;
    for (int i = 0; i < n; ++i)
        err = std::max(err, std::abs(static_cast<double>(C_fp[i]) -
                                     static_cast<double>(C_i8[i]) * sc));
    return err;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main()
{
    struct Run { int N; int reps; };
    const Run runs[] = { {128, 1000}, {512, 100}, {1024, 10} };

    std::cout << "\n── EdgeZoo Phase 3: INT8 vs FP32 Scaling Benchmark ──\n\n";
    std::cout << std::left
              << std::setw(12) << "Size"
              << std::setw(8)  << "Reps"
              << std::setw(14) << "FP32 (ms)"
              << std::setw(14) << "INT8 (ms)"
              << std::setw(12) << "Speedup"
              << "Max Err\n";
    std::cout << std::string(74, '-') << "\n";

    for (auto& run : runs) {
        const int N = run.N, K = N, ne = N * K;

        std::vector<float>  A_fp(ne), B_fp(ne), C_fp(N * N);
        std::vector<int8_t> A_i8(ne), B_i8(ne), C_i8(N * N);

        fill_float(A_fp.data(), ne, 2.54f, 42u);
        fill_float(B_fp.data(), ne, 2.54f, 99u);
        quantize(A_fp.data(), A_i8.data(), ne, S_A);
        quantize(B_fp.data(), B_i8.data(), ne, S_B);

        // Compute S_C from the FP32 reference output
        std::fill(C_fp.begin(), C_fp.end(), 0.f);
        matmul_fp32(A_fp.data(), B_fp.data(), C_fp.data(), N, K);
        const float sc      = compute_sc(C_fp.data(), N * N);
        const float requant = S_A * S_B / sc;

        double fp_ms = bench([&]() {
            std::fill(C_fp.begin(), C_fp.end(), 0.f);
            matmul_fp32(A_fp.data(), B_fp.data(), C_fp.data(), N, K);
        }, run.reps);

        double i8_ms = bench([&]() {
            matmul_int8(A_i8.data(), B_i8.data(), C_i8.data(), N, K, requant);
        }, run.reps);

        // Fresh outputs for error measurement
        std::fill(C_fp.begin(), C_fp.end(), 0.f);
        matmul_fp32(A_fp.data(), B_fp.data(), C_fp.data(), N, K);
        matmul_int8(A_i8.data(), B_i8.data(), C_i8.data(), N, K, requant);
        double err = max_abs_error(C_fp.data(), C_i8.data(), N * N, sc);

        std::string label = std::to_string(N) + "x" + std::to_string(N);
        std::cout << std::left << std::fixed << std::setprecision(3)
                  << std::setw(12) << label
                  << std::setw(8)  << run.reps
                  << std::setw(14) << fp_ms
                  << std::setw(14) << i8_ms
                  << std::setw(12) << (fp_ms / i8_ms)
                  << err << "\n";
    }

    std::cout << "\nSpeedup > 1 means INT8 is faster. < 1 means FP32 is faster.\n"
              << "Max Err is in FP32 output units (one S_C step = one INT8 grid step).\n\n";
    return 0;
}