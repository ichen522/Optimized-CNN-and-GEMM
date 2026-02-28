#include "q.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Homework 1 (CPU Matrix Multiplication Microbenchmark)
// -----------------------------------------------------------------------------
// Goal:
//   Compare runtime across multiple loop orderings / data layouts for GEMM.
//
// Alternatives in this file:
//   1) gemm_ijk
//   2) gemm_kij
//   3) gemm_jki
//   4) gemm_ijk_bt   (B is transposed first)
//   5) gemm_tiled  (tile size T)
//
// Student tasks:
//   A) Run timing experiments for all alternatives.
//   B) Report average runtime over multiple trials.
//   C) Sweep matrix size N (e.g., 128, 256, 512, 1024, ...).
//   D) For blocked GEMM, sweep tile size T (e.g., 8, 16, 32, 64).
//   E) Explain trends using cache locality and memory-access patterns.
//
// Notes:
//   - Use the helper utilities near the bottom of this file for timing.
//   - Always reset output matrix C between trials.
//   - Run one warm-up iteration before measuring.
//   - Keep compiler flags constant across all variants.
// -----------------------------------------------------------------------------

void gemm_ijk(const Matrix& A, const Matrix& B, Matrix& C) {
  const int N = A.size();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i][j] = 0.0;
      for (int k = 0; k < N; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void gemm_kij(const Matrix& A, const Matrix& B, Matrix& C) {
  const int N = A.size();
  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void gemm_jki(const Matrix& A, const Matrix& B, Matrix& C) {
  const int N = A.size();
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
      for (int i = 0; i < N; ++i) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void gemm_ijk_bt(const Matrix& A, const Matrix& BT, Matrix& C) {
  const int N = A.size();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i][j] = 0.0;
      for (int k = 0; k < N; ++k) {
        C[i][j] += A[i][k] * BT[j][k];
      }
    }
  }
}

void gemm_tiled(const Matrix& A, const Matrix& B, Matrix& C, int T) {
  const int N = A.size();
  // Three outer loops iterate over tiles
  // Note that the stride is T.
  // T is the tile size (block size) for the blocked matrix multiplication.
  // It determines the size of the sub-matrices processed at once to improve
  // cache locality.
  for (int i = 0; i < N; i += T) {
    for (int j = 0; j < N; j += T) {
      for (int k = 0; k < N; k += T) {
        // three inner loops iterate within tiles
        for (int jl = j; jl < j + T; ++jl) {
          for (int il = i; il < i + T; ++il) {
            for (int kl = k; kl < k + T; ++kl) {
              C[il][jl] += A[il][kl] * B[kl][jl];
            }
          }
        }
      }
    }
  }
}

namespace {

// -----------------------------------------------------------------------------
// Helper functions for timing and matrix generation.
// PLEASE DO NOT MODIFY BELOW THIS LINE.
// -----------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

// Generates a random N x N matrix with values in [-1.0, 1.0].
Matrix make_random_matrix(int n, uint32_t seed = 0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  Matrix M(n, std::vector<double>(n));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      M[i][j] = dist(rng);
    }
  }
  return M;
}

// Transposes a matrix.
Matrix transpose(const Matrix& M) {
  const int n = static_cast<int>(M.size());
  Matrix MT(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      MT[j][i] = M[i][j];
    }
  }
  return MT;
}
// Resets a matrix to all zeros.
void reset_matrix(Matrix& C) {
  for (auto& row : C) {
    std::fill(row.begin(), row.end(), 0.0);
  }
}

// Micro-benchmark helper: runs the given GEMM implementation multiple times.
// Returns the average runtime in milliseconds.}

template <typename Fn>
double benchmark_ms(const std::string& name, Fn&& gemm_impl, const Matrix& A,
                    const Matrix& B, Matrix& C, int trials = 5) {
  std::cout << "  Running " << name << "..." << std::endl;
  // Warm-up
  reset_matrix(C);
  gemm_impl(A, B, C);

  auto t0 = Clock::now();
  for (int r = 0; r < trials; ++r) {
    reset_matrix(C);
    gemm_impl(A, B, C);
  }
  auto t1 = Clock::now();

  const double total_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return total_ms / static_cast<double>(trials);
}

}  // namespace

// -----------------------------------------------------------------------------
// Optional helper to run one benchmark sweep.
// Call this from your own driver / test code if desired.
// -----------------------------------------------------------------------------
void run_homework_benchmark_example(int n, int tile_size, int trials = 5) {
  int T = tile_size;
  int N = n;

  Matrix A = make_random_matrix(N, 1);
  Matrix B = make_random_matrix(N, 2);
  Matrix BT = transpose(B);
  Matrix C(N, std::vector<double>(N, 0.0));

  const double t_ijk = benchmark_ms("gemm_ijk", gemm_ijk, A, B, C, trials);
  const double t_kij = benchmark_ms("gemm_kij", gemm_kij, A, B, C, trials);
  const double t_jki = benchmark_ms("gemm_jki", gemm_jki, A, B, C, trials);
  const double t_bt =
      benchmark_ms("gemm_ijk_bt", gemm_ijk_bt, A, BT, C, trials);
  const double t_blk = benchmark_ms(
      "gemm_tiled",
      [&](const Matrix& A, const Matrix& B, Matrix& C) {
        gemm_tiled(A, B, C, T);
      },
      A, B, C, trials);

  std::cout << "N=" << N << ", T=" << T << ", trials=" << trials << "\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "  gemm_ijk     : " << t_ijk << " ms\n";
  std::cout << "  gemm_kij     : " << t_kij << " ms\n";
  std::cout << "  gemm_jki     : " << t_jki << " ms\n";
  std::cout << "  gemm_ijk_bt  : " << t_bt << " ms\n";
  std::cout << "  gemm_tiled : " << t_blk << " ms\n";
}

int main(int argc, char** argv) {
  int n = 1024;
  int tile = 64;
  int trials = 1;

  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    tile = std::atoi(argv[2]);
  }
  if (argc > 3) {
    trials = std::atoi(argv[3]);
  }

  std::cout << "Running benchmark with settings: N=" << n << ", T=" << tile
            << ", trials=" << trials << std::endl;

  run_homework_benchmark_example(n, tile, trials);
  return 0;
}