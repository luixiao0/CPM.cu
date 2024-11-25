#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void _cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(error) (_cudaCheck(error, __FILE__, __LINE__))

void print_matrix(const float *A, int M, int N) {
  int i;
  std::cerr << std::setprecision(2) << std::fixed;
  std::cerr << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      std::cerr << std::setw(5) << A[i];
    else
      std::cerr << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        std::cerr << ";\n";
    }
  }
  std::cerr << "]\n";
}

bool verify(float *A, float *B, int N, float epsilon = 1e-2) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(A[i] - B[i]);
    if (diff > epsilon) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n", A[i], B[i], diff, i);
      return false;
    }
  }
  return true;
}