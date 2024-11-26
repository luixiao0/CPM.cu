#include "utils.cuh"

bool initialized = false;
cublasHandle_t cublas_handle;

void init_cublas() {
  if (initialized) return;
  cublasCheck(cublasCreate(&cublas_handle));
  initialized = true;
}