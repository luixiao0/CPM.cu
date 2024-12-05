#include "utils.cuh"

bool initialized = false;

cudaStream_t calc_stream;
cublasHandle_t cublas_handle;

bool graphCreated = false;
cudaGraph_t graph;
cudaGraphExec_t graphExec;

void init_resources() {
  if (initialized) return;
  cudaCheck(cudaStreamCreate(&calc_stream));
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasSetStream(cublas_handle, calc_stream));
  initialized = true;
}