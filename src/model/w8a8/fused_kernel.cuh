#pragma once
#include <cuda_fp16.h>
#include <cassert>
#include "../../reduction_utils.cuh"
#include "../../utils.cuh"
#include "../../utilsq.cuh"


namespace{
template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(T *__restrict__ input,
                             int8_t *__restrict__ output, scale_type scale,
                             int num_tokens, int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx * hidden_size + i];
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = __float2half_rn(block_amax_val / 127.0f);
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / __half2float(scale));
    }
  }
}
}

template <typename T, bool use_per_token_quant>
struct Quantizer
{
    int dim;
    int8_t* output;
    // using ScaleType = typename std::conditional<use_per_token_quant, half *, half>::type;
    half* output_scale;
    
    Quantizer(int dim)
    {
        this->dim = dim;
    }

    int64_t init_output_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(int8_t));
    }

    int64_t init_output_scale_ptr(Memory * memory, int32_t num_tokens, int64_t offset){
        return memory->allocate((void**)&this->output_scale, offset, num_tokens * sizeof(half));
    }
    

    void invoke(T *input,
                int num_tokens)
    {
        dim3 grid(num_tokens);
        dim3 block(std::min(dim, 1024));

        quant_kernel<T, half *, true><<<grid, block, 0, calc_stream>>>(input, output, output_scale, num_tokens, dim);

    }
};