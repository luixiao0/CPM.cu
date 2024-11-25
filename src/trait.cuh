#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__half> {
    static __inline__ __device__ float to_float(__half h) {
        return __half2float(h);
    }

    static __inline__ __device__ __half from_float(float f) {
        return __float2half(f);
    }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    static __inline__ __device__ float to_float(__nv_bfloat16 b) {
        return __bfloat162float(b);
    }

    static __inline__ __device__ __nv_bfloat16 from_float(float f) {
        return __float2bfloat16(f);
    }
};