#ifndef _TRT_FP16_H_
#define _TRT_FP16_H_

#include <cublas_v2.h>
#include <cstdint>

namespace fp16
{
// Code added before equivalent code was available via cuda.
// This code needs to be removed when we ship for cuda-9.2.
template<typename T, typename U> T bitwise_cast(U u);

__half __float2half(float f);

float __half2float(__half h);

};

#endif // _TRT_FP16_H_
