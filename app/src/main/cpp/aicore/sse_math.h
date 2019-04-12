#ifdef __INTEL_SSE__
#ifndef _SSE_MATH_H_
#define _SSE_MATH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <xmmintrin.h>
	
typedef __m128 v4sf;

v4sf log_ps(v4sf x);
v4sf exp_ps(v4sf x);
v4sf sin_ps(v4sf x);
v4sf cos_ps(v4sf x);
void sincos_ps(v4sf x, v4sf *s, v4sf *c);

#ifdef __cplusplus
}
#endif

#endif
#endif