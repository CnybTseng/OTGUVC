#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#include "zutils.h"

/** @brief 神经元激活函数.目前仅支持线性整流激活,泄漏线性整流激活,线性激活和逻辑斯蒂激活.
 ** @param X 神经元原始输出或激活输出.
 ** @param n 神经元个数.
 ** @param activation 激活方法.
 **        activation=RELU,线性整流激活.
 **        activation=LEAKY,泄漏线性整流激活.
 **        activation=LINEAR,线性激活.
 **        activation=LOGISTIC,逻辑斯蒂激活.
 **/
AICORE_LOCAL void activate(float *X, int n, ACTIVATION activation);

#ifdef __cplusplus
}
#endif

#endif