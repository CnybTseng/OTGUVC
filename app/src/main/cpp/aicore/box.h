#ifndef _BOX_H_
#define _BOX_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#include "zutils.h"

AICORE_LOCAL int equ_val(void *v1, void *v2);
AICORE_LOCAL void free_val(void *v);
AICORE_LOCAL float box_intersection(box *b1, box *b2);
AICORE_LOCAL float box_union(box *b1, box *b2);
AICORE_LOCAL float IOU(box *b1, box *b2);
AICORE_LOCAL float penalize_score(float sigma, float score, float iou);

#ifdef __cplusplus
}
#endif

#endif