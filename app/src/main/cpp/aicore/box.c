#include <math.h>
#include "box.h"

int equ_val(void *v1, void *v2)
{
	if (!v1 || !v2) return -1;
	
	detection *det1 = (detection *)v1;
	detection *det2 = (detection *)v2;
	
	if (fabs(det1->bbox.x - det2->bbox.x) < 1e-7 &&
		fabs(det1->bbox.y - det2->bbox.y) < 1e-7 &&
		fabs(det1->bbox.w - det2->bbox.w) < 1e-7 &&
		fabs(det1->bbox.h - det2->bbox.h) < 1e-7) {
		return 0;
	}
	
	return -1;
}

void free_val(void *v)
{
	if (!v) return;
	
	detection *det = (detection *)v;
	if (det->probabilities) {
		list_free_mem(det->probabilities);
		det->probabilities = NULL;
	}
	
	list_free_mem(det);
}

float box_intersection(box *b1, box *b2)
{
	float c1 = (b1->x - b1->w / 2) * 768;
	float c2 = (b2->x - b2->w / 2) * 768;
	
	float d1 = (b1->x + b1->w / 2) * 768;
	float d2 = (b2->x + b2->w / 2) * 768;
	
	float j1 = (b1->y - b1->h / 2) * 576;
	float j2 = (b2->y - b2->h / 2) * 576;
	
	float k1 = (b1->y + b1->h / 2) * 576;
	float k2 = (b2->y + b2->h / 2) * 576;
	
	float left = c1 > c2 ? c1 : c2;
	float right = d1 < d2 ? d1 : d2;
	float top = j1 > j2 ? j1 : j2;
	float bottom = k1 < k2 ? k1 : k2;
	
	float width = right - left + 1;
	float height = bottom - top + 1;
	
	if (width < 1 || height < 1) return 0;

	return width * height;
}

float box_union(box *b1, box *b2)
{
	float a1 = b1->w * b1->h * 768 * 576;
	float a2 = b2->w * b2->h * 768 * 576;
	return a1 + a2 - box_intersection(b1, b2);
}

float IOU(box *b1, box *b2)
{
	return box_intersection(b1, b2) / box_union(b1, b2);
}

float penalize_score(float sigma, float score, float iou)
{
	return score * exp(-iou * iou / sigma);
}

list *soft_nms(list *l, float sigma)
{
	if (!l) return NULL;
	
	list *result = make_list();
	while (l->size) {
		node *n = l->head;
		detection *best = list_alloc_mem(sizeof(detection));
		best->probabilities = list_alloc_mem(((detection *)n->val)->classes * sizeof(float));
		while (n) {
			detection *det = (detection *)n->val;
			if (det->objectness > best->objectness) {
				best->bbox = det->bbox;
				best->classes = det->classes;
				best->objectness = det->objectness;
				for (int i = 0; i < det->classes; ++i) {
					best->probabilities[i] = det->probabilities[i];
				}
			}
			n = n->next;
		}

		list_add_tail(result, best);
		list_del_node(l, best, equ_val, free_val);
		
		n = l->head;
		while (n) {
			detection *det = (detection *)n->val;
			float iou = IOU(&best->bbox, &det->bbox);
			det->objectness = penalize_score(sigma, det->objectness, iou);
			n = n->next;
		}
	}
	
	return result;
}