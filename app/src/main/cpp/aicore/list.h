#ifndef _LIST_H_
#define _LIST_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "zutils.h"

typedef struct _node {
	void *val;
	struct _node *next;
} node;

typedef struct {
	node *head;
	node *tail;
	int size;
} list;

AICORE_LOCAL list *make_list();
AICORE_LOCAL void *list_alloc_mem(size_t size);
AICORE_LOCAL void list_free_mem(void *mem);
AICORE_LOCAL int list_add_tail(list *l, void *val);
AICORE_LOCAL node *list_del_node(list *l, void *val, int (*equ_val)(void *v1, void *v2),
                    void (*free_val)(void *v));
AICORE_LOCAL void list_clear(list *l);

#ifdef __cplusplus
}
#endif

#endif