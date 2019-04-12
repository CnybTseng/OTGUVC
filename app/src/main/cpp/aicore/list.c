#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list()
{
	list *l = calloc(1, sizeof(list));
	if (!l) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return l;
	}
	
	l->head = NULL;
	l->tail = NULL;
	l->size = 0;
	
	return l;
}

void *list_alloc_mem(size_t size)
{
	return  calloc(1, size);
}

void list_free_mem(void *mem)
{
	if (mem) free(mem);
}

int list_add_tail(list *l, void *val)
{
	if (!l) {
		fprintf(stderr, "invalid list[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	node *n = calloc(1, sizeof(node));
	if (!n) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	n->val  = val;
	n->next = NULL;
	
	if (!l->head) {
		l->head = n;
		l->tail = n;
	} else {
		l->tail->next = n;
		l->tail = n;
	}
	++l->size;
	
	return 0;
}

node *list_del_node(list *l, void *val, int (*equ_val)(void *v1, void *v2),
                    void (*free_val)(void *v))
{
	if (!l) {
		fprintf(stderr, "invalid list[%s:%d].\n", __FILE__, __LINE__);
		return NULL;
	}
	
	node *n = l->head;
	if (!equ_val(val, n->val)) {
		l->head = n->next;
		if (n->val) free_val(n->val);
		free(n);
		if (!l->head) l->tail = NULL;
		--l->size;
		return l->head;
	}
	
	node *prev = n;
	n = n->next;
	while (n) {
		if (!equ_val(val, n->val)) {
			prev->next = n->next;
			if (n->val) free_val(n->val);
			free(n);
			if (!prev->next) l->tail = prev;
			--l->size;
			return prev->next;
		}
		prev = n;
		n = n->next;
	}
	
	return NULL;
}

void list_clear(list *l)
{
	if (!l) return;
	
	node *n = l->head;
	while (n) {
		node *item = n;
		n = n->next;
		if (item->val) free(item->val);
		free(item);
	}
	
	free(l);
}