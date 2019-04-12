//
// Created by luoji on 2/22/2019.
//

#ifndef TFOTG_LOCK_H
#define TFOTG_LOCK_H


#include <pthread.h>

typedef struct{
    pthread_mutex_t lock;
}lock_t;

extern void lock_init(lock_t *);
extern void lock(lock_t *);
extern void unlock(lock_t *);
extern void lock_destroy(lock_t *);

#endif //TFOTG_LOCK_H
