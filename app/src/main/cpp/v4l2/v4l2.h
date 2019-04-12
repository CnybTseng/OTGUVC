#ifndef __V4L2_H__
#define __V4L2_H__
#include <stdint.h>
#include <linux/videodev2.h>
#include "list.h"
#include "lock.h"

#ifdef __cplusplus
extern "C"{
#endif

#define __MAX_REQ_BUF_COUNT__   4
#define __MAX_VIDEO_PLANE__     8
struct video_frame
{
    struct list_head head;

    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[__MAX_VIDEO_PLANE__];

    int width;
    int height;
    int stride;

    void *virt_addr[__MAX_VIDEO_PLANE__];
    void *phys_addr[__MAX_VIDEO_PLANE__];
};

struct v4l2;

enum V4L2_EVENT
{
    V4L2_EVENT_NONE = 0,
    V4L2_EVENT_FRAME_READY,
    V4L2_EVENT_MAX,
};

typedef int32_t (*v4l2_event_notify)(struct v4l2*, int32_t, void*, void*);

struct v4l2_event_action
{
    v4l2_event_notify notify;
    void* object;
    struct v4l2_event_action* next;
};

struct v4l2_operation
{
    int32_t (*init)(struct v4l2*);
    int32_t (*release)(struct v4l2*);

    int32_t (*config)(struct v4l2*,char*,int,int);
    int32_t (*open)(struct v4l2*);
    int32_t (*close)(struct v4l2*);

    int32_t (*prepare_input)(struct v4l2*);
    int32_t (*start_input)(struct v4l2*);
    int32_t (*stop_input)(struct v4l2*);

    int32_t (*prepare_output)(struct v4l2*);
    int32_t (*start_output)(struct v4l2*);
    int32_t (*stop_output)(struct v4l2*);
    int32_t (*transfer_output)(struct v4l2*,uint8_t*,uint32_t);

    int32_t (*put_frame)(struct v4l2*,struct list_head* queue,struct video_frame* pframe);
    struct video_frame* (*get_frame)(struct v4l2*,struct list_head* queue);

    int32_t (*register_notify)(struct v4l2*, int32_t, v4l2_event_notify notify, void*);
    int32_t (*unregister_notify)(struct v4l2*, int32_t, void*);
    int32_t (*trigger_notify)(struct v4l2*, int32_t, void*);
};

struct v4l2
{
    struct list_head head;
    lock_t lock;

    struct list_head capture_free_queue;
    struct list_head capture_used_queue;

    struct list_head output_free_queue;
    struct list_head output_used_queue;

    char src[128];
    int fd;
    int width;
    int height;
    enum v4l2_buf_type capture_type;
    uint32_t capture_pix_fmt;
    enum v4l2_buf_type output_type;

    int running_flag;
    pthread_t pid;

    struct v4l2_operation* op;
    struct v4l2_event_action *paction[V4L2_EVENT_MAX];
};

int32_t create_init_v4l2(struct v4l2** pv4l2);
void release_destroy_v4l2(struct v4l2* pv4l2);

#ifdef __cplusplus
}
#endif

#endif
