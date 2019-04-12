#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <poll.h>
#include <stdio.h>
#include "v4l2.h"

#define __V4L2_MAX_POLL_FD__    1

static void local_frame_save(uint8_t* data, int length)
{
    char name[64];
    FILE* fp;
    sprintf(name, "./raw.yuv");
    fp = fopen(name, "ab+");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        fwrite(data, length, 1, fp);
        fclose(fp);
    }
}

static void* v4l2_capture_poll_thread(void* context)
{
    int ret;
    struct video_frame* pframe;
    struct v4l2* pv4l2;
    struct pollfd pfds[__V4L2_MAX_POLL_FD__];

    pv4l2 = (struct v4l2*)context;
    if(!pv4l2){
        return NULL;
    }

    while(pv4l2->running_flag){
        memset(pfds,0,sizeof(struct pollfd) * __V4L2_MAX_POLL_FD__);
        pfds[0].fd = pv4l2->fd;
        pfds[0].events = POLLIN | POLLOUT | POLLERR;
        ret = poll(pfds, __V4L2_MAX_POLL_FD__, 2000);
        if(ret < 0){
            printf("poll fd with error\n");
            break;
        }
        else if(ret == 0){
            printf("poll fd time out\n");
            continue;
        }else{
            if(pfds[0].revents & POLLIN){
                pframe = pv4l2->op->get_frame(pv4l2,&pv4l2->capture_free_queue);
                if(!pframe){
                    printf("poll in no frame\n");
                    continue;
                }
                ret = ioctl(pv4l2->fd, VIDIOC_DQBUF, &pframe->v4l2_buf);
                if(ret < 0){
                    printf("%s get capture frame failed %d\n", pv4l2->src, ret);
                    continue;
                }

                pv4l2->op->trigger_notify(pv4l2,V4L2_EVENT_FRAME_READY,pframe);
                //printf("poll frame success,length = %d\n",pframe->v4l2_buf.length);
                //local_frame_save(pframe->virt_addr[0],pframe->v4l2_buf.length);
                //printf("poll frame success,length = %d\n",pframe->v4l2_buf.m.planes[0].bytesused);
                //local_frame_save((uint8_t*)pframe->virt_addr[0],pframe->v4l2_buf.m.planes[0].bytesused);

                ret = ioctl(pv4l2->fd, VIDIOC_QBUF, &pframe->v4l2_buf);
                if(ret < 0){
                    printf("put buffers into queue failed %d\n", ret);
                    continue;
                }
                pv4l2->op->put_frame(pv4l2, &pv4l2->capture_free_queue, pframe);
            }
            if(pfds[0].revents & POLLOUT){
                printf("poll out\n");
            }
            if(pfds[0].revents & POLLERR){
                printf("poll error\n");
                break;
            }
        }
    }
    return NULL;
}

static void* v4l2_output_poll_thread(void* context)
{
    int ret;
    struct video_frame* pframe;
    struct v4l2* pv4l2;
    struct pollfd pfds[__V4L2_MAX_POLL_FD__];

    pv4l2 = (struct v4l2*)context;
    if(!pv4l2){
        return NULL;
    }

    while(pv4l2->running_flag){
        memset(pfds,0,sizeof(struct pollfd) * __V4L2_MAX_POLL_FD__);
        pfds[0].fd = pv4l2->fd;
        pfds[0].events = POLLIN | POLLOUT | POLLERR;
        ret = poll(pfds, __V4L2_MAX_POLL_FD__, 1000);
        if(ret < 0){
            printf("poll fd with error\n");
            break;
        }
        else if(ret == 0){
            printf("poll fd time out\n");
            continue;
        }else{
            if(pfds[0].revents & POLLIN){
                printf("poll in\n");
            }
            if(pfds[0].revents & POLLOUT){
                pframe = pv4l2->op->get_frame(pv4l2,&pv4l2->output_used_queue);
                if(!pframe){
                    printf("poll in no frame\n");
                    continue;
                }
                ret = ioctl(pv4l2->fd, VIDIOC_DQBUF, &pframe->v4l2_buf);
                if(ret < 0){
                    printf("%s get capture frame failed %d\n", pv4l2->src, ret);
                    continue;
                }

                printf("poll output frame success\n");

#if 0
                ret = ioctl(pv4l2->fd, VIDIOC_QBUF, &pframe->v4l2_buf);
                if(ret < 0){
                    printf("put buffers into queue failed %d\n", ret);
                    continue;
                }
#endif
                pv4l2->op->put_frame(pv4l2, &pv4l2->output_free_queue, pframe);
            }
            if(pfds[0].revents & POLLERR){
                printf("poll error\n");
                break;
            }
        }
    }
    return NULL;
}

#if 0
static void* v4l2_capture_select_thread(void* context)
{
    int ret;
    struct video_frame* pframe;
    struct v4l2* pv4l2;

    pv4l2 = (struct v4l2*)context;
    if(!pv4l2){
        return NULL;
    }

    while(pv4l2->running_flag){
        struct timeval tv;
        tv.tv_sec = 3;
        tv.tv_usec = 0;
        fd_set fdread;
        FD_ZERO(&fdread);
        FD_SET(pv4l2->fd, &fdread);
        ret = select(FD_SETSIZE, &fdread, NULL, NULL, &tv);
        if(ret < 0){
            printf("select error\n");
            break;
        }else if(ret == 0){
            printf("select no data\n");
            continue;
        }
        if(FD_ISSET(pv4l2->fd, &fdread)){
            printf("select data\n");
        }
    }
    return NULL;
}
#endif

static int32_t v4l2_init(struct v4l2* pv4l2)
{
    INIT_LIST_HEAD(&(pv4l2->head));
    lock_init(&(pv4l2->lock));

    INIT_LIST_HEAD(&(pv4l2->capture_free_queue));
    INIT_LIST_HEAD(&(pv4l2->capture_used_queue));

    INIT_LIST_HEAD(&(pv4l2->output_free_queue));
    INIT_LIST_HEAD(&(pv4l2->output_used_queue));

    return 0;
}

static int32_t v4l2_release(struct v4l2* pv4l2)
{
    int j;
    struct video_frame* pframe;
    pv4l2->op->stop_input(pv4l2);
    pv4l2->op->stop_output(pv4l2);
    while(!list_empty(&pv4l2->capture_used_queue)){
        pframe = (struct video_frame*)list_first_entry(&pv4l2->capture_used_queue, struct video_frame, head);
        list_del(&pframe->head);
        list_add_tail(&pframe->head,&pv4l2->capture_free_queue);
    }
    while(!list_empty(&pv4l2->capture_free_queue)){
        pframe = (struct video_frame*)list_first_entry(&pv4l2->capture_free_queue, struct video_frame, head);
        list_del(&pframe->head);
        if(pv4l2->capture_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE){
            for(j = 0; j < pframe->v4l2_buf.length; j++){
                munmap(pframe->virt_addr[j], (pframe->v4l2_buf.m.planes[j].length));
            }
        }else{
            j = 0;
            munmap(pframe->virt_addr[j], (pframe->v4l2_buf.length));
        }
        free(pframe);
    }

    while(!list_empty(&pv4l2->output_used_queue)){
        pframe = (struct video_frame*)list_first_entry(&pv4l2->output_used_queue, struct video_frame, head);
        list_del(&pframe->head);
        list_add_tail(&pframe->head,&pv4l2->output_free_queue);
    }
    while(!list_empty(&pv4l2->output_free_queue)){
        pframe = (struct video_frame*)list_first_entry(&pv4l2->output_free_queue, struct video_frame, head);
        list_del(&pframe->head);
        if(pv4l2->capture_type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE){
            for(j = 0; j < pframe->v4l2_buf.length; j++){
                munmap(pframe->virt_addr[j], (pframe->v4l2_buf.m.planes[j].length));
            }
        }else{
            j = 0;
            munmap(pframe->virt_addr[j], (pframe->v4l2_buf.length));
        }
        free(pframe);
    }
    pv4l2->op->close(pv4l2);
    lock_destroy((&pv4l2->lock));
    return 0;
}

static int32_t v4l2_config(struct v4l2* pv4l2,char* src,int width,int height)
{
    if(!src){
        return -EINVAL;
    }
    strcpy(pv4l2->src,src);
    pv4l2->width = width;
    pv4l2->height = height;
    return 0;
}

static int32_t v4l2_open(struct v4l2* pv4l2)
{
    int ret;
    struct v4l2_capability cap;

    if(pv4l2->fd <= 0){
        pv4l2->fd = open(pv4l2->src,O_RDWR | O_NONBLOCK, 0);
        if(pv4l2->fd <= 0){
            printf("open v4l2 device %s error\n",pv4l2->src);
            return -ENODEV;
        }
    }
    ret = ioctl(pv4l2->fd, VIDIOC_QUERYCAP, &cap);
    if (ret < 0) {
        printf("unable to query device: %s (%d)\n", strerror(errno),errno);
        close(pv4l2->fd);
        return ret;
    }
    printf("device is %s on bus %s\n", cap.card, cap.bus_info);

    if(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE){
        printf("type is V4L2_CAP_VIDEO_CAPTURE\n");
        pv4l2->capture_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    }
    if(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE){
        printf("type is V4L2_CAP_VIDEO_CAPTURE_MPLANE\n");
        pv4l2->capture_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    }
    if(cap.capabilities & V4L2_CAP_VIDEO_OUTPUT){
        printf("type is V4L2_CAP_VIDEO_CAPTURE\n");
        pv4l2->output_type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    }
    if(cap.capabilities & V4L2_CAP_VIDEO_OUTPUT_MPLANE){
        printf("type is V4L2_CAP_VIDEO_OUTPUT_MPLANE\n");
        pv4l2->output_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    }

    return 0;
}

static int32_t v4l2_close(struct v4l2* pv4l2)
{
    if(pv4l2->fd > 0){
        close(pv4l2->fd);
        pv4l2->fd = 0;
    }
    return 0;
}

static int32_t v4l2_prepare_input(struct v4l2* pv4l2)
{
    int i,j;
    int ret;
    struct v4l2_fmtdesc fmtdesc;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;

    if(!pv4l2->capture_type){
        printf("not capture type\n");
        return -EFAULT;
    }
#if 1
    fmtdesc.index = 0;
    fmtdesc.type = pv4l2->capture_type;
    while(1){
        ret = ioctl(pv4l2->fd, VIDIOC_ENUM_FMT, &fmtdesc);
        if(ret != -1){
            printf("no format found\n");
            break;
        }
        printf("%d:%s\n",fmtdesc.index,fmtdesc.description);
        fmtdesc.index++;
    }
#endif

#if 1
    memset(&fmt,0,sizeof(struct v4l2_format));
    fmt.type = pv4l2->capture_type;
    ret = ioctl(pv4l2->fd, VIDIOC_G_FMT, &fmt);
    if(ret < 0){
        printf("get v4l2 fmt error\n");
        return ret;
    }

    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    pv4l2->capture_pix_fmt = fmt.fmt.pix.pixelformat;
    fmt.fmt.pix_mp.width = pv4l2->width;
    fmt.fmt.pix_mp.height = pv4l2->height;

    if(pv4l2->capture_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE){
        fmt.fmt.pix_mp.num_planes = 1;
        fmt.fmt.pix_mp.plane_fmt[0].bytesperline = ((fmt.fmt.pix_mp.width)+15)>>4<<4;
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage = (fmt.fmt.pix_mp.plane_fmt[0].bytesperline * (fmt.fmt.pix_mp.height));
    }else{

    }

    ret = ioctl(pv4l2->fd, VIDIOC_S_FMT, &fmt);
    if(ret < 0){
        printf("set video format failed %d\n", ret);
        return ret;
    }
#endif

    req.count = __MAX_REQ_BUF_COUNT__;
    req.type = pv4l2->capture_type;
    req.memory = V4L2_MEMORY_MMAP;
    ret = ioctl(pv4l2->fd, VIDIOC_REQBUFS, &req);
    if(ret < 0){
        printf("request buffer error\n");
    }
    printf("request buffer count = %d\n",req.count);
    for(i = 0; i < req.count; i++){
        struct video_frame* pframe;
        pframe = (struct video_frame*)malloc(sizeof(struct video_frame));
        if(!pframe){
            printf("malloc frame error\n");
            continue;
        }
        memset(pframe,0,sizeof(struct video_frame));

        pframe->v4l2_buf.type = pv4l2->capture_type;
        pframe->v4l2_buf.memory = V4L2_MEMORY_MMAP;
        pframe->v4l2_buf.index = i;
        pframe->height = pv4l2->height;
        pframe->width = pv4l2->width;
        pframe->stride = ((fmt.fmt.pix_mp.width)+15)>>4<<4;
        if(pv4l2->capture_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE){
            pframe->v4l2_buf.length = 1;
            pframe->v4l2_buf.m.planes = pframe->planes;
        }
        ret = ioctl(pv4l2->fd, VIDIOC_QUERYBUF, &pframe->v4l2_buf);
        if(ret < 0){
            printf("query driver buffer %d failed %d\n", i, ret);
        }
        printf("buffer len = %d\n",pframe->v4l2_buf.length);
        if(pv4l2->capture_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE){
            for(j = 0; j < pframe->v4l2_buf.length; j++){
                pframe->v4l2_buf.m.planes[j].bytesused = pframe->v4l2_buf.m.planes[j].length;
                pframe->v4l2_buf.m.planes[j].data_offset = 0;

                pframe->phys_addr[j] = (void*)pframe->planes[j].m.mem_offset;
                pframe->virt_addr[j] = mmap(NULL, (pframe->v4l2_buf.m.planes[j].length), PROT_READ|PROT_WRITE, MAP_SHARED, pv4l2->fd, (long )pframe->phys_addr[j]);
                if(pframe->virt_addr[j] == MAP_FAILED){
                    printf("mmap %s output planes 0 buffer %d failed\n", pv4l2->src, i);
                    perror("mmap");
                }
            }
        }else{
            j = 0;
            pframe->phys_addr[j] = (void*)pframe->v4l2_buf.m.offset;
            pframe->virt_addr[j] = mmap(NULL, (pframe->v4l2_buf.length), PROT_READ|PROT_WRITE, MAP_SHARED, pv4l2->fd, (long )pframe->phys_addr[j]);
            if(pframe->virt_addr[j] == MAP_FAILED){
                printf("mmap %s output planes 0 buffer %d failed\n", pv4l2->src, i);
                perror("mmap");
            }
        }
        ret = ioctl(pv4l2->fd, VIDIOC_QBUF, &pframe->v4l2_buf);
        if(ret < 0){
            printf("put buffers into queue failed %d\n", ret);
            continue;
        }
        pv4l2->op->put_frame(pv4l2, &pv4l2->capture_free_queue, pframe);
    }
    return 0;
}

static int32_t v4l2_start_input(struct v4l2* pv4l2)
{
    int ret;
    ret = ioctl(pv4l2->fd, VIDIOC_STREAMON, &pv4l2->capture_type);
    if(ret < 0){
        printf("device %s enable capture failed %d\n", pv4l2->src, ret);
        return ret;
    }
    pv4l2->running_flag = 1;
    ret = pthread_create(&pv4l2->pid,NULL,v4l2_capture_poll_thread,pv4l2);
    if(ret < 0){
        printf("create thread error\n");
        return ret;
    }
    return 0;
}

static int32_t v4l2_stop_input(struct v4l2* pv4l2)
{
    int ret;
    ret = ioctl(pv4l2->fd, VIDIOC_STREAMOFF, &pv4l2->capture_type);
    if(ret < 0){
        printf("device %s disable capture failed %d\n", pv4l2->src, ret);
        return ret;
    }
    if(pv4l2->running_flag){
        pv4l2->running_flag = 0;
        pthread_join(pv4l2->pid,NULL);
    }
    return 0;
}

static int32_t v4l2_prepare_output(struct v4l2* pv4l2)
{
    int i,j;
    int ret;
    struct v4l2_fmtdesc fmtdesc;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;

    if(!pv4l2->output_type){
        printf("not capture type\n");
        return -EFAULT;
    }

#if 1
    fmtdesc.index = 0;
    fmtdesc.type = pv4l2->capture_type;
    while(1){
        ret = ioctl(pv4l2->fd, VIDIOC_ENUM_FMT, &fmtdesc);
        if(ret != -1){
            printf("no format found\n");
            break;
        }
        printf("%d:%s\n",fmtdesc.index,fmtdesc.description);
        fmtdesc.index++;
    }
#endif
#if 1
    memset(&fmt,0,sizeof(struct v4l2_format));
    fmt.type = pv4l2->capture_type;
    ret = ioctl(pv4l2->fd, VIDIOC_G_FMT, &fmt);
    if(ret < 0){
        printf("get v4l2 fmt error\n");
        return ret;
    }
    fmt.fmt.pix_mp.width = pv4l2->width;
    fmt.fmt.pix_mp.height = pv4l2->height;

    if(pv4l2->output_type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE){
        fmt.fmt.pix_mp.num_planes = 1;
        fmt.fmt.pix_mp.plane_fmt[0].bytesperline = ((fmt.fmt.pix_mp.width)+15)>>4<<4;
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage = (fmt.fmt.pix_mp.plane_fmt[0].bytesperline * (fmt.fmt.pix_mp.height));
    }else{

    }
    ret = ioctl(pv4l2->fd, VIDIOC_S_FMT, &fmt);
    if(ret < 0){
        printf("set video format failed %d\n", ret);
        return ret;
    }
#endif

    req.count = __MAX_REQ_BUF_COUNT__;
    req.type = pv4l2->output_type;
    req.memory = V4L2_MEMORY_MMAP;
    ret = ioctl(pv4l2->fd, VIDIOC_REQBUFS, &req);
    if(ret < 0){
        printf("request buffer error\n");
    }
    printf("request buffer count = %d\n",req.count);
    for(i = 0; i < req.count; i++){
        struct video_frame* pframe;
        pframe = (struct video_frame*)malloc(sizeof(struct video_frame));
        if(!pframe){
            printf("malloc frame error\n");
            continue;
        }
        memset(pframe,0,sizeof(struct video_frame));

        pframe->v4l2_buf.type = pv4l2->output_type;
        pframe->v4l2_buf.memory = V4L2_MEMORY_MMAP;
        pframe->v4l2_buf.index = i;
        pframe->height = pv4l2->height;
        pframe->width = pv4l2->width;
        pframe->stride = ((fmt.fmt.pix_mp.width)+15)>>4<<4;
        if(pv4l2->output_type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE){
            pframe->v4l2_buf.length = 1;
            pframe->v4l2_buf.m.planes = pframe->planes;
        }
        ret = ioctl(pv4l2->fd, VIDIOC_QUERYBUF, &pframe->v4l2_buf);
        if(ret < 0){
            printf("query driver buffer %d failed %d\n", i, ret);
        }
        printf("buffer len = %d\n",pframe->v4l2_buf.length);
        if(pv4l2->output_type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE){
            for(j = 0; j < pframe->v4l2_buf.length; j++){
                pframe->v4l2_buf.m.planes[j].bytesused = pframe->v4l2_buf.m.planes[j].length;
                pframe->v4l2_buf.m.planes[j].data_offset = 0;

                pframe->phys_addr[j] = (void*)pframe->planes[j].m.mem_offset;
                pframe->virt_addr[j] = mmap(NULL, (pframe->v4l2_buf.m.planes[j].length), PROT_READ|PROT_WRITE, MAP_SHARED, pv4l2->fd, (long )pframe->phys_addr[j]);
                if(pframe->virt_addr[j] == MAP_FAILED){
                    printf("mmap %s output planes 0 buffer %d failed\n", pv4l2->src, i);
                    perror("mmap");
                }
            }
        }else{
            j = 0;
            pframe->phys_addr[j] = (void*)pframe->v4l2_buf.m.offset;
            pframe->virt_addr[j] = mmap(NULL, (pframe->v4l2_buf.length), PROT_READ|PROT_WRITE, MAP_SHARED, pv4l2->fd, (long )pframe->phys_addr[j]);
            if(pframe->virt_addr[j] == MAP_FAILED){
                printf("mmap %s output planes 0 buffer %d failed\n", pv4l2->src, i);
                perror("mmap");
            }
        }
        ret = ioctl(pv4l2->fd, VIDIOC_QBUF, &pframe->v4l2_buf);
        if(ret < 0){
            printf("put buffers into queue failed %d\n", ret);
            continue;
        }
        pv4l2->op->put_frame(pv4l2, &pv4l2->output_used_queue, pframe);
    }
    return 0;
}

static int32_t v4l2_start_output(struct v4l2* pv4l2)
{
    int ret;
    ret = ioctl(pv4l2->fd, VIDIOC_STREAMON, &pv4l2->output_type);
    if(ret < 0){
        printf("device %s enable capture failed %d\n", pv4l2->src, ret);
        return ret;
    }
    pv4l2->running_flag = 1;
    ret = pthread_create(&pv4l2->pid,NULL,v4l2_output_poll_thread,pv4l2);
    if(ret < 0){
        printf("create thread error\n");
        return ret;
    }
    return 0;
}

static int32_t v4l2_stop_output(struct v4l2* pv4l2)
{
    int ret;
    ret = ioctl(pv4l2->fd, VIDIOC_STREAMOFF, &pv4l2->output_type);
    if(ret < 0){
        printf("device %s disable capture failed %d\n", pv4l2->src, ret);
        return ret;
    }
    if(pv4l2->running_flag){
        pv4l2->running_flag = 0;
        pthread_join(pv4l2->pid,NULL);
    }
    return 0;
}

static int32_t v4l2_transfer_output(struct v4l2* pv4l2,uint8_t* data,uint32_t length)
{
    int ret;
    struct video_frame* pframe;
    pframe = NULL;
    pframe = pv4l2->op->get_frame(pv4l2, &pv4l2->output_free_queue);
    if(!pframe){
        printf("no buffer\n");
        return -ENOMEM;
    }

    if(pv4l2->output_type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE){
        memcpy(pframe->virt_addr[0],data,length);
    }else{
        memcpy(pframe->virt_addr[0],data,length);
    }

    ret = ioctl(pv4l2->fd, VIDIOC_QBUF, &pframe->v4l2_buf);
    if(ret < 0){
        printf("put buffers into queue failed %d\n", ret);
        return -EFAULT;
    }
    pv4l2->op->put_frame(pv4l2, &pv4l2->output_used_queue, pframe);
    return 0;
}

static int32_t v4l2_put_frame(struct v4l2* pv4l2,struct list_head* queue,struct video_frame* pframe)
{
    lock(&pv4l2->lock);
    list_add_tail(&pframe->head,queue);
    unlock(&pv4l2->lock);
    return 0;
}

static struct video_frame* v4l2_get_frame(struct v4l2* pv4l2,struct list_head* queue)
{
    struct video_frame* pframe;
    pframe = NULL;
    lock(&pv4l2->lock);
    if(!list_empty(queue)){
        pframe = (struct video_frame*)list_first_entry(queue, struct video_frame, head);
        list_del(&pframe->head);
    }
    unlock(&pv4l2->lock);
    return pframe;
}

static int32_t v4l2_register_notify(struct v4l2* pv4l2, int32_t event, v4l2_event_notify notify, void* object)
{
    struct v4l2_event_action* paction;
    if(!notify || (event <= V4L2_EVENT_NONE) || (event >= V4L2_EVENT_MAX)){
        return -EINVAL;
    }
    paction = (struct v4l2_event_action*)malloc(sizeof(struct v4l2_event_action));
    if(!paction){
        printf("malloc error\n");
        return -ENOMEM;
    }
    paction->notify = notify;
    paction->object = object;
    lock(&(pv4l2->lock));
    paction->next = pv4l2->paction[event];
    pv4l2->paction[event] = paction;
    unlock(&(pv4l2->lock));
    return 0;
}

static int32_t v4l2_unregister_notify(struct v4l2* pv4l2, int32_t event, void* object)
{
    struct v4l2_event_action *paction,* ptmp;
    if((event <= V4L2_EVENT_NONE) || (event >= V4L2_EVENT_MAX)){
        return -EINVAL;
    }
    lock(&(pv4l2->lock));
    paction = pv4l2->paction[event];
    if(paction->object == object){
        pv4l2->paction[event] = paction->next;
        free(paction);
    }else{
        while(paction->next){
            if(paction->next->object == object){
                ptmp = paction->next;
                paction->next = ptmp->next;
                free(ptmp);
                break;
            }
            paction = paction->next;
        }
    }
    unlock(&(pv4l2->lock));
    return 0;
}

static int32_t v4l2_trigger_notify(struct v4l2* pv4l2, int32_t event, void* context)
{
    struct v4l2_event_action* paction;
    if((event <= V4L2_EVENT_NONE) || (event >= V4L2_EVENT_MAX)){
        return -EINVAL;
    }
    paction = pv4l2->paction[event];
    while(paction){
        paction->notify(pv4l2, event, paction->object, context);
        paction = paction->next;
    }
    return 0;
}

static struct v4l2_operation v4l2_op =
{
    .init = v4l2_init,
    .release = v4l2_release,

    .config = v4l2_config,
    .open = v4l2_open,
    .close = v4l2_close,

    .prepare_input = v4l2_prepare_input,
    .start_input = v4l2_start_input,
    .stop_input = v4l2_stop_input,

    .prepare_output = v4l2_prepare_output,
    .start_output = v4l2_start_output,
    .stop_output = v4l2_stop_output,
    .transfer_output = v4l2_transfer_output,

    .put_frame = v4l2_put_frame,
    .get_frame = v4l2_get_frame,

    .register_notify = v4l2_register_notify,
    .unregister_notify = v4l2_unregister_notify,
    .trigger_notify = v4l2_trigger_notify,
};

int32_t create_init_v4l2(struct v4l2** pv4l2)
{
    int32_t ret;
    struct v4l2* ptmp;
    (*pv4l2) = (struct v4l2*)malloc(sizeof(struct v4l2));
    if(!(*pv4l2)){
        printf("malloc error\n");
        return -ENOMEM;
    }
    ptmp = *pv4l2;
    memset(ptmp,0,sizeof(struct v4l2));
    ptmp->op = &v4l2_op;
    ret = ptmp->op->init(ptmp);
    if(ret < 0){
        printf("init error\n");
        release_destroy_v4l2(ptmp);
        return ret;
    }
    return 0;
}

void release_destroy_v4l2(struct v4l2* pv4l2)
{
    if(pv4l2){
        pv4l2->op->release(pv4l2);
        free(pv4l2);
    }
}

