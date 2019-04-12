#ifndef __LOCAL_FFMPEG_H__
#define __LOCAL_FFMPEG_H__

#include <stdint.h>
#include "utils/list.h"
#include "utils/lock.h"
extern "C"{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavdevice/avdevice.h>
    #include <libswresample/swresample.h>
    #include <libavutil/opt.h>
    #include <libavutil/channel_layout.h>
    #include <libavutil/samplefmt.h>
    #include <libavutil/imgutils.h>
};

struct local_ffmpeg;

enum LOCAL_FFMPEG_EVENT
{
    LOCAL_FFMPEG_EVENT_NONE = 0,
    LOCAL_FFMPEG_EVENT_AUDIO_FRAME_READY,
    LOCAL_FFMPEG_EVENT_VIDEO_FRAME_READY,
    LOCAL_FFMPEG_EVENT_MAX,
};

typedef int32_t (*local_ffmpeg_event_notify)(struct local_ffmpeg*, int32_t, void*, void*);

struct local_ffmpeg_event_action
{
    local_ffmpeg_event_notify notify;
    void* object;
    struct local_ffmpeg_event_action* next;
};

struct local_ffmpeg_operation
{
    int32_t (*init)(struct local_ffmpeg*);
    int32_t (*release)(struct local_ffmpeg*);

    int32_t (*query_local_camera)(struct local_ffmpeg*);
    int32_t (*query_local_micro)(struct local_ffmpeg*);

    int32_t (*config_video_src)(struct local_ffmpeg*, char*);
    int32_t (*encode_video)(struct local_ffmpeg*);
    int32_t (*encode_audio)(struct local_ffmpeg*);

    int32_t (*audio_convert_init)(struct local_ffmpeg*, int src_ch_layout, int src_rate, AVSampleFormat src_sample_fmt, int dst_ch_layout, int dst_rate, AVSampleFormat dst_sample_fmt);
    int32_t (*audio_convert_frame)(struct local_ffmpeg*, uint8_t* src_data, int src_linesize);

    int32_t (*audio_packet_encode)(struct local_ffmpeg*, uint8_t* data, int size);

    int32_t (*video_convert_init)(struct local_ffmpeg*,int src_width,int src_height, AVPixelFormat src_pixel_fmt,int dst_width,int dst_height, AVPixelFormat dst_pixel_fmt);
    int32_t (*video_convert_frame)(struct local_ffmpeg*,uint8_t* src,uint8_t* dst,int dvalue);

    int32_t (*register_notify)(struct local_ffmpeg*, int32_t, local_ffmpeg_event_notify notify, void*);
    int32_t (*unregister_notify)(struct local_ffmpeg*, int32_t, void*);
    int32_t (*trigger_notify)(struct local_ffmpeg*, int32_t, void*);
};

struct local_ffmpeg
{
    struct list_head head;
    lock_t lock;

#define __MAX_DEVICE_COUNT__ 16
    int camera_count;
    char camera_name[__MAX_DEVICE_COUNT__][256];

    int micro_count;
    char micro_name[__MAX_DEVICE_COUNT__][256];

    char video_src[512];
    char audio_src[512];
    int video_index;
    int audio_index;

    AVFormatContext *ifmt_ctx;
    AVInputFormat *ifmt;
    AVStream* i_video_stream;
    AVStream* i_audio_stream;

    AVCodecContext	*pVideoCodecCtx;
    AVCodec			*pVideoCodec;
    AVCodecContext	*pAudioCodecCtx;
    AVCodec			*pAudioCodec;

    AVCodecContext  *pVideoEncodeCodecCtx;
    AVCodec         *pVideoEncodeCodec;
    AVCodecContext  *pAudioEncodeCodecCtx;
    AVCodec         *pAudioEncodeCodec;

    struct SwrContext *audio_convert_ctx;
    AVSampleFormat audio_dst_sample_fmt;
    AVSampleFormat audio_src_sample_fmt;
    int audio_dst_bufsize;

    int audio_src_nb_samples;
    int audio_src_nb_channels;
    uint8_t **audio_src_data;
    int audio_src_linesize;

    int audio_dst_nb_samples;
    int audio_dst_nb_channels;
    uint8_t **audio_dst_data;
    int audio_dst_linesize;

    struct SwsContext *pvideoSwsContext;
    int video_src_width;
    int video_src_height;
    AVPixelFormat video_src_pixel_fmt;
    AVFrame* video_src_frame;

    int video_dst_width;
    int video_dst_height;
    AVPixelFormat video_dst_pixel_fmt;
    AVFrame* video_dst_frame;

    int running_flag;
    pthread_t pid;

    struct local_ffmpeg_operation* op;
    struct local_ffmpeg_event_action *paction[LOCAL_FFMPEG_EVENT_MAX];
};

int32_t create_init_local_ffmpeg(struct local_ffmpeg** plocal_ffmpeg);
void release_destroy_local_ffmpeg(struct local_ffmpeg* plocal_ffmpeg);

#endif
