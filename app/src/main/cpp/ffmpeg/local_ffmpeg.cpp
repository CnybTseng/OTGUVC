#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

#include "local_ffmpeg.h"
#include "ffmpeg_log.h"

static void* local_ffmpeg_encode_video_thread(void* context)
{
    int ret;
    int picture_size;
    AVPacket *packet;
    AVFrame	*pframe;
    AVPacket encodePacket;
    AVFrame	*pframeyuv;
    uint8_t* yuvbuffer;
    struct SwsContext *pSwsContext;

    struct local_ffmpeg* pffmpeg;
    AVCodecContext	*pVideoCodecCtx;
    AVCodecContext  *pVideoEncodeCodecCtx;
    AVFormatContext *ifmt_ctx;

    pffmpeg = (struct local_ffmpeg*)context;
    if (!pffmpeg) {
        printf("error param when thread\n");
        return NULL;
    }
    pVideoCodecCtx = pffmpeg->pVideoCodecCtx;
    pVideoEncodeCodecCtx = pffmpeg->pVideoEncodeCodecCtx;
    ifmt_ctx = pffmpeg->ifmt_ctx;

    picture_size = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, pVideoCodecCtx->width, pVideoCodecCtx->height, 1);
    pframeyuv = av_frame_alloc();
    yuvbuffer = (uint8_t *)av_malloc(picture_size);
    av_image_fill_arrays(pframeyuv->data, pframeyuv->linesize, yuvbuffer, AV_PIX_FMT_YUV420P, pVideoCodecCtx->width, pVideoCodecCtx->height, 1);
    pSwsContext = sws_getContext(pVideoCodecCtx->width, pVideoCodecCtx->height, pVideoCodecCtx->pix_fmt, pVideoEncodeCodecCtx->width, pVideoEncodeCodecCtx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);

    packet = (AVPacket *)av_malloc(sizeof(AVPacket));
    av_new_packet(&encodePacket, picture_size);

    pframe = av_frame_alloc();
    while (pffmpeg->running_flag) {
        ret = av_read_frame(ifmt_ctx, packet);
        if (ret >= 0) {
            if (packet->stream_index == pffmpeg->video_index) {
                ret = avcodec_send_packet(pVideoCodecCtx, packet);
                if (ret < 0) {
                    printf("Send Decode Error\n");
                    continue;
                }
                ret = avcodec_receive_frame(pVideoCodecCtx, pframe);
                if (ret != 0) {
                    printf("Decode Error\n");
                    continue;
                }
                ret = sws_scale(pSwsContext, (const uint8_t* const*)pframe->data, pframe->linesize, 0, pVideoCodecCtx->height, pframeyuv->data, pframeyuv->linesize);
                if (pVideoEncodeCodecCtx) {
                    pframeyuv->width = pVideoEncodeCodecCtx->width;
                    pframeyuv->height = pVideoEncodeCodecCtx->height;
                    pframeyuv->format = AV_PIX_FMT_YUV420P;
                    ret = avcodec_send_frame(pVideoEncodeCodecCtx, pframeyuv);
                    if (ret < 0) {
                        printf("Send Encode Error\n");
                        continue;
                    }
                    ret = avcodec_receive_packet(pVideoEncodeCodecCtx, &encodePacket);
                    if (ret < 0) {
                        printf("Encode Error\n");
                        continue;
                    }
                    pffmpeg->op->trigger_notify(pffmpeg, LOCAL_FFMPEG_EVENT_VIDEO_FRAME_READY, &encodePacket);
                    av_packet_unref(&encodePacket);
                }
            }
            av_packet_unref(packet);
        }
        else if (ret == AVERROR(EAGAIN)) {
            usleep(10 * 1000);
        }
        else {
            printf("read packet error\n");
            break;
        }
    }

    sws_freeContext(pSwsContext);
    av_free(yuvbuffer);
    av_free(pframeyuv);

    av_free(pframe);
    av_free(packet);
    return NULL;
}


#define PCMA_MAX 32635
static unsigned char pcma_encode(short pcm)
{
    int exponent;
    int expMask;

    int sign = (pcm & 0x8000) >> 8;
    if (sign != 0) {
        pcm = -pcm;
    }
    if (pcm > PCMA_MAX) {
        pcm = PCMA_MAX;
    }
    exponent = 7;
    for (expMask = 0x4000; (pcm & expMask) == 0 && exponent > 0; exponent--, expMask >>= 1) {

    }
    int mantissa = (pcm >> ((exponent == 0) ? 4 : (exponent + 3))) & 0x0f;
    unsigned char alaw = (unsigned char)(sign | exponent << 4 | mantissa);
    return (unsigned char)(alaw ^ 0xD5);
}

static void local_audio_test_save_raw_file(uint8_t* data, int length)
{
    char name[64];
    FILE* fp;
    sprintf(name, "D:\\local_pcm.pcm");
    fp = fopen(name, "ab+");
    if (fp) {
        fseek(fp, 0L, SEEK_END);
        fwrite(data, length, 1, fp);
        fclose(fp);
    }
}

static void* local_ffmpeg_encode_audio_thread(void* context)
{
    int ret;
    AVPacket *packet;
    AVFrame	*pframe;
    AVPacket encodePacket;

    struct local_ffmpeg* pffmpeg;
    AVCodecContext	*pAudioCodecCtx;
    AVFormatContext *ifmt_ctx;

    AVPacket src_packet;

    pffmpeg = (struct local_ffmpeg*)context;
    if (!pffmpeg) {
        printf("error param when thread\n");
        return NULL;
    }
    pAudioCodecCtx = pffmpeg->pAudioCodecCtx;
    ifmt_ctx = pffmpeg->ifmt_ctx;

    packet = (AVPacket *)av_malloc(sizeof(AVPacket));
    av_new_packet(&encodePacket, 44100 * 4);

    pframe = av_frame_alloc();
    while (pffmpeg->running_flag) {
        ret = av_read_frame(ifmt_ctx, packet);
        if (ret >= 0) {
            if (packet->stream_index == pffmpeg->audio_index) {
                //local_audio_test_save_raw_file(packet->data, packet->size);
#if 0
                ret = avcodec_send_packet(pAudioCodecCtx, packet);
				if (ret < 0) {
					printf("Send Decode Error\n");
					continue;
				}
				ret = avcodec_receive_frame(pAudioCodecCtx, pframe);
				if (ret != 0) {
					printf("Decode Error\n");
					continue;
				}
#endif
                ret = pffmpeg->op->audio_convert_init(pffmpeg, pAudioCodecCtx->channels, pAudioCodecCtx->sample_rate, pAudioCodecCtx->sample_fmt, 1, 8000, pAudioCodecCtx->sample_fmt);
                if (ret != 0) {
                    printf("convert init error\n");
                    continue;
                }
                //pffmpeg->op->trigger_notify(pffmpeg, LOCAL_FFMPEG_EVENT_AUDIO_FRAME_READY, pframe);
                ret = pffmpeg->op->audio_convert_frame(pffmpeg, packet->data, packet->size);
                if(ret > 0){
#if 0
                    int i;
					short* buffer = (short*)pffmpeg->audio_dst_data[0];
					for (i = 0; i < ret / 2; i++) {
						packet->data[i] = pcma_encode(buffer[i]);
					}
					packet->size = ret/2;
					local_audio_test_save_raw_file(packet->data, packet->size);
					pffmpeg->op->trigger_notify(pffmpeg, LOCAL_FFMPEG_EVENT_AUDIO_FRAME_READY, packet);
#endif
                    pffmpeg->op->audio_packet_encode(pffmpeg, pffmpeg->audio_dst_data[0], ret);
                }
            }
            av_packet_unref(packet);
        }
        else if (ret == AVERROR(EAGAIN)) {
            usleep(10 * 1000);
        }
        else {
            printf("read packet error\n");
            break;
        }
    }
    av_free(pframe);
    av_free(packet);
    return NULL;
}

static int32_t local_ffmpeg_init(struct local_ffmpeg* pffmpeg)
{
    //av_register_all();
    avdevice_register_all();
    avformat_network_init();
    av_log_set_callback(ffmpeg_av_log_default_callback);
    av_log_set_level(AV_LOG_INFO);

    INIT_LIST_HEAD(&(pffmpeg->head));
    lock_init(&(pffmpeg->lock));

    pffmpeg->audio_index = -1;
    pffmpeg->video_index = -1;

    pffmpeg->op->query_local_micro(pffmpeg);
    if (pffmpeg->micro_count > 0) {
        sprintf(pffmpeg->audio_src, "audio=%s", pffmpeg->micro_name[0]);
    }

    pffmpeg->op->query_local_camera(pffmpeg);
    if (pffmpeg->camera_count > 0) {
        sprintf(pffmpeg->video_src, "video=%s", pffmpeg->camera_name[0]);
    }
    return 0;
}

static int32_t local_ffmpeg_release(struct local_ffmpeg* pffmpeg)
{
    if (pffmpeg->running_flag) {
        pffmpeg->running_flag = 0;
        pthread_join(pffmpeg->pid, NULL);
    }
    if (pffmpeg->audio_src_data) {
        av_freep(&pffmpeg->audio_src_data[0]);
    }
    av_freep(&pffmpeg->audio_src_data);

    if (pffmpeg->audio_dst_data) {
        av_freep(&pffmpeg->audio_dst_data[0]);
    }
    av_freep(&pffmpeg->audio_dst_data);

    if (pffmpeg->audio_convert_ctx) {
        swr_free(&pffmpeg->audio_convert_ctx);
    }
    if (pffmpeg->ifmt_ctx) {
        avformat_close_input(&pffmpeg->ifmt_ctx);
    }
    if (pffmpeg->pAudioEncodeCodecCtx) {
        avcodec_free_context(&pffmpeg->pAudioEncodeCodecCtx);
    }
    if (pffmpeg->pAudioEncodeCodecCtx) {
        avcodec_free_context(&pffmpeg->pAudioEncodeCodecCtx);
    }
    lock_destroy((&pffmpeg->lock));
    return 0;
}


static int32_t local_ffmpeg_query_local_camera(struct local_ffmpeg* pffmpeg)
{
    return 0;
}

static int32_t local_ffmpeg_query_local_micro(struct local_ffmpeg* pffmpeg)
{
    return 0;
}

static int32_t local_ffmpeg_config_video_src(struct local_ffmpeg* pffmpeg,char* src)
{
    strcpy(pffmpeg->video_src,src);
    return 0;
}

static int32_t local_ffmpeg_encode_video(struct local_ffmpeg* pffmpeg)
{
    int ret;
    unsigned int i;
    AVDictionary* options = NULL;
    AVFormatContext *ifmt_ctx;
    AVInputFormat *ifmt;

    av_dict_set(&options, "video_size", "640x480", 0);
    //av_dict_set(&options, "framerate", "25", 0);

    pffmpeg->ifmt_ctx = avformat_alloc_context();
    if (!pffmpeg->ifmt_ctx) {
        printf("alloc input format error\n");
        pffmpeg->ifmt_ctx = NULL;
        return -EINVAL;
    }
    pffmpeg->ifmt = av_find_input_format("video4linux2");
    //pffmpeg->ifmt = av_find_input_format("vfwcap");
    //pffmpeg->ifmt = av_find_input_format("dshow");

    ifmt_ctx = pffmpeg->ifmt_ctx;
    ifmt = pffmpeg->ifmt;

    ret = avformat_open_input(&ifmt_ctx, pffmpeg->video_src, ifmt, &options);
    if (ret < 0) {
        printf("can not open input stream\n");
        return -EFAULT;
    }
    ret = avformat_find_stream_info(ifmt_ctx, NULL);
    if (ret < 0) {
        printf("can not find stream\n");
        return -EFAULT;
    }
    for (i = 0; i < ifmt_ctx->nb_streams; i++) {
        if (ifmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            pffmpeg->i_video_stream = ifmt_ctx->streams[i];
            pffmpeg->video_index = i;
        }
    }
    if (pffmpeg->video_index < 0) {
        printf("can not find stream\n");
        return -EFAULT;
    }
    pffmpeg->pVideoCodecCtx = avcodec_alloc_context3(NULL);
    if (pffmpeg->pVideoCodecCtx) {
        avcodec_parameters_to_context(pffmpeg->pVideoCodecCtx, ifmt_ctx->streams[pffmpeg->video_index]->codecpar);
    }
    pffmpeg->pVideoCodec = avcodec_find_decoder(pffmpeg->pVideoCodecCtx->codec_id);
    if (pffmpeg->pVideoCodec == NULL) {
        printf("Codec not found\n");
        return -EFAULT;
    }
    if (avcodec_open2(pffmpeg->pVideoCodecCtx, pffmpeg->pVideoCodec, NULL) < 0) {
        printf("Could not open codec\n");
        return -EFAULT;
    }
#if 0
    //pffmpeg->pVideoEncodeCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
    pffmpeg->pVideoEncodeCodec = avcodec_find_encoder(AV_CODEC_ID_VP8);
    if (!pffmpeg->pVideoEncodeCodec) {
        printf("Could not open h264 codec\n");
        return -EFAULT;
    }
    pffmpeg->pVideoEncodeCodecCtx = avcodec_alloc_context3(pffmpeg->pVideoEncodeCodec);
    if (!pffmpeg->pVideoEncodeCodecCtx) {
        printf("Could not open h264 codec context\n");
        return -EFAULT;
    }
    pffmpeg->pVideoEncodeCodecCtx->width = 320;//֡��
    pffmpeg->pVideoEncodeCodecCtx->height = 256;
    pffmpeg->pVideoEncodeCodecCtx->bit_rate = 300000;//������
    pffmpeg->pVideoEncodeCodecCtx->time_base.num = 1;
    pffmpeg->pVideoEncodeCodecCtx->time_base.den = 25;
    pffmpeg->pVideoEncodeCodecCtx->gop_size = 25;
    pffmpeg->pVideoEncodeCodecCtx->max_b_frames = 1;
    pffmpeg->pVideoEncodeCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    if (avcodec_open2(pffmpeg->pVideoEncodeCodecCtx, pffmpeg->pVideoEncodeCodec, NULL) < 0) {
        printf("avcodec_open2 ERR\n");
        return -EFAULT;
    }
#endif
    pffmpeg->running_flag = 1;
    ret = pthread_create(&(pffmpeg->pid), NULL, local_ffmpeg_encode_video_thread, pffmpeg);
    if (ret < 0) {
        printf("create video thread error %d \n", ret);
        perror("error:");
        return -EINVAL;
    }
    return 0;
}

static int32_t local_ffmpeg_encode_audio(struct local_ffmpeg* pffmpeg)
{
    int ret;
    unsigned int i;
    AVDictionary* options = NULL;
    AVFormatContext *ifmt_ctx;
    AVInputFormat *ifmt;

    av_dict_set_int(&options, "sample_rate", (long)44100, 0);
    av_dict_set_int(&options, "sample_size", (long)16, 0);
    //av_dict_set_int(&options, "audio_buffer_size", (long)44100 * 2 * (16 / 8) / 25 / 1000, 0);

    pffmpeg->ifmt_ctx = avformat_alloc_context();
    if (!pffmpeg->ifmt_ctx) {
        printf("alloc input format error\n");
        pffmpeg->ifmt_ctx = NULL;
        return -EINVAL;
    }
    pffmpeg->ifmt = av_find_input_format("dshow");
    ifmt_ctx = pffmpeg->ifmt_ctx;
    ifmt = pffmpeg->ifmt;

    ret = avformat_open_input(&ifmt_ctx, pffmpeg->audio_src, ifmt, &options);
    if (ret < 0) {
        printf("can not open input stream\n");
        return -EFAULT;
    }
    ret = avformat_find_stream_info(ifmt_ctx, NULL);
    if (ret < 0) {
        printf("can not find stream\n");
        return -EFAULT;
    }
    for (i = 0; i < ifmt_ctx->nb_streams; i++) {
        if (ifmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            pffmpeg->i_audio_stream = ifmt_ctx->streams[i];
            pffmpeg->audio_index = i;
        }
    }
    if (pffmpeg->audio_index < 0) {
        printf("can not find stream\n");
        return -EFAULT;
    }
    pffmpeg->pAudioCodecCtx = avcodec_alloc_context3(NULL);
    if (pffmpeg->pAudioCodecCtx) {
        avcodec_parameters_to_context(pffmpeg->pAudioCodecCtx, ifmt_ctx->streams[pffmpeg->audio_index]->codecpar);
    }
    pffmpeg->pAudioCodec = avcodec_find_decoder(pffmpeg->pAudioCodecCtx->codec_id);
    if (pffmpeg->pAudioCodec == NULL) {
        printf("audio Codec not found\n");
        return -EFAULT;
    }
    if (avcodec_open2(pffmpeg->pAudioCodecCtx, pffmpeg->pAudioCodec, NULL) < 0) {
        printf("Could not open codec");
        return -EFAULT;
    }

    pffmpeg->pAudioEncodeCodec = avcodec_find_encoder(AV_CODEC_ID_PCM_ALAW);
    if (!pffmpeg->pAudioEncodeCodec) {
        fprintf(stderr, "Codec not found\n");
        return -EFAULT;
    }
    pffmpeg->pAudioEncodeCodecCtx = avcodec_alloc_context3(pffmpeg->pAudioEncodeCodec);
    if (!pffmpeg->pAudioEncodeCodecCtx) {
        fprintf(stderr, "Could not allocate audio codec context\n");
        return -EFAULT;
    }
    pffmpeg->pAudioEncodeCodecCtx->time_base.num = 1;
    pffmpeg->pAudioEncodeCodecCtx->time_base.den = 8000 / 8000;
    pffmpeg->pAudioEncodeCodecCtx->bit_rate = 8000;
    pffmpeg->pAudioEncodeCodecCtx->sample_fmt = AV_SAMPLE_FMT_S16;
    pffmpeg->pAudioEncodeCodecCtx->sample_rate = 8000;
    pffmpeg->pAudioEncodeCodecCtx->channel_layout = AV_CH_LAYOUT_MONO;
    pffmpeg->pAudioEncodeCodecCtx->channels = 1;
#if 0
    /* put sample parameters */
	pffmpeg->pAudioEncodeCodecCtx->frame_size = 8000;
	pffmpeg->pAudioEncodeCodecCtx->bit_rate = 8000;
	/* check that the encoder supports s16 pcm input */
	pffmpeg->pAudioEncodeCodecCtx->sample_fmt = AV_SAMPLE_FMT_S16;
	/* select other audio parameters supported by the encoder */
	pffmpeg->pAudioEncodeCodecCtx->sample_rate = 8000;
	pffmpeg->pAudioEncodeCodecCtx->channel_layout = AV_CH_LAYOUT_MONO;
	pffmpeg->pAudioEncodeCodecCtx->channels = av_get_channel_layout_nb_channels(AV_CH_LAYOUT_MONO);
#endif
    if (avcodec_open2(pffmpeg->pAudioEncodeCodecCtx, pffmpeg->pAudioEncodeCodec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return -EFAULT;
    }
    pffmpeg->running_flag = 1;
    ret = pthread_create(&(pffmpeg->pid), NULL, local_ffmpeg_encode_audio_thread, pffmpeg);
    if (ret < 0) {
        printf("create audio thread error %d \n", ret);
        perror("error:");
        return -EINVAL;
    }
    return 0;
}



static int32_t local_ffmpeg_audio_convert_init(struct local_ffmpeg* pffmpeg, int src_ch_layout, int src_rate, AVSampleFormat src_sample_fmt, int dst_ch_layout, int dst_rate, AVSampleFormat dst_sample_fmt)
{
    int ret;
    struct SwrContext *audio_convert_ctx;
    if (!pffmpeg->audio_convert_ctx) {
        pffmpeg->audio_convert_ctx = swr_alloc();
        if (!pffmpeg->audio_convert_ctx) {
            printf("alloc audio convert error\n");
            return -EFAULT;
        }
        audio_convert_ctx = pffmpeg->audio_convert_ctx;
        av_opt_set_int(audio_convert_ctx, "in_channel_layout", AV_CH_LAYOUT_STEREO, 0);
        av_opt_set_int(audio_convert_ctx, "in_sample_rate", src_rate, 0);
        av_opt_set_sample_fmt(audio_convert_ctx, "in_sample_fmt", src_sample_fmt, 0);

        av_opt_set_int(audio_convert_ctx, "out_channel_layout", AV_CH_LAYOUT_MONO, 0);
        av_opt_set_int(audio_convert_ctx, "out_sample_rate", dst_rate, 0);
        av_opt_set_sample_fmt(audio_convert_ctx, "out_sample_fmt", dst_sample_fmt, 0);
        ret = swr_init(audio_convert_ctx);
        if (ret < 0) {
            printf("Failed to initialize the resampling context\n");
            return -EFAULT;
        }
        pffmpeg->audio_dst_sample_fmt = dst_sample_fmt;
        pffmpeg->audio_src_sample_fmt = src_sample_fmt;

        pffmpeg->audio_src_nb_samples = src_rate;
        pffmpeg->audio_src_nb_channels = av_get_channel_layout_nb_channels(AV_CH_LAYOUT_STEREO);
        ret = av_samples_alloc_array_and_samples(&pffmpeg->audio_src_data, &pffmpeg->audio_src_linesize, pffmpeg->audio_src_nb_channels, pffmpeg->audio_src_nb_samples, src_sample_fmt, 0);
        if (ret < 0) {
            printf("Could not allocate source samples\n");
            return -EFAULT;
        }
        pffmpeg->audio_dst_nb_samples = dst_rate;
        pffmpeg->audio_dst_nb_channels = av_get_channel_layout_nb_channels(AV_CH_LAYOUT_MONO);
        ret = av_samples_alloc_array_and_samples(&pffmpeg->audio_dst_data, &pffmpeg->audio_dst_linesize, pffmpeg->audio_dst_nb_channels, pffmpeg->audio_dst_nb_samples, dst_sample_fmt, 0);
        if (ret < 0) {
            printf("Could not allocate destination samples\n");
            return -EFAULT;
        }
    }
    return 0;
}

static int32_t local_ffmpeg_audio_convert_frame(struct local_ffmpeg* pffmpeg, uint8_t* src_data, int src_linesize)
{
    int ret;
    if (src_linesize > pffmpeg->audio_src_linesize) {
        printf("error audio frame %d\n", src_linesize);
        return -EFAULT;
    }
    memcpy(pffmpeg->audio_src_data[0], src_data, src_linesize);
    ret = swr_convert(pffmpeg->audio_convert_ctx, pffmpeg->audio_dst_data, pffmpeg->audio_dst_nb_samples, (const uint8_t **)pffmpeg->audio_src_data, src_linesize/4);
    if (ret < 0) {
        printf("Error while converting\n");
        return ret;
    }
    pffmpeg->audio_dst_bufsize = av_samples_get_buffer_size(&pffmpeg->audio_dst_linesize, pffmpeg->audio_dst_nb_channels, ret, pffmpeg->audio_dst_sample_fmt, 1);
    if (pffmpeg->audio_dst_bufsize < 0) {
        printf("Could not get sample buffer size\n");
    }
    return pffmpeg->audio_dst_bufsize;
}

static int32_t local_ffmpeg_audio_packet_encode(struct local_ffmpeg* pffmpeg, uint8_t *data, int size)
{
    int i;
    int64_t avtime;
    AVPacket avpkt, *pkt = &avpkt;
    struct ff_video_frame *frame;
    AVFrame *avframe;
    AVCodecContext *c = NULL;
    int ret, got_output, buffer_size;
    uint8_t *output_data;

#if 0
    uint8_t u8temp;
	for (i = 0; i < size; i += 2) {
		u8temp = data[i];
		data[i] = data[i + 1];
		data[i + 1] = u8temp;
	}
#endif
    c = pffmpeg->pAudioEncodeCodecCtx;
    if (!c) {

        return -ENOSYS;
    }
    avframe = av_frame_alloc();
    av_init_packet(pkt);
    if (!avframe) {
        printf("cant not alloc avframe\n");
        return -ENOMEM;
    }
    //avframe->nb_samples = c->frame_size;
    avframe->nb_samples     = size/2;
    avframe->format = c->sample_fmt;
    avframe->channel_layout = c->channel_layout;
    buffer_size = av_samples_get_buffer_size(NULL, c->channels, avframe->nb_samples, c->sample_fmt, 0);
    output_data = (uint8_t*)av_malloc(buffer_size);
    if (!output_data) {
        printf("cloud not allocate audio bufer\n");
        av_log(NULL, AV_LOG_ERROR, "Could not allocate audio buffer\n");
        av_frame_free(&avframe);
        av_packet_unref(pkt);
        return -EFAULT;
    }
    ret = avcodec_fill_audio_frame(avframe, c->channels, c->sample_fmt, (const uint8_t*)output_data, buffer_size, 0);
    if (ret < 0) {
        av_freep(&output_data);
        av_frame_free(&avframe);
        av_packet_unref(pkt);
        printf("convert audio frame failed\n");
        return -EFAULT;
    }

    memcpy(output_data, data, size);
    if ((ret = avcodec_send_frame(c, avframe)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "send audio sample to encode failed %d\n", ret);
    }
    //DBG(DBG_WARN,"send frame end\n");
    if ((ret = avcodec_receive_packet(c, pkt)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "receive audio sample to encode failed %d\n", ret);
    }
#if 0
    //DBG(DBG_WARN,"recevice frame end\n");
	if (ret >= 0) {
		/*notify audio packet*/
		avtime = av_gettime_monotonic();
		pkt->pts = (avtime - paudio->g_time);
		pkt->dts = pkt->pts;

		//av_log(NULL, AV_LOG_DEBUG,"g_time  = %lld ,avtime = %lld,  pkt dts = %lld,pkt->pts = %lld pkt->size = %d\n",paudio->g_time,avtime,pkt->dts,pkt->pts,pkt->size);
		//av_log(NULL, AV_LOG_ERROR,"g_time  = %lld ,avtime = %lld,  pkt dts = %lld,pkt->pts = %lld pkt->size = %d\n",paudio->g_time,avtime,pkt->dts,pkt->pts,pkt->size);

		pkt->stream_index = 1; //audio

		file_save("record.aac", pkt->data, pkt->size);
	}
#endif
    pkt->flags = 0;
    pkt->stream_index = 1;
    pffmpeg->op->trigger_notify(pffmpeg, LOCAL_FFMPEG_EVENT_AUDIO_FRAME_READY, pkt);
    av_packet_unref(pkt);
    //DBG(DBG_WARN,"pcm to aac end\n");
    av_freep(&output_data);
    av_frame_free(&avframe);
    return 0;
}

static int32_t local_ffmpeg_video_convert_init(struct local_ffmpeg* pffmpeg,int src_width,int src_height, AVPixelFormat src_pixel_fmt,int dst_width,int dst_height, AVPixelFormat dst_pixel_fmt)
{
    if((pffmpeg->video_src_width == src_width)
    && (pffmpeg->video_src_height == src_height)
    && (pffmpeg->video_src_pixel_fmt == src_pixel_fmt)
    && (pffmpeg->video_dst_width == dst_width)
    && (pffmpeg->video_dst_width == dst_height)
    && (pffmpeg->video_dst_pixel_fmt == dst_pixel_fmt)){
        return 0;
    }

    if(pffmpeg->pvideoSwsContext){
        sws_freeContext(pffmpeg->pvideoSwsContext);
        pffmpeg->pvideoSwsContext = NULL;
    }

    pffmpeg->video_src_width = src_width;
    pffmpeg->video_src_height = src_height;
    pffmpeg->video_src_pixel_fmt = src_pixel_fmt;

    pffmpeg->video_dst_width = dst_width;
    pffmpeg->video_dst_height = dst_height;
    pffmpeg->video_dst_pixel_fmt = dst_pixel_fmt;

#if 1
    if(!pffmpeg->video_src_frame) {
        pffmpeg->video_src_frame = av_frame_alloc();
    }
    if(!pffmpeg->video_dst_frame) {
        pffmpeg->video_dst_frame = av_frame_alloc();
    }
    pffmpeg->pvideoSwsContext = sws_getContext(src_width, src_height, src_pixel_fmt, dst_width, dst_height, dst_pixel_fmt, SWS_BICUBIC, NULL, NULL, NULL);
#endif
    return 0;
}

static int32_t local_ffmpeg_video_convert_frame(struct local_ffmpeg* pffmpeg,uint8_t* src,uint8_t* dst,int dvalue)
{
    int ret;
    AVFrame* srcframe;
    AVFrame* dstframe;
    if(!pffmpeg->pvideoSwsContext){
        return -EFAULT;
    }

    srcframe = pffmpeg->video_src_frame;
    dstframe = pffmpeg->video_dst_frame;
    av_image_fill_arrays(srcframe->data, srcframe->linesize, src, pffmpeg->video_src_pixel_fmt, pffmpeg->video_src_width, pffmpeg->video_src_height, 1);
    av_image_fill_arrays(dstframe->data, dstframe->linesize, dst, pffmpeg->video_dst_pixel_fmt, pffmpeg->video_dst_width, pffmpeg->video_dst_height, 1);

    memset(dst, dvalue ,pffmpeg->video_dst_width * pffmpeg->video_dst_height * 4);
    return 0;

    ret = sws_scale(pffmpeg->pvideoSwsContext, (const uint8_t* const*)srcframe->data, srcframe->linesize, 0, srcframe->height, dstframe->data, dstframe->linesize);
    if(ret != dstframe->height){
        return -EFAULT;
    }

    return 0;
}

static int32_t local_ffmpeg_register_notify(struct local_ffmpeg* plocal_ffmpeg, int32_t event, local_ffmpeg_event_notify notify, void* object)
{
    struct local_ffmpeg_event_action* paction;
    if(!notify || (event <= LOCAL_FFMPEG_EVENT_NONE) || (event >= LOCAL_FFMPEG_EVENT_MAX)){
        return -EINVAL;
    }
    paction = (struct local_ffmpeg_event_action*)malloc(sizeof(struct local_ffmpeg_event_action));
    if(!paction){
        printf("malloc error\n");
        return -ENOMEM;
    }
    paction->notify = notify;
    paction->object = object;
    lock(&(plocal_ffmpeg->lock));
    paction->next = plocal_ffmpeg->paction[event];
    plocal_ffmpeg->paction[event] = paction;
    unlock(&(plocal_ffmpeg->lock));
    return 0;
}

static int32_t local_ffmpeg_unregister_notify(struct local_ffmpeg* plocal_ffmpeg, int32_t event, void* object)
{
    struct local_ffmpeg_event_action *paction,* ptmp;
    if((event <= LOCAL_FFMPEG_EVENT_NONE) || (event >= LOCAL_FFMPEG_EVENT_MAX)){
        return -EINVAL;
    }
    lock(&(plocal_ffmpeg->lock));
    paction = plocal_ffmpeg->paction[event];
    if(paction->object == object){
        plocal_ffmpeg->paction[event] = paction->next;
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
    unlock(&(plocal_ffmpeg->lock));
    return 0;
}

static int32_t local_ffmpeg_trigger_notify(struct local_ffmpeg* plocal_ffmpeg, int32_t event, void* context)
{
    struct local_ffmpeg_event_action* paction;
    if((event <= LOCAL_FFMPEG_EVENT_NONE) || (event >= LOCAL_FFMPEG_EVENT_MAX)){
        return -EINVAL;
    }
    paction = plocal_ffmpeg->paction[event];
    while(paction){
        paction->notify(plocal_ffmpeg, event, paction->object, context);
        paction = paction->next;
    }
    return 0;
}

static struct local_ffmpeg_operation local_ffmpeg_op =
{
    .init = local_ffmpeg_init,
    .release = local_ffmpeg_release,

    .query_local_camera = local_ffmpeg_query_local_camera,
    .query_local_micro = local_ffmpeg_query_local_micro,

    .config_video_src = local_ffmpeg_config_video_src,
    .encode_video = local_ffmpeg_encode_video,
    .encode_audio = local_ffmpeg_encode_audio,

    .audio_convert_init = local_ffmpeg_audio_convert_init,
    .audio_convert_frame = local_ffmpeg_audio_convert_frame,

    .audio_packet_encode = local_ffmpeg_audio_packet_encode,

    .video_convert_init = local_ffmpeg_video_convert_init,
    .video_convert_frame = local_ffmpeg_video_convert_frame,

    .register_notify = local_ffmpeg_register_notify,
    .unregister_notify = local_ffmpeg_unregister_notify,
    .trigger_notify = local_ffmpeg_trigger_notify,
};

int32_t create_init_local_ffmpeg(struct local_ffmpeg** plocal_ffmpeg)
{
    int32_t ret;
    struct local_ffmpeg* ptmp;
    (*plocal_ffmpeg) = (struct local_ffmpeg*)malloc(sizeof(struct local_ffmpeg));
    if(!(*plocal_ffmpeg)){
        printf("malloc error\n");
        return -ENOMEM;
    }
    ptmp = *plocal_ffmpeg;
    memset(ptmp,0,sizeof(struct local_ffmpeg));
    ptmp->op = &local_ffmpeg_op;
    ret = ptmp->op->init(ptmp);
    if(ret < 0){
        printf("init error\n");
        release_destroy_local_ffmpeg(ptmp);
        return ret;
    }
    return 0;
}

void release_destroy_local_ffmpeg(struct local_ffmpeg* plocal_ffmpeg)
{
    if(plocal_ffmpeg){
        plocal_ffmpeg->op->release(plocal_ffmpeg);
        free(plocal_ffmpeg);
    }
}
