#include <jni.h>
#include <string>
#include <errno.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include "utilbase.h"
#include "libuvc/libuvc.h"
#include "local_ffmpeg.h"
#include "aicore.h"

struct local_ffmpeg* pffmpeg = NULL;

int frame_width;
int frame_height;
int surface_width;
int surface_height;
float zoom;
int display_height;
int display_width;

const char* mUsbFs = "/dev/bus/usb";
int mFd;
uvc_context_t *mContext;
uvc_device_t *mDevice;
uvc_device_handle_t *mDeviceHandle;

ANativeWindow* g_nativeWindow = NULL;
uvc_frame_t *g_showFrame = NULL;
uint8_t* showFrameData = NULL;

float ai_threshold = 0.5;
#define __MAX_OBJECT_DETECT__  8
object_t ai_object[__MAX_OBJECT_DETECT__];

struct timeval g_tv;
int g_real_fps;
int g_fps;
int g_ai_detect = 0;

static void native_uvc_frame_ready_callback(uvc_frame_t *src_frame, void *object)
{
    int i,j,k;
    int split;
    int ret;
    uint32_t* rgbx;
    struct timeval t_tv;
    g_fps++;
    gettimeofday(&t_tv,NULL);
    if((t_tv.tv_sec * 1000 - g_tv.tv_sec*1000) + (t_tv.tv_usec/1000 - t_tv.tv_usec/1000) >= 1000){
        g_real_fps = g_fps;
        g_fps = 0;
        gettimeofday(&g_tv,NULL);
    }

    ANativeWindow_Buffer wbuffer;
#if 1
    uvc_any2rgbx(src_frame, g_showFrame);
    ret = ai_core_send_image((const char*)g_showFrame->data,frame_width * frame_height * 4);
    if(ret < 0){
        LOGE("ai_core_send_image error\n");
    }
    ret = ai_core_fetch_object(ai_object, __MAX_OBJECT_DETECT__, ai_threshold);
    if(ret >= 0){
        g_ai_detect = 1;
        rgbx = (uint32_t*)g_showFrame->data;
        for(j = 0; j < ret; j++){
            for(k = ai_object[j].x; k < (ai_object[j].x + ai_object[j].w); k++){
                rgbx[(ai_object[j].y) * frame_width + k] = 0xFFFF0000;
                rgbx[(ai_object[j].y + 1) * frame_width + k] = 0xFFFF0000;
                rgbx[(ai_object[j].y - 1 + ai_object[j].h) * frame_width + k] = 0xFFFF0000;
                rgbx[(ai_object[j].y + ai_object[j].h) * frame_width + k] = 0xFFFF0000;
            }

            for(k = ai_object[j].y; k < (ai_object[j].y + ai_object[j].h); k++){
                rgbx[k * frame_width + ai_object[j].x] = 0xFFFF0000;
                rgbx[k * frame_width + ai_object[j].x + 1] = 0xFFFF0000;
                rgbx[k * frame_width + ai_object[j].x + ai_object[j].w - 1] = 0xFFFF0000;
                rgbx[k * frame_width + ai_object[j].x + ai_object[j].w] = 0xFFFF0000;
            }

        }
        //LOGE("============ai_core_fetch_object success,cound %d============\n",ret);
    }else{
        g_ai_detect = 0;
    }
    ANativeWindow_lock(g_nativeWindow, &wbuffer, 0);
    if(surface_width < surface_height){
        split = (display_height - frame_height)/2;
        showFrameData = ((uint8_t*)wbuffer.bits) + (split*frame_width*4);
        memcpy(showFrameData, g_showFrame->data, frame_width * frame_height * 4);
    }else{
    }
    ANativeWindow_unlockAndPost(g_nativeWindow);
#else
    ANativeWindow_lock(g_nativeWindow, &wbuffer, 0);
    showFrameData = g_showFrame->data;
    g_showFrame->data = wbuffer.bits;
    uvc_any2rgbx(src_frame, g_showFrame);
    g_showFrame->data = showFrameData;
    ANativeWindow_unlockAndPost(g_nativeWindow);
#endif
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lin_otguvc_MainActivity_createCaptureFromJNI(
        JNIEnv* env,
        jobject /* this */,
        jint vid,
        jint pid,
        jint fd,
        jint busnum,
        jint devnum,
        jint width,
        jint height,
        jint minfps,
        jint maxfps,
        jint surfacewidth,
        jint surfaceheight) {

    int ret;
    uvc_stream_ctrl_t ctrl;
    uvc_frame_desc_t *frame_desc;

#if 1
    ret = ai_core_init(width,height);
    if(ret < 0){
        LOGE("create aicore error\n");
        return ret;
    }
#endif
#if 0
    ret = create_init_local_ffmpeg(&pffmpeg);
    if(ret < 0){
        LOGE("create ffmpeg error\n");
    }
#endif
    ret = uvc_init2(&mContext, NULL, mUsbFs);
    if(ret < 0){
        LOGE("uvc_init2 error\n");
        return ret;
    }
    ret = uvc_get_device_with_fd(mContext, &mDevice, vid, pid, NULL, fd, busnum, devnum);
    if(ret < 0){
        LOGE("uvc_get_device_with_fd error\n");
        return ret;
    }
    ret = uvc_open(mDevice, &mDeviceHandle);
    if(ret < 0){
        LOGE("uvc_open error\n");
        return ret;
    }
    mFd = fd;
    ret = uvc_get_stream_ctrl_format_size_fps(mDeviceHandle, &ctrl, UVC_FRAME_FORMAT_YUYV,  width, height, minfps, maxfps);
    if(ret < 0){
        LOGE("uvc_get_stream_ctrl_format_size_fps error\n");
        return ret;
    }
    ret = uvc_get_frame_desc(mDeviceHandle, &ctrl, &frame_desc);
    if(ret < 0){
        LOGE("uvc_get_frame_desc error\n");
        return ret;
    }

    frame_width = width;
    frame_height = height;
    surface_width = surfacewidth;
    surface_height = surfaceheight;

    if(surface_width < surface_height) {
        zoom = frame_width / (surface_height * 1.0);
        display_height = surfaceheight * zoom;
        ANativeWindow_setBuffersGeometry(g_nativeWindow, frame_width, display_height,WINDOW_FORMAT_RGBA_8888);
    }else{
        zoom = frame_height / (surface_height * 1.0);
        display_width = surface_width * zoom;
        ANativeWindow_setBuffersGeometry(g_nativeWindow, display_width, frame_height,WINDOW_FORMAT_RGBA_8888);
    }
    g_showFrame = uvc_allocate_frame(frame_width* frame_height * 4);
    if(!g_showFrame){
        LOGE("malloc g_showFrame error\n");
        return -ENOMEM;
    }

    ret = uvc_start_streaming_bandwidth( mDeviceHandle, &ctrl, native_uvc_frame_ready_callback, NULL, 0, 0);
    if(ret < 0){
        LOGE("uvc_start_streaming_bandwidth error\n");
        return ret;
    }
    gettimeofday(&g_tv,NULL);
    return 0;
}


extern "C" JNIEXPORT jint JNICALL
Java_com_lin_otguvc_MainActivity_destroyCaptureFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    if(mDeviceHandle) {
        uvc_close(mDeviceHandle);
        mDeviceHandle = NULL;
    }
    if(mDevice){
        uvc_unref_device(mDevice);
        mDevice = NULL;
    }
    if(mFd > 0){
        close(mFd);
        mFd = 0;
    }
    if(g_showFrame){
        uvc_free_frame(g_showFrame);
        g_showFrame = NULL;
    }
#if 0
    if(pffmpeg){
        release_destroy_local_ffmpeg(pffmpeg);
        pffmpeg = NULL;
    }
#endif
#if 1
    ai_core_free();
#endif
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lin_otguvc_MainActivity_setSurfaceviewFromJNI(
        JNIEnv* env,
        jobject /* this */,
        jobject surface,
        jint surfacewidth,
        jint surfaceheight) {

    surface_width = surfacewidth;
    surface_height = surfaceheight;
    if(surface_width < surface_height) {
        zoom = frame_width / (surface_width * 1.0);
        display_height = surfacewidth * zoom;
        g_nativeWindow = ANativeWindow_fromSurface(env, surface);
        ANativeWindow_setBuffersGeometry(g_nativeWindow, frame_width, display_height, WINDOW_FORMAT_RGBA_8888);
    }else{
        zoom = frame_height / (surface_height * 1.0);
        display_width = surface_width * zoom;
        ANativeWindow_setBuffersGeometry(g_nativeWindow, display_width, frame_height,WINDOW_FORMAT_RGBA_8888);
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lin_otguvc_MainActivity_setAIThresholdFromJNI(
        JNIEnv* env,
        jobject /* this */,
        jfloat threshold)
{
    ai_threshold = threshold;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lin_otguvc_MainActivity_getFpsFromJNI(
        JNIEnv* env,
        jobject /* this */)
{
    return g_real_fps;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lin_otguvc_MainActivity_getDetectFromJNI(
        JNIEnv* env,
        jobject /* this */)
{
    return g_ai_detect;
}