#ifndef _ZUTILS_H_
#define _ZUTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#ifdef OPENCL
#	include "CL/opencl.h"
#endif
#ifdef __ANDROID__
#	include <android/log.h>
#endif

#ifdef __linux__
#	define BINARY_FILENAME_TO_START(name, suffix) \
		_binary_##name##_##suffix##_start
#	define BINARY_FILENAME_TO_END(name, suffix) \
		_binary_##name##_##suffix##_end
#	define BINARY_FILENAME_TO_SIZE(name, suffix) \
		_binary_##name##_##suffix##_size
#elif defined(_WIN32)
#	define BINARY_FILENAME_TO_START(name, suffix) \
		binary_##name##_##suffix##_start
#	define BINARY_FILENAME_TO_END(name, suffix) \
		binary_##name##_##suffix##_end	
#	define BINARY_FILENAME_TO_SIZE(name, suffix) \
		binary_##name##_##suffix##_size
#else
#	error "unsupported operation system!"
#endif

#ifdef OPENCL
#ifdef USE_FLOAT
#	define PARSE_PRECISION strcat(options, " -DFLOAT -DDATA_TYPE=float -DREAD_WRITE_DATA_TYPE=f")
#	define IMAGE_CHANNEL_DATA_TYPE CL_FLOAT
#	define MEM_MAP_PTR_TYPE cl_float
#else
#	define PARSE_PRECISION strcat(options, " -DDATA_TYPE=half -DREAD_WRITE_DATA_TYPE=h")
#	define IMAGE_CHANNEL_DATA_TYPE CL_HALF_FLOAT
#	define MEM_MAP_PTR_TYPE cl_half
#endif
#endif

#ifdef __ANDROID__
#	define LOG_TAG "aicore"
#	ifdef NDEBUG
#		define LOGV(FMT, ...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, "[%s:%d:%s]:" FMT,	\
			__FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#		define LOGD(FMT, ...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s:%d:%s]:" FMT,	\
			__FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#		define LOGI(FMT, ...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "[%s:%d:%s]:" FMT,	\
			__FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	else
#		define LOGV(FMT, ...)
#		define LOGD(FMT, ...)
#		define LOGI(FMT, ...)	
#	endif
#	define LOGW(FMT, ...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "[%s:%d:%s]:" FMT,	\
		__FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGE(FMT, ...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "[%s:%d:%s]:" FMT,	\
		__FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGF(FMT, ...) __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, "[%s:%d:%s]:" FMT,	\
		__FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#else
#	define LOGV(FMT, ...) fprintf(stderr, "[%s:%d:%s]:" FMT, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGD(FMT, ...) fprintf(stderr, "[%s:%d:%s]:" FMT, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGI(FMT, ...) fprintf(stderr, "[%s:%d:%s]:" FMT, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGW(FMT, ...) fprintf(stderr, "[%s:%d:%s]:" FMT, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGE(FMT, ...) fprintf(stderr, "[%s:%d:%s]:" FMT, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#	define LOGF(FMT, ...) fprintf(stderr, "[%s:%d:%s]:" FMT, __FILE__, __LINE__, __FUNCTION__, ## __VA_ARGS__)
#endif

#ifdef AICORE_BUILD_DLL
#ifdef _WIN32
#	define AICORE_LOCAL
#else
#	define AICORE_LOCAL __attribute__ ((visibility("hidden")))
#endif
#else
#ifdef _WIN32
#	define AICORE_LOCAL
#else
#	define AICORE_LOCAL __attribute__ ((visibility("hidden")))
#endif
#endif

AICORE_LOCAL void mmfree(int n, ...);
AICORE_LOCAL void mset(char *const X, size_t size, const char *const val, int nvals);
AICORE_LOCAL void mcopy(const char *const X, char *const Y, size_t size);
AICORE_LOCAL void save_volume(float *data, int width, int height, int nchannels, const char *path);
#ifdef OPENCL
AICORE_LOCAL int nchw_to_nhwc(const float *const input, MEM_MAP_PTR_TYPE *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch, int channel_block_size);
AICORE_LOCAL int nhwc_to_nchw(const MEM_MAP_PTR_TYPE *const input, float *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch, int channel_block_size);
#endif
AICORE_LOCAL int round_up_division_2(int x);
AICORE_LOCAL int round_up_division_4(int x);
AICORE_LOCAL unsigned int roundup_power_of_2(unsigned int a);
AICORE_LOCAL unsigned int round_up_multiple_of_8(unsigned int x);

#ifdef __cplusplus
}
#endif

#endif