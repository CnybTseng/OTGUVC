/** @file aicore.c - Implementation
 ** @brief 智慧农业核心模块
 ** @author 曾志伟
 ** @date 2018.11.16
 **/

/*
Copyright (C) 2018 Chengdu ZLT Technology Co., Ltd.
All rights reserved.

This file is part of the smart agriculture toolkit and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "aicore.h"
#include "znet.h"
#include "list.h"
#include "fifo.h"
#include "image.h"
#include "zutils.h"
#include "cl_wrapper.h"
#include "half.h"

typedef int (*lock_method)(pthread_mutex_t *mutex);
typedef int (*unlock_method)(pthread_mutex_t *mutex);

typedef enum {
	THREAD_RUNNING,
	THREAD_DEAD
} thread_status_t;

typedef struct {
	void *data;
	int busy;
	pthread_mutex_t mutex;
	lock_method lock;
	unlock_method unlock;
} memory_unit_t;

typedef struct {
	int image_width;
	int image_height;
	int roix;
	int roiy;
	int roiw;
	int roih;
	int standard_width;
	int standard_height;
	int cache_num;
	void *layers[24];
	znet *net;
	int init_status;
	pthread_t dnn_inference_tid;
	thread_status_t thread_status;
	Fifo *image_queue;
	char *image_queue_read_buffer;
	char *image_queue_write_buffer;
	Fifo *object_queue;
	char *object_queue_read_buffer;
	char *object_queue_write_buffer;
	image_standardizer *standardizer;
	memory_unit_t mempry_pool[16];
	int memory_pool_size;
	float threshold;
	unsigned int sample_interval;
	unsigned long long frame_counter;
} ai_core_param_t;

static ai_core_param_t core_param;
static pthread_once_t core_create_once_control = PTHREAD_ONCE_INIT;

static void ai_core_init_routine();
static void **create_dnn();
static int create_dnn_inference_thread();
static void *dnn_inference_thread(void *param);
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
static void *create_standard_image_object();
#ifdef ION
static cl_mem allocate_image_from_ion(int ion_filedesc, void *ion_hostptr, int width, int height);
#endif
#endif
static int init_memory_pool();
static void *allocate_from_memory_pool();
static void back_to_memory_pool(memory_unit_t *mem);
static void free_memory_pool();
static void wait_for_thread_dead(pthread_t tid);
static void clear_object_list();
static void __attribute__((unused)) save_standard_image(void *image, int width, int height, const char *filename);

#ifdef OPENCL
cl_wrapper wrapper;
#endif

int ai_core_init(unsigned int width, unsigned int height)
{
#ifdef OPENCL	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		LOGE("cl_create_wrapper fail, error code:%d.\n", errcode);
		core_param.init_status = AIC_OPENCL_INIT_FAIL;
		return core_param.init_status;
	}
#endif	
	core_param.image_width = width;
	core_param.image_height = height;
	core_param.roiw = 1000;
	core_param.roih = 1000;
	core_param.roix = (core_param.image_width - core_param.roiw) >> 1;
	core_param.roiy = (core_param.image_height - core_param.roih) >> 1;
	pthread_once(&core_create_once_control, &ai_core_init_routine);
	system("rm -f *.cl.bin");
	
	return core_param.init_status;
}

int ai_core_send_image(const char *const rgb24, size_t size)
{
	if (core_param.frame_counter++ % core_param.sample_interval != 0) return AIC_FRAME_DISCARD;
	void *standard_image = allocate_from_memory_pool();
	if (!standard_image) return AIC_ALLOCATE_FAIL;

 	standardize_image(core_param.standardizer, (unsigned char *)rgb24, core_param.image_width,
 		core_param.image_height, core_param.roix, core_param.roiy, core_param.roiw, core_param.roih, standard_image);
 	memcpy(core_param.image_queue_read_buffer, &standard_image, sizeof(void *));

	int timer = 11;
	const struct timespec req = {0, 100000};
	const unsigned int request_size = roundup_power_of_2(sizeof(void *));
	while (--timer) {
		unsigned int write_size = fifo_put(core_param.image_queue, core_param.image_queue_read_buffer, request_size);
		if (write_size == request_size) break;
		nanosleep(&req, NULL);
	}
	
	if (0 == timer) {
		back_to_memory_pool(standard_image);
		return AIC_ENQUEUE_FAIL;
	}
	
	return AIC_OK;
}

int ai_core_send_ion_image(int ion_filedesc, void *const ion_hostptr, int width, int height)
{
#ifdef ION
	if (core_param.frame_counter++ % core_param.sample_interval != 0) return AIC_FRAME_DISCARD;
	cl_mem input = allocate_image_from_ion(ion_filedesc, ion_hostptr, width, height);
	if (!input) return AIC_ALLOCATE_FAIL;

	void *standard_image = allocate_from_memory_pool();
	if (!standard_image) return AIC_ALLOCATE_FAIL;

 	standardize_ion_image(core_param.standardizer, input, width, height, core_param.roix, core_param.roiy,
		core_param.roiw, core_param.roih, standard_image);
	clReleaseMemObject(input);
 	memcpy(core_param.image_queue_read_buffer, &standard_image, sizeof(void *));

	int timer = 11;
	const struct timespec req = {0, 100000};
	const unsigned int request_size = roundup_power_of_2(sizeof(void *));
	while (--timer) {
		unsigned int write_size = fifo_put(core_param.image_queue, core_param.image_queue_read_buffer, request_size);
		if (write_size == request_size) break;
		nanosleep(&req, NULL);
	}
	
	if (0 == timer) {
		back_to_memory_pool(standard_image);
		return AIC_ENQUEUE_FAIL;
	}	
	
	return AIC_OK;
#endif
	return 0;
}

int ai_core_fetch_object(object_t *const object, size_t number, float threshold)
{
	int timer = 11;
	const struct timespec req = {0, 100000};
	const int request_size = roundup_power_of_2(sizeof(list *));	
	while (--timer) {
		int read_size = fifo_get(core_param.object_queue, core_param.object_queue_write_buffer, request_size);
		if (read_size == request_size) break;
		nanosleep(&req, NULL);
	}
	
	if (0 == timer) return AIC_DEQUEUE_FAIL;
	core_param.threshold = threshold;
	
	int counter = 0;
	list *object_list = NULL;
	memcpy(&object_list, core_param.object_queue_write_buffer, sizeof(list *));
	node *n = object_list->head;
	while (n) {
		detection *det = (detection *)n->val;
		if (det->objectness < threshold) {
			n = n->next;
			continue;
		}
		
		int most_likely_class_id = 0;
		float most_likely_class_prob = 0;
		for (int i = 0; i < det->classes; ++i) {
			if (det->probabilities[i] > most_likely_class_prob) {
				most_likely_class_prob = det->probabilities[i];
				most_likely_class_id = i;
			}
		}
		
		int left = (int)((det->bbox.x - det->bbox.w / 2) * core_param.roiw + core_param.roix);		
		int right = (int)((det->bbox.x + det->bbox.w / 2) * core_param.roiw + core_param.roix);
		int top = (int)((det->bbox.y - det->bbox.h / 2) * core_param.roih + core_param.roiy);
		int bottom = (int)((det->bbox.y + det->bbox.h / 2) * core_param.roih + core_param.roiy);
		
		if (left < 0) left = 0;
		if (left > core_param.image_width - 1) left = core_param.image_width - 1;
		if (right < 0) right = 0;
		if (right > core_param.image_width - 1) right = core_param.image_width - 1;
		if (top < 0) top = 0;
		if (top > core_param.image_height - 1) top = core_param.image_height - 1;
		if (bottom < 0) bottom = 0;
		if (bottom > core_param.image_height - 1) bottom = core_param.image_height - 1;
		
		object[counter].x = left;
		object[counter].y = top;
		object[counter].w = right - left + 1;
		object[counter].h = bottom - top + 1;
		object[counter].classt = (class_t)(CORDYCEPS + most_likely_class_id);
		object[counter].objectness = det->objectness;
		object[counter].probability = most_likely_class_prob;
		
		++counter;
		if (counter > number) break;
		n = n->next;
	}

	free_detections(object_list);
	return counter;
}

void ai_core_free()
{
	core_param.thread_status = THREAD_DEAD;
	wait_for_thread_dead(core_param.dnn_inference_tid);
	if (core_param.net) {
		znet_destroy(core_param.net);
		core_param.net = NULL;
	}
	
	if (core_param.image_queue) {
		fifo_delete(core_param.image_queue);
		core_param.image_queue = NULL;
	}
	
	if (core_param.image_queue_read_buffer) {
		free(core_param.image_queue_read_buffer);
		core_param.image_queue_read_buffer = NULL;
	}
	
	if (core_param.image_queue_write_buffer) {
		free(core_param.image_queue_write_buffer);
		core_param.image_queue_write_buffer = NULL;
	}
	
	if (core_param.object_queue) {
		clear_object_list();
		fifo_delete(core_param.object_queue);
		core_param.object_queue = NULL;
	}
	
	if (core_param.object_queue_read_buffer) {
		free(core_param.object_queue_read_buffer);
		core_param.object_queue_read_buffer = NULL;
	}
	
	if (core_param.object_queue_write_buffer) {
		free(core_param.object_queue_write_buffer);
		core_param.object_queue_write_buffer = NULL;
	}
	
	if (core_param.standardizer) {
		free_image_standardizer(core_param.standardizer);
		core_param.standardizer = NULL;
	}
	
	free_memory_pool();
	
#ifdef OPENCL
	cl_destroy_wrapper(wrapper);
#endif	
}

int create_dnn_inference_thread()
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	
	int ret = pthread_create(&core_param.dnn_inference_tid, &attr, dnn_inference_thread, NULL);
	if (0 != ret) {
		LOGE("pthread_create fail.\n");
		core_param.thread_status = THREAD_DEAD;
		return -1;
	}
	
	return 0;
}

void ai_core_init_routine()
{
	core_param.standard_width = 416;
	core_param.standard_height = 416;
	core_param.cache_num = 8;
	core_param.net = NULL;
	core_param.init_status = -INT_MAX;
	core_param.image_queue = NULL;
	core_param.image_queue_read_buffer = NULL;
	core_param.image_queue_write_buffer = NULL;
	core_param.object_queue = NULL;
	core_param.object_queue_read_buffer = NULL;
	core_param.object_queue_write_buffer = NULL;
	core_param.standardizer = NULL;
	core_param.memory_pool_size = sizeof(core_param.mempry_pool) / sizeof(memory_unit_t);
	core_param.threshold = 0.5f;
	core_param.sample_interval = 1;
	core_param.frame_counter = 0;
	
	void **layers = create_dnn();
	int nlayers = sizeof(core_param.layers) / sizeof(core_param.layers[0]);
	
	core_param.net = znet_create(layers, nlayers, "agriculture.weights");
	if (!core_param.net) {
		core_param.init_status = AIC_NETWORK_INIT_FAIL;
		return ai_core_free();
	}
	
	core_param.image_queue = fifo_alloc(roundup_power_of_2(sizeof(void *)) * core_param.cache_num);
	if (!core_param.image_queue) {
		core_param.init_status = AIC_FIFO_ALLOC_FAIL;
		return ai_core_free();
	}
	
	core_param.image_queue_read_buffer = calloc(roundup_power_of_2(sizeof(void *)), 1);
	if (!core_param.image_queue_read_buffer) {
		core_param.init_status = AIC_ALLOCATE_FAIL;
		return ai_core_free();
	}
	
	core_param.image_queue_write_buffer = calloc(roundup_power_of_2(sizeof(void *)), 1);
	if (!core_param.image_queue_write_buffer) {
		core_param.init_status = AIC_ALLOCATE_FAIL;
		return ai_core_free();
	}
	
	core_param.object_queue = fifo_alloc(roundup_power_of_2(sizeof(list *) * core_param.cache_num));
	if (!core_param.object_queue) {
		core_param.init_status = AIC_FIFO_ALLOC_FAIL;
		return ai_core_free();
	}
	
	core_param.object_queue_read_buffer = calloc(roundup_power_of_2(sizeof(list *)), 1);
	if (!core_param.object_queue_read_buffer) {
		core_param.init_status = AIC_ALLOCATE_FAIL;
		return ai_core_free();
	}
	
	core_param.object_queue_write_buffer = calloc(roundup_power_of_2(sizeof(list *)), 1);
	if (!core_param.object_queue_write_buffer) {
		core_param.init_status = AIC_ALLOCATE_FAIL;
		return ai_core_free();
	}
	
	core_param.standardizer = create_image_standardizer(core_param.image_width, core_param.image_height,
		core_param.standard_width, core_param.standard_height, 3);
	if (!core_param.standardizer) {
		core_param.init_status = AIC_IMAGE_STANDARDIZER_INIT_FAIL;
		return ai_core_free();
	}
	
	int ret = init_memory_pool();
	if (ret != AIC_OK) {
		core_param.init_status = ret;
		return ai_core_free();
	}
	
	core_param.thread_status = THREAD_RUNNING;
	if (create_dnn_inference_thread()) {
		core_param.init_status = AIC_THREAD_CREATE_FAIL;
		return ai_core_free();
	}

	core_param.init_status = AIC_OK;
}

void **create_dnn()
{
	void **layers = core_param.layers;
	dim3 output_size;
	
	int bigger_mask[] = {3, 4, 5};
	int smaller_mask[] = {0, 1, 2};
	int anchor_boxes[] = {61,117, 62,191, 199,118, 128,195, 92,293, 191,291};
	const int scales = 3;
	const int classes = 1;
	const int object_tensor_depth = (4 + 1 + classes) * scales;
	
	dim3 input_size = {core_param.standard_width, core_param.standard_height, 3};
	layers[0] = make_convolutional_layer(LEAKY, input_size, 3, 16, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[1] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[2] = make_convolutional_layer(LEAKY, input_size, 3, 32, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[3] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[4] = make_convolutional_layer(LEAKY, input_size, 3, 64, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[5] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[6] = make_convolutional_layer(LEAKY, input_size, 3, 128, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[7] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[8] = make_convolutional_layer(LEAKY, input_size, 3, 256, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[9] = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	
	input_size = output_size;
	layers[10] = make_convolutional_layer(LEAKY, input_size, 3, 512, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[11] = make_maxpool_layer(input_size, 2, 1, 1, 1, &output_size);
	
	input_size = output_size;
	layers[12] = make_convolutional_layer(LEAKY, input_size, 3, 1024, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[13] = make_convolutional_layer(LEAKY, input_size, 1, 256, 1, 0, 1, 1, &output_size);
	input_size = output_size;
	layers[14] = make_convolutional_layer(LEAKY, input_size, 3, 512, 1, 1, 1, 1, &output_size);
	input_size = output_size;
	layers[15] = make_convolutional_layer(LINEAR, input_size, 1, object_tensor_depth, 1, 0, 1, 0, &output_size);
	
	input_size = output_size;
	layers[16] = make_yolo_layer(input_size, 1, 3, 6, classes, bigger_mask, anchor_boxes);
	
	int layer_ids1[] = {13};
	void *routes1[] = {layers[13]};
	layers[17] = make_route_layer(1, 1, routes1, layer_ids1, &output_size);
	
	input_size = output_size;
	layers[18] = make_convolutional_layer(LEAKY, input_size, 1, 128, 1, 0, 1, 1, &output_size);
	
	input_size = output_size;
	layers[19] = make_resample_layer(input_size, 1, 2, &output_size);
	
	int layer_ids2[] = {19, 8};
	void *routes2[] = {layers[19], layers[8]};
	layers[20] = make_route_layer(1, 2, routes2, layer_ids2, &output_size);
	
	input_size = output_size;
	layers[21] = make_convolutional_layer(LEAKY, input_size, 3, 256, 1, 1, 1, 1, &output_size);
	
	input_size = output_size;
	layers[22] = make_convolutional_layer(LINEAR, input_size, 1, object_tensor_depth, 1, 0, 1, 0, &output_size);
	
	input_size = output_size;
	layers[23] = make_yolo_layer(input_size, 1, 3, 6, classes, smaller_mask, anchor_boxes);
	
	return layers;
}

void *dnn_inference_thread(void *param)
{
	void *standard_image = NULL;
	const unsigned int request_read = roundup_power_of_2(sizeof(void *));
	const unsigned int request_write = roundup_power_of_2(sizeof(list *));
	const struct timespec req = {0, 1000000};
	while (core_param.thread_status == THREAD_RUNNING) {
		unsigned int read_size = fifo_get(core_param.image_queue, core_param.image_queue_write_buffer, request_read);
		if (read_size != request_read) {
			nanosleep(&req, NULL);
			continue;
		}

		memcpy(&standard_image, core_param.image_queue_write_buffer, sizeof(void *));
		znet_inference(core_param.net, standard_image);
		back_to_memory_pool(standard_image);
		
		list *detections = get_detections(core_param.net, core_param.threshold, core_param.roiw, core_param.roih);	
		list *bests = soft_nms(detections, 2.5);
		
		memcpy(core_param.object_queue_read_buffer, &bests, sizeof(list *));
		unsigned int write_size = fifo_put(core_param.object_queue, core_param.object_queue_read_buffer, request_write);
		if (write_size != request_write) {
			LOGW("fifo_put fail.\n");
			free_detections(bests);
		}

		free_detections(detections);
	}
	
	return 0;
}

#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
void *create_standard_image_object()
{
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = core_param.standard_width;
	image_desc.image_height = core_param.standard_height;
	
	cl_int errcode;
	cl_mem standard_image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		LOGE("clCreateImage fail, error code:%d.\n", errcode);
		return 0;
	}

	size_t origin[] = {0, 0, 0};
	size_t region[] = {core_param.standard_width, core_param.standard_height, 1};
	float fill_color[] = {0.5f, 0.5f, 0.5f, 0};
	clEnqueueFillImage(wrapper.command_queue, standard_image, fill_color, origin, region, 0, NULL, NULL);
	if (CL_SUCCESS != errcode) {
		LOGE("clEnqueueFillImage fail, error code:%d.\n", errcode);
	}
	
	return standard_image;
}

#ifdef ION
cl_mem allocate_image_from_ion(int ion_filedesc, void *ion_hostptr, int width, int height)
{
	cl_mem_ion_host_ptr ion_mem;
	ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
	ion_mem.ext_host_ptr.host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
	ion_mem.ion_filedesc                   = ion_filedesc;
	ion_mem.ion_hostptr                    = ion_hostptr;

	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width;
	image_desc.image_height = height;
	image_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_format, image_desc);
	
	cl_int errcode;
	cl_mem image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, &ion_mem, &errcode);
	if (CL_SUCCESS != errcode) {
		LOGE("clCreateImage fail, error code:%d.\n", errcode);
		return (void *)(0);
	}
	
	return image;
}
#endif
#endif

int init_memory_pool()
{
	for (int i = 0; i < core_param.memory_pool_size; ++i) {
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
		core_param.mempry_pool[i].data = create_standard_image_object();
		if (!core_param.mempry_pool[i].data) return AIC_ALLOCATE_FAIL;
#else
		core_param.mempry_pool[i].data = calloc(core_param.standard_width * core_param.standard_height * 3, sizeof(float));
		if (!core_param.mempry_pool[i].data) return AIC_ALLOCATE_FAIL;
		image img = {core_param.standard_width, core_param.standard_height, 3, core_param.mempry_pool[i].data};
		set_image(&img, 0.5);
#endif
		core_param.mempry_pool[i].busy = 0;
		if (0 != pthread_mutex_init(&core_param.mempry_pool[i].mutex, NULL)) {
			LOGE("pthread_mutex_init fail.\n");
			return AIC_ALLOCATE_FAIL;
		}
		core_param.mempry_pool[i].lock = pthread_mutex_lock;
		core_param.mempry_pool[i].unlock = pthread_mutex_unlock;
	}
	
	return AIC_OK;
}

void *allocate_from_memory_pool()
{
	for (int i = 0; i < core_param.memory_pool_size; ++i) {
		if (0 != core_param.mempry_pool[i].busy) continue;
		core_param.mempry_pool[i].lock(&core_param.mempry_pool[i].mutex);
		core_param.mempry_pool[i].busy = 1;
		core_param.mempry_pool[i].unlock(&core_param.mempry_pool[i].mutex);
		return core_param.mempry_pool[i].data;
	}
	
	return (void *)(0);
}

void back_to_memory_pool(memory_unit_t *mem)
{
	for (int i = 0; i < core_param.memory_pool_size; ++i) {
		if (mem != core_param.mempry_pool[i].data) continue;
		core_param.mempry_pool[i].lock(&core_param.mempry_pool[i].mutex);
		core_param.mempry_pool[i].busy = 0;
		core_param.mempry_pool[i].unlock(&core_param.mempry_pool[i].mutex);
		break;
	}
}

void free_memory_pool()
{
	for (int i = 0; i < core_param.memory_pool_size; ++i) {
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
		clReleaseMemObject(core_param.mempry_pool[i].data);
#else
		free(core_param.mempry_pool[i].data);
#endif	
		pthread_mutex_destroy(&core_param.mempry_pool[i].mutex);
	}
}

void wait_for_thread_dead(pthread_t tid)
{
	int timer = 1000;
	const struct timespec req = {0, 10000000};
	while (timer--) {
		int ret = pthread_kill(tid, 0);
		if (ESRCH == ret) {
			LOGI("the thread didn't exists or already quit.\n");
			return;
		} else if (EINVAL == ret) {
			LOGE("signal is invalid.\n");
			return;
		} else {
			nanosleep(&req, NULL);
			continue;
		}
	}
}

void clear_object_list()
{
	const int request_size = roundup_power_of_2(sizeof(list *));
	while (fifo_len(core_param.object_queue)) {
		while (fifo_get(core_param.object_queue, core_param.object_queue_write_buffer, request_size) != request_size);
		LOGI("clear object queue...\n");
		list *object_list = NULL;
		memcpy(&object_list, core_param.object_queue_write_buffer, sizeof(list *));
		free_detections(object_list);
	}
	LOGI("clear object queue over!\n");
}

void save_standard_image(void *image, int width, int height, const char *filename)
{
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	char *red = calloc(width * height, sizeof(unsigned char));
	if (!red) {
		LOGE("calloc fail.\n");
		return;
	}
	
	for (int i = 0; i < width * height; ++i) {
		red[i] = (unsigned char)(((float *)image)[i] * 255);
	}
	
	const int bits_per_pixel = 8;
	bitmap *bmp = create_bmp((const char *)red, width, height, bits_per_pixel);
	save_bmp(bmp, filename);
	free(red);
	delete_bmp(bmp);
#else
	char *rgb24 = calloc(width * height * 3, sizeof(unsigned char));
	if (!rgb24) {
		LOGE("calloc fail.\n");
		return;
	}

	cl_int errcode;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {width, height, 1};
	size_t row_pitch, slice_pitch;
	MEM_MAP_PTR_TYPE *h_image = clEnqueueMapImage(wrapper.command_queue, image, CL_TRUE, CL_MAP_READ,
		origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &errcode);

	row_pitch = row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float red = DEVICE_TO_HOST(h_image[y * row_pitch + (x << 2)]);
			float green = DEVICE_TO_HOST(h_image[y * row_pitch + (x << 2) + 1]);
			float blue = DEVICE_TO_HOST(h_image[y * row_pitch + (x << 2) + 2]);
			rgb24[y * width * 3 + x * 3 + 2] = (unsigned char)(red * 255);
			rgb24[y * width * 3 + x * 3 + 1] = (unsigned char)(green * 255);
			rgb24[y * width * 3 + x * 3 + 0] = (unsigned char)(blue * 255);
		}
	}

	clEnqueueUnmapMemObject(wrapper.command_queue, image, h_image, 0, NULL, NULL);
	
	const int bits_per_pixel = 24;
	bitmap *bmp = create_bmp((const char *)rgb24, width, height, bits_per_pixel);
	save_bmp(bmp, filename);
	free(rgb24);
	delete_bmp(bmp);
#endif
}