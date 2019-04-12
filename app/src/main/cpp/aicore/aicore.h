/** @file aicore.h
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

#ifndef _AICORE_H_
#define _AICORE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

#ifdef AICORE_BUILD_DLL
#ifdef _WIN32
#	define AICORE_EXPORT __declspec(dllexport)
#else
#	define AICORE_EXPORT __attribute__ ((visibility("default"))) extern
#endif
#else
#ifdef _WIN32
#	define AICORE_EXPORT __declspec(dllimport)
#else
#	define AICORE_EXPORT __attribute__ ((visibility("default")))
#endif
#endif

#define AIC_OK                              0
#define AIC_ALLOCATE_FAIL                  -1
#define AIC_FILE_NOT_EXIST                 -2
#define AIC_NETWORK_INIT_FAIL              -3
#define AIC_FIFO_ALLOC_FAIL                -4
#define AIC_IMAGE_STANDARDIZER_INIT_FAIL   -5
#define AIC_THREAD_CREATE_FAIL             -5
#define AIC_ENQUEUE_FAIL                   -6
#define AIC_DEQUEUE_FAIL                   -7
#define AIC_OPENCL_INIT_FAIL               -8
#define AIC_FRAME_DISCARD                  -9

/** @typedef enum class_t
 ** @brief 枚举物体类别
 **/
typedef enum {
	CORDYCEPS
} class_t;

/** @typedef struct object_t
 ** @brief 物体的结构体定义.
 **/
typedef struct {
	int x;				// 物体左上角横坐标.
	int y;				// 物体左上角纵坐标.
	int w;				// 物体宽度.
	int h;				// 物体高度.
	class_t classt;		// 类别ID.
	float objectness;	// 区域含有物体的概率.
	float probability;	// 类别概率.
} object_t;

/** @brief 初始化AICore模块.
 ** @param width 图像宽度.
 ** @param height 图像高度.
 ** @return 如果初始化成功,返回AIC_OK.
 **         如果初始化失败,返回错误码.
 **/
AICORE_EXPORT int ai_core_init(unsigned int width, unsigned int height);

/** @brief 将图像放入AICore模块的队列缓冲区.
 ** @param rgb24 RGB24格式图像数据.
 ** @param size 图像数据字节大小.
 ** @return 如果发送成功,返回AIC_OK.
 **         如果发送失败,返回错误码.
 **/
AICORE_EXPORT int ai_core_send_image(const char *const rgb24, size_t size);

/** @brief 将通过ION创建的图像放入AICore模块的队列缓冲区.请使用RGBA格式的图像.
 ** @param ion_filedesc ION文件描述符.
 ** @param ion_hostptr 指向通过ION分配的内存的主机端指针.
 ** @param width 图像宽度.
 ** @param height 图像高度.
 ** @return 如果发送成功,返回AIC_OK.
 **         如果发送失败,返回错误码.
 **/
AICORE_EXPORT int ai_core_send_ion_image(int ion_filedesc, void *const ion_hostptr, int width, int height);

/** @brief 从AICore模块获取检测到的物体.
 ** @param object 物体buffer.
 ** @param number 最多返回的物体个数.如果置信度超过阈值的物体多于需要返回的最大个数,则返回置信度最靠前的number个物体.
 ** @param threshold 候选物体的置信度阈值,取值范围为[0,1],值越高判断地越严格.
 ** @return 如果获取成功,返回实际检测到的物体个数.
 **         如果获取失败,返回错误码.
 **/
AICORE_EXPORT int ai_core_fetch_object(object_t *const object, size_t number, float threshold);

/** @brief 释放AICore模块所有资源.
 **/
AICORE_EXPORT void ai_core_free();

#ifdef __cplusplus
}
#endif

#endif