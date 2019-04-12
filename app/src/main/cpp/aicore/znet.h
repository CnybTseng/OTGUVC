#ifndef _ZNET_H_
#define _ZNET_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>
#include <stdlib.h>
#ifdef NNPACK
#	include "nnpack.h"
#endif
#include "list.h"
#ifdef OPENCL
#	include "CL/opencl.h"
#endif
#include "zutils.h"

/** @typedef enum WORK_MODE
 ** @brief 枚举znet的工作模式.
 **/
typedef enum {
	INFERENCE,	// 推断模式
	TRAIN		// 训练模式
} WORK_MODE;

/** @typedef enum LAYER_TYPE
 ** @brief 枚举卷积网络层类型.
 **/
typedef enum {
	CONVOLUTIONAL,	// 卷积层
	MAXPOOL,		// 最大池化层
	ROUTE,			// 路线层
	RESAMPLE,		// 重采样层
	YOLO			// 瞄一眼层
} LAYER_TYPE;

/** @typedef enum ACTIVATION
 ** @brief 枚举神经元激活类型.
 **/
typedef enum {
	RELU,			// 线性整流激活
	LEAKY,			// 泄漏线性整流激活
	LINEAR,			// 线性激活
	LOGISTIC		// 逻辑斯蒂激活
} ACTIVATION;

/** @typedef struct data_store
 ** @brief 数据商店的结构.
 **/
typedef struct {
	
} data_store;

/** @typedef struct train_options
 ** @brief 训练参数设置的结构.
 **/
typedef struct {
	
} train_options;

struct znet;
typedef struct znet znet;

typedef struct {
	int w;	// 宽度
	int h;	// 高度
	int c;	// 通道数
} dim3;

typedef struct {
	int w;			// 宽度
	int h;			// 高度
	int c;			// 通道数
	float *data;	// 数据
#ifdef OPENCL
	cl_mem d_data;	// CPU和GPU的共享数据
#endif
} image;

typedef struct {
	float x;	// 中心横坐标
	float y;	// 中心纵坐标
	float w;	// 框框宽度
	float h;	// 框框高度
} box;

typedef struct {
	box bbox;					// 框框
	int classes;				// 物体的类别数量
	float *probabilities;		// 物体的类别概率分布
	float objectness;			// 物体存在的概率
} detection;

/** @name 卷积网络层的创建.目前仅支持卷积层,最大池化层,重采样层,路线层和瞄一眼层.
 ** @ { */
/** @brief 创建卷积层.
 ** @param activation 神经元激活类型.
 ** @param input_size 输入的三维张量的尺寸.
 ** @param filter_size 滤波器尺寸.
 ** @param nfilters	滤波器的个数.
 ** @param stride 卷积移动步长.
 ** @param padding 填充量.
 ** @param batch_size 批量大小.
 ** @param batch_norm 是否批正规化.
 ** @param output_size 输出的三维张量的大小.
 ** @return 返回卷积层实例指针.
 **/
AICORE_LOCAL void *make_convolutional_layer(ACTIVATION activation, dim3 input_size, int filter_size, int nfilters,
                               int stride, int padding, int batch_size, int batch_norm, dim3 *output_size);

/** @brief 创建最大池化层.
 ** @param input_size 输入的三维张量的尺寸.
 ** @param filter_size 滤波器尺寸.
 ** @param stride 滤波器移动步长.
 ** @param padding 填充量.
 ** @param batch_size 批量大小.
 ** @param output_size 输出的三维张量的大小.
 ** @return 返回最大池化层实例指针.
 **/ 
AICORE_LOCAL void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size);
						 
/** @brief 创建瞄一眼层.
 ** @param input_size 输入的三维张量的尺寸.
 ** @param batch_size 批量大小.
 ** @param nscales 本层需要检测物体的尺寸等级数量.
 ** @param total_scales 所有瞄一眼层需要检测物体的尺寸等级总数.
 ** @param classes 需要检测的物体类别总数.
 ** @param mask 尺寸聚类中心的掩码.
 ** @param anchor_boxes 定标框框.
 ** @return 返回瞄一眼层实例指针.
 **/ 
AICORE_LOCAL void *make_yolo_layer(dim3 input_size, int batch_size, int nscales, int total_scales, int classes, int *mask,
                      int *anchor_boxes);
					  
/** @brief 创建路线层.
 ** @param batch_size 批量大小.
 ** @param nroutes 连接到本层的路线数量.
 ** @param layers 连接到本层的层.
 ** @param layer_id 连接到本层的层的识别号.
 ** @param output_size 输出的三维张量的大小.
 ** @return 返回路线层实例的指针.
 **/ 
AICORE_LOCAL void *make_route_layer(int batch_size, int nroutes, void *layers[], int *layer_id, dim3 *output_size);

/** @brief 创建重采样层.
 ** @param input_size 输入的三维张量的尺寸.
 ** @param batch_size 批量大小.
 ** @param stride 采样器步长.
 ** @param output_size 输出的三维张量的大小.
 ** @return 返回重采样层实例的指针.
 **/
AICORE_LOCAL void *make_resample_layer(dim3 input_size, int batch_size, int stride, dim3 *output_size);
/** @ }*/



/** @name 卷积网络的创建,训练,推断,销毁,查询等操作.
 ** @ { */
/** @brief 创建卷积神经网络.
 ** @param layers 神经网络层列表.
 ** @param nlayers 神经网络层列表长度.
 ** @param filename 权重文件名.
 ** @return 卷积神经网络实例指针.
 **/
AICORE_LOCAL znet *znet_create(void *layers[], int nlayers, const char *weight_filename);

/** @brief 训练卷积神经网络.本接口暂未实现.
 ** @param net 卷积神经网络实例指针.
 ** @param ds 数据商店实例指针.
 ** @param opts 训练参数设置.
 **/
AICORE_LOCAL void znet_train(znet *net, data_store *ds, train_options *opts);

/** @brief 经卷积神经网络推断.
 ** @param net 卷积神经网络实例指针.
 ** @param input 输入的图像.
 ** @return 返回推断输出.
 **/
AICORE_LOCAL float *znet_inference(znet *net, void *input);

/** @brief 摧毁卷积神经网络实例.
 ** @param net 卷积神经网络实例指针.
 **/
AICORE_LOCAL void znet_destroy(znet *net);

/** @brief 打印卷积神经网络架构信息. 
 ** @param net 卷积神经网络实例指针.
 **/
AICORE_LOCAL void znet_architecture(znet *net);

/** @brief 获取卷积神经网络的工作模式. 
 ** @param net 卷积神经网络实例指针.
 ** @return 卷积神经网络的工作模式.
 **/
AICORE_LOCAL WORK_MODE znet_workmode(znet *net);

/** @brief 获取卷积神经网络的所有层.
 ** @param net 卷积神经网络实例指针.
 ** @return 返回卷积神经网络的所有层.
 **/
AICORE_LOCAL void **znet_layers(znet *net);

#ifdef NNPACK
/** @brief 获取卷积神经网络的线程池句柄.
 ** @param net 卷积神经网络实例指针.
 ** @return 返回卷积神经网络的线程池句柄.
 **/
AICORE_LOCAL pthreadpool_t znet_threadpool(znet *net);
#endif

/** @brief 获取卷积神经网络输入层的宽度.
 ** @param net 卷积神经网络实例指针
 ** @return 返回卷积神经网络输入层的宽度.
 **/
AICORE_LOCAL int znet_input_width(znet *net);

/** @brief 获取卷积神经网络输入层的高度.
 ** @param net 卷积神经网络实例指针.
 ** @return 返回卷积神经网络输入层的高度.
 **/
AICORE_LOCAL int znet_input_height(znet *net);
/** @ } */



/** @name 获取和释放物体识别结果.
 ** @ { */
/** @brief 获取物体的检测结果.
 ** @param net 卷积神经网络实例指针.
 ** @param thresh 判别物体存在的概率阈值.
 ** @param width 输入卷积神经网络图像的宽度.
 ** @param height 输入卷积神经网络图像的高度.
 ** @return 返回物体检测结果的链表.
 **/
AICORE_LOCAL list *get_detections(znet *net, float thresh, int width, int height);

/** @name 轻柔的非最大值抑制. 
 ** @param l 未经非最大值抑制的物体检测结果的链表.
 ** @param sigma 非最大值抑制力度.
 ** @return 经非最大值抑制的物体检测结果的链表.
 **/
AICORE_LOCAL list *soft_nms(list *l, float sigma);

/** @brief 释放物体检测结果占用的内存.
 ** @param l 物体检测结果的链表.
 **/
AICORE_LOCAL void free_detections(list *l);
/** @ } */

#ifdef __cplusplus
}
#endif

#endif