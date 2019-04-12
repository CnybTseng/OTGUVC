#ifndef _CL_COMMON_H_
#define _CL_COMMON_H_

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define cl_khr_fp16
#define HALF_MAX	0x1.ffcp15h

#define VECTOR_DATA_TYPE_STRING(data_type, size) data_type##size
#define VECTOR_DATA_TYPE(data_type, size) VECTOR_DATA_TYPE_STRING(data_type, size)
#define DATA_TYPE4 VECTOR_DATA_TYPE(DATA_TYPE, 4)

#define READ_WRITE_TYPE_STRING(read_write, type) read_write##type
#define READ_WRITE_TYPE(read_write, type) READ_WRITE_TYPE_STRING(read_write, type)
#define READ_IMAGE(image, coord) READ_WRITE_TYPE(read_image, READ_WRITE_DATA_TYPE)(image, coord)
#define READ_IMAGES(image, sampler, coord) READ_WRITE_TYPE(read_image, READ_WRITE_DATA_TYPE)(image, sampler, coord)
#define WRITE_IMAGE(image, coord, value) READ_WRITE_TYPE(write_image, READ_WRITE_DATA_TYPE)(image, coord, value)

#endif


