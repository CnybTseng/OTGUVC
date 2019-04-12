#ifndef _WINOGRAD_CONVOLUTION_H_
#define _WINOGRAD_CONVOLUTION_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#include "zutils.h"

typedef enum {
	F6x6_3x3, F4x4_3x3, F2x2_3x3
} WINOGRAD_CONV_TYPE;

#ifdef OPENCL
struct weight_transform_context;
typedef struct weight_transform_context weight_transform_context;

struct input_transform_context;
typedef struct input_transform_context input_transform_context;

struct matrix_multiplication_context;
typedef struct matrix_multiplication_context matrix_multiplication_context;

struct output_inverse_transform_context;
typedef struct output_inverse_transform_context output_inverse_transform_context;
#endif

AICORE_LOCAL int get_image_tile_size(WINOGRAD_CONV_TYPE conv);
AICORE_LOCAL int get_tile_output_size(WINOGRAD_CONV_TYPE conv);
#ifdef OPENCL
AICORE_LOCAL weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv, int filter_channels, int nfilters);
AICORE_LOCAL void get_transformed_weight_image_size(weight_transform_context *context, int *width, int *height);
AICORE_LOCAL void transform_weight(weight_transform_context *context, float *weights, float *biases, float *transformed_weights);
AICORE_LOCAL void free_weight_transform_context(weight_transform_context *context);
AICORE_LOCAL input_transform_context *create_input_transform_context(WINOGRAD_CONV_TYPE conv, int input_width,
	int input_height, int input_channels, int stride, int padding);
AICORE_LOCAL void get_input_image_size(input_transform_context *context, int *width, int *height);
AICORE_LOCAL void get_transformed_input_image_size(input_transform_context *context, int *width, int *height);
AICORE_LOCAL void transform_input(input_transform_context *context, float *transformed_input);
AICORE_LOCAL void free_input_transform_context(input_transform_context *context);
AICORE_LOCAL matrix_multiplication_context *create_matrix_multiplication_context(weight_transform_context *wtc, input_transform_context *itc);
AICORE_LOCAL void get_transformed_output_image_size(matrix_multiplication_context *context, int *width, int *height);
AICORE_LOCAL void multiply_transformed_matrix(matrix_multiplication_context *context, float *output);
AICORE_LOCAL void free_matrix_multiplication_context(matrix_multiplication_context *context);
AICORE_LOCAL output_inverse_transform_context *create_output_inverse_transform_context(matrix_multiplication_context *mmc, ACTIVATION act);
AICORE_LOCAL void get_inverse_transformed_output_image_size(output_inverse_transform_context *context, int *width, int *height);
AICORE_LOCAL void inverse_transform_output(output_inverse_transform_context *context, float *inverse_transformed_output);
AICORE_LOCAL void free_output_inverse_transform_context(output_inverse_transform_context *context);
AICORE_LOCAL void set_winograd_convolution_input(input_transform_context *context, void *input);
AICORE_LOCAL void *get_winograd_convolution_output(output_inverse_transform_context *context);
#endif

#ifdef __cplusplus
}
#endif

#endif