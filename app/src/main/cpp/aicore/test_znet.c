#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <sys/stat.h>
#include "znet.h"
#include "im2col.h"
#include "zutils.h"
#include "gemm.h"
#include "activation.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "resample_layer.h"
#include "bitmap.h"
#include "image.h"
#include "list.h"
#include "box.h"
#include "agriculture.names"
#ifdef __ARM_NEON__
	#include <arm_neon.h>
#endif
#ifdef NNPACK
#	include "nnpack.h"
#endif
#include "winograd_convolution.h"
#include "cl_wrapper.h"
#include "half.h"

typedef struct {
	bitmap *original;
	image *standard;
} test_image;

#ifdef __ARM_NEON__							   
void test_resize_faster(int argc, char *argv[]);
#endif
#ifdef OPENCL
cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(utils, cl);
extern char BINARY_FILENAME_TO_END(utils, cl);
#endif

extern void set_fill_color(float (*color)[4], int nchannels);
test_image load_test_image(int argc, char *argv[], int std_width, int std_height);
void draw_detections(bitmap *bmp, list *detections, char *names[], float thresh,
	int roix, int roiy, int roiw, int roih);
void test_multi_free(int argc, char *argv[]);
void test_im2col(int argc, char *argv[]);
void test_gemm(int argc, char *argv[]);
void test_activate(int argc, char *argv[]);
void test_convolutional_layer(int argc, char *argv[]);
void test_maxpool_layer(int argc, char *argv[]);
void test_mset(int argc, char *argv[]);
void test_mcopy(int argc, char *argv[]);
void test_bmp(int argc, char *argv[]);
void test_split(int argc, char *argv[]);
void test_resize(int argc, char *argv[]);
void test_embed(int argc, char *argv[]);
void test_standard(int argc, char *argv[]);
void test_list(int argc, char *argv[]);
void test_split_sse(int argc, char *argv[]);
void test_split_compare(int argc, char *argv[]);
void test_resize_compare(int argc, char *argv[]);
void test_activate_neon(int argc, char *argv[]);
void test_nnpack(int argc, char *argv[]);
void test_winograd_weight_transformation(int argc, char *argv[]);
void test_winograd_input_transformation(int argc, char *argv[]);
void test_winograd_convolution(int argc, char *argv[]);
void test_normalize_image_with_gpu(int argc, char *argv[]);
void test_maxpool_layer_with_gpu(int argc, char *argv[]);
void test_direct_convolution(int argc, char *argv[]);
void test_resample_layer_with_gpu(int argc, char *argv[]);
void test_nhwc_to_nchw(int argc, char *argv[]);
void test_image_standardizer(int argc, char *argv[]);
void test_yolov3_tiny_with_cpu_or_gpu(int argc, char *argv[]);
void test_ion_image_standardizer(int argc, char *argv[]);
void test_half(int argc, char *argv[]);
void test_read_half_value_from_float_image(int argc, char *argv[]);
void test_set_color(int argc, char *argv[]);

int main_bak(int argc, char *argv[])
{
	system("rm -f *.cl.bin");
	test_yolov3_tiny_with_cpu_or_gpu(argc, argv);
	
	return 0;
}

test_image load_test_image(int argc, char *argv[], int std_width, int std_height)
{
	test_image ti = {NULL, NULL};
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return ti;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return ti;
	}
	
	int rsz_width, rsz_height;
	if (std_width / (float)width < std_height / (float)height) {
		rsz_width = std_width;
		rsz_height = (int)(height * std_width / (float)width);
	} else {
		rsz_width = (int)(width * std_height / (float)height);
		rsz_height = std_height;
	}
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		free(splited);
		return ti;
	}
	
	int pitch = get_bmp_pitch(bmp);
	split_channel(data, splited, pitch, width, height);
	resize_image(splited, rsz_splited, width, height, rsz_width, rsz_height, nchannels);
	
	image *standard = create_image(std_width, std_height, nchannels);
	if (!standard) {
		fprintf(stderr, "create_image[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		free(splited);
		free(rsz_splited);
		return ti;
	}
	
	set_image(standard, 0.5);
	embed_image(rsz_splited, standard, rsz_width, rsz_height);
	
	free(splited);
	free(rsz_splited);
	
	ti.original = bmp;
	ti.standard = standard;
	
	return ti;
}

void draw_detections(bitmap *bmp, list *detections, char *names[], float thresh,
	int roix, int roiy, int roiw, int roih)
{
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	int nchannels = bit_count >> 3;
	node *n = detections->head;
	int color[3] = {0, 255, 255};
	const int lw = 3;
	while (n) {
		detection *det = (detection *)n->val;
		if (det->objectness < thresh) {
			n = n->next;
			continue;
		}
		
		printf("objectness:%.5f ", det->objectness);
		int maybe = 0;
		for (int i = 0; i < det->classes; ++i) {
			if (det->probabilities[i] < thresh) continue;
			if (maybe > 0) printf(",");
			printf("%s:%.0f%%", names[i], det->probabilities[i] * 100);
			++maybe;
		}
		
		if (maybe) printf("\n");
		int left    = (int)((det->bbox.x - det->bbox.w / 2) * roiw) + roix;		
		int right   = (int)((det->bbox.x + det->bbox.w / 2) * roiw) + roix;
		int _top    = (int)((det->bbox.y - det->bbox.h / 2) * roih) + roiy;
		int _bottom = (int)((det->bbox.y + det->bbox.h / 2) * roih) + roiy;
		int top = height - 1 - _bottom;
		int bottom = height - 1 - _top;
		
		if (left < 0) left = 0;
		if (left > width - 1) left = width - 1;
		if (right < 0) right = 0;
		if (right > width - 1) right = width - 1;
		if (top < 0) top = 0;
		if (top > height - 1) top = height - 1;
		if (bottom < 0) bottom = 0;
		if (bottom > height - 1) bottom = height - 1;
		
		for (int c = 0; c < nchannels; ++c) {
			for (int y = top; y < bottom; ++y) {
				for (int l = 0; l < lw; ++l) {
					data[y * pitch + (left + l) * nchannels + c] = color[c];
					data[y * pitch + (right - l) * nchannels + c] = color[c];
				}
			}
			
			for (int x = left; x < right; ++x) {
				for (int l = 0; l < lw; ++l) {
					data[(top + l) * pitch + x * nchannels + c] = color[c];
					data[(bottom - l) * pitch + x * nchannels + c] = color[c];
				}
			}
		}
		
		n = n->next;
	}
}

void test_multi_free(int argc, char *argv[])
{
	char *buf1 = (char *)malloc(1024);
	if (!buf1) {
		return;
	}
	
	int *buf2 = (int *)malloc(1024);
	if (!buf2) {
		return mmfree(1, buf1);
	}
	
	float *buf3 = (float *)malloc(1024);
	if (!buf3) {
		return mmfree(2, buf1, buf2);
	}
	
	mmfree(3, buf1, buf2, buf3);
}

void test_im2col(int argc, char *argv[])
{
	int width = 8;
	int height = 8;
	int nchannels = 3;
	int fsize = 3;
	int stride = 2;
	int padding = 1;
	
	float *image = (float *)malloc(width * height * nchannels * sizeof(float));
	if (!image) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		exit(-1);
	}
	
	int convw = (width + 2 * padding - fsize) / stride + 1;
	int convh = (height + 2 * padding - fsize) / stride + 1;
	
	float *matrix = (float *)malloc(fsize * fsize * nchannels * convw * convh * sizeof(float));
	if (!matrix) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		mmfree(1, image);
		exit(-1);
	}
		
	for (int c = 0; c < nchannels; c++) {
		for (int i = 0; i < width * height; i++) {
			image[i + c * width * height] = 1 + i + c * width * height;
		}
	}
	
	im2col_cpu(image, width, height, nchannels, fsize, stride, padding, matrix);
	
	FILE *fp = fopen("matrix.txt", "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		mmfree(2, image, matrix);
		exit(-1);
	}
	
	for (int y = 0; y < fsize * fsize * nchannels; y++) {
		for (int x = 0; x < convw * convh; x++) {
			fprintf(fp, "%.0f\t", matrix[y * convw * convh + x]);
		}
		fputs("\n", fp);
	}
	
	fclose(fp);
	mmfree(2, image, matrix);
}

void test_gemm(int argc, char *argv[])
{
#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
#endif	
	
	int ah = 1024;
	int aw = 1024;
	int bh = 1024;
	int bw = 1024;
	
	float *A = (float *)malloc(aw * ah * sizeof(float));
	if (!A) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		exit(-1);
	}
	
	float *B = (float *)malloc(bw * bh * sizeof(float));
	if (!B) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		mmfree(1, A);
		exit(-1);
	}
	
	int transa = 0;
	int transb = 0;
	
	int cw, ch;
	if (!transa && !transb) {
		cw = bw;
		ch = ah;
	} else if (!transa && transb) {
		cw = bh;
		ch = ah;
	} else if (transa && !transb) {
		cw = bw;
		ch = aw;
	} else {
		cw = bh;
		ch = aw;
	}
	
	gemm_context *gc = create_gemm_context(transa, transb, ch, cw, aw);
	
	float *C = (float *)malloc(cw * ch * sizeof(float));
	if (!C) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		mmfree(2, A, B);
		exit(-1);
	}
	
	srand(time(NULL));
	int na = ah * aw;
	for (int i = 0; i < na; i++) {
		A[i] = (float)rand() / RAND_MAX;
	}
	
	int nb = bh * bw;
	for (int i = 0; i < nb; i++) {
		B[i] = (float)rand() / RAND_MAX;
	}
	
	int nc = ch * cw;
	for (int i = 0; i < nc; i++) {
		C[i] = 1;
	}
	
	int N = 10000;
	if (argc > 1) N = atoi(argv[1]);
	printf("gemm iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		gemm(gc, transa, transb, ch, cw, aw, 0.56, A, aw, B, bw, 0.84, C, cw);
	gettimeofday(&t2, NULL);
	float duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("duration: %f ms.\n", duration);
	
	FILE *fp = fopen("matrix.txt", "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		mmfree(3, A, B, C);
		exit(-1);
	}
	
	for (int i = 0; i < ah; i++) {
		for (int j = 0; j < aw; j++) {
			fprintf(fp, "%.8f ", A[i * aw + j]);
		}
		fputs(";", fp);
		fputs("\n", fp);
	}
	
	fputs("\n", fp);
	for (int i = 0; i < bh; i++) {
		for (int j = 0; j < bw; j++) {
			fprintf(fp, "%.8f ", B[i * bw + j]);
		}
		fputs(";", fp);
		fputs("\n", fp);
	}
	
	fputs("\n", fp);
	for (int i = 0; i < ch; i++) {
		for (int j = 0; j < cw; j++) {
			fprintf(fp, "%.8f ", C[i * cw + j]);
		}
		fputs(";", fp);
		fputs("\n", fp);
	}
	
	fclose(fp);
	mmfree(3, A, B, C);
	free_gemm_context(gc);
	
#ifdef OPENCL
	cl_destroy_wrapper(wrapper);
#endif
}

void test_activate(int argc, char *argv[])
{
	float output[16];
	srand(time(NULL));
	for (int i = 0; i < 16; i++) {
		output[i] = 2 * (rand() / (double)RAND_MAX - 0.5);
		printf("%.5f ", output[i]);
	}
	
	activate(output, 16, LEAKY);
	
	printf("\n");
	for (int i = 0; i < 16; i++) {
		printf("%.5f ", output[i]);
	}
}

void test_convolutional_layer(int argc, char *argv[])
{
	dim3 input_size = {26, 26, 3};
	float *input = (float *)malloc(input_size.w * input_size.h * input_size.c * sizeof(float));
	if (!input) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	dim3 output_size;
	void *layers[] = {
		make_convolutional_layer(LINEAR, input_size, 3, 512, 1, 1, 1, 0, &output_size)};
	
	znet *net = znet_create(layers, 1, "coco.weights");
	znet_architecture(net);
	
	convolutional_layer *layer = (convolutional_layer *)layers[0];
	
	srand(time(NULL));
	for (int i = 0; i < layer->ninputs; ++i) {
		input[i] = (rand() / (double)RAND_MAX - 0.5) * 2;
	}
	
	int size = layer->filter_size * layer->filter_size * layer->input_size.c;
	for (int i = 0; i < layer->nfilters; ++i) {
		for (int j = 0; j < size; ++j) {
			layer->weights[i * size + j] = 1;
		}
	}
	
	layer->input = input;
	forward_convolutional_layer(layer, net);
	
	FILE *fp = fopen("convolution.txt", "w");
	
	for (int c = 0; c < input_size.c; c++) {
		for (int y = 0; y < input_size.h; y++) {
			for (int x = 0; x < input_size.w; x++) {
				int id = c * input_size.w * input_size.h + y * input_size.w + x;
				if (layer->input[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->input[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fputs("-----------------------------------------\n", fp);
	fputs("-----------------------------------------\n", fp);
	for (int c = 0; c < output_size.c; c++) {
		for (int y = 0; y < output_size.h; y++) {
			for (int x = 0; x < output_size.w; x++) {
				int id = c * output_size.w * output_size.h + y * output_size.w + x;
				if (layer->output[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->output[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fclose(fp);
	znet_destroy(net);
	free(input);
}

void test_maxpool_layer(int argc, char *argv[])
{
	dim3 input_size = {27, 27, 3};
	float *input = (float *)malloc(input_size.w * input_size.h * input_size.c * sizeof(float));
	if (!input) {
		fprintf(stderr, "malloc[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	dim3 output_size;
	void *layers[] = {make_maxpool_layer(input_size, 3, 3, 0, 1, &output_size)};
	
	znet *net = znet_create(layers, 1, "coco.weights");
	znet_architecture(net);
	
	maxpool_layer *layer = (maxpool_layer *)layers[0];
	
	srand(time(NULL));
	for (int i = 0; i < layer->ninputs; ++i) {
		input[i] = (rand() / (double)RAND_MAX - 0.5) * 2;
	}
	
	layer->input = input;
	forward_maxpool_layer(layer, net);
	
	FILE *fp = fopen("maxpool.txt", "w");
	
	for (int c = 0; c < input_size.c; c++) {
		for (int y = 0; y < input_size.h; y++) {
			for (int x = 0; x < input_size.w; x++) {
				int id = c * input_size.w * input_size.h + y * input_size.w + x;
				if (layer->input[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->input[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fputs("-----------------------------------------\n", fp);
	fputs("-----------------------------------------\n", fp);
	for (int c = 0; c < output_size.c; c++) {
		for (int y = 0; y < output_size.h; y++) {
			for (int x = 0; x < output_size.w; x++) {
				int id = c * output_size.w * output_size.h + y * output_size.w + x;
				if (layer->output[id] > 0) fputs(" ", fp);
				fprintf(fp, "%.5f ", layer->output[id]);
			}
			fputs("\n", fp);
		}
		fputs("-----------------------------------------\n", fp);
	}
	
	fclose(fp);
	znet_destroy(net);
	free(input);
}

void test_mset(int argc, char *argv[])
{
	float X[128];
	float val = 3.14159;
	
	mset((char *const)X, sizeof(X), (const char *const)&val, sizeof(float));
	
	for (int i = 0; i < 128; ++i) {
		printf("%.5f ", X[i]);
	}
}

void test_mcopy(int argc, char *argv[])
{
	float X[] = {1.111 ,2.222, 3.333, 4.444, 5.555};
	float Y[5];
	
	mcopy((const char *const)X, (char *const)Y, sizeof(X));
	
	for (int i = 0; i < 5; ++i) {
		printf("%f ", Y[i]);
	}
}

void test_bmp(int argc, char *argv[])
{
	bitmap *bmp = read_bmp("dog.bmp");
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	
	save_bmp(bmp, "girl.bmp");
	delete_bmp(bmp);
}

void test_split(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);
	split_channel(data, splited, pitch, width, height);
	
	FILE *fp = fopen("split_channel.txt", "w");
	for (int i = 0; i < width * nchannels; ++i) {
		fprintf(fp, "%u ", data[i + width * (height - 1) * 3]);
	}
	
	fputs("\n\n", fp);
	for (int c = 0; c < nchannels; ++c) {
		for (int i = 0; i < width; ++i) {
			fprintf(fp, "%u ", splited[i + c * width * height]);
		}
		fputs("\n", fp);
	}
	
	char *red = calloc(width * height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		fclose(fp);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < width * height; ++i) {
		red[i] = splited[i];
	}
	
	bitmap *red_bmp = create_bmp(red, width, height, 8);
	save_bmp(red_bmp, "splited.bmp");
	
	fclose(fp);
	free(splited);
	free(red);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}

void test_resize(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);	
	split_channel(data, splited, pitch, width, height);
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		resize_image_hv(splited, rsz_splited, width, height, rsz_width, rsz_height, nchannels);
	gettimeofday(&t2, NULL);
	float duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("duration: %f ms.\n", duration);
	
	char *red = calloc(rsz_width * rsz_height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < rsz_width * rsz_height; ++i) {
		red[i] = rsz_splited[i];
	}
	
	bitmap *red_bmp = create_bmp(red, rsz_width, rsz_height, 8);
	save_bmp(red_bmp, "resized.bmp");
	
	free(red);
	free(splited);
	free(rsz_splited);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}

void test_embed(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);
	split_channel(data, splited, pitch, width, height);
	resize_image(splited, rsz_splited, width, height, rsz_width, rsz_height, nchannels);
	
	image *standard = create_image(416, 416, nchannels);
	if (!standard) {
		fprintf(stderr, "create_image[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		delete_bmp(bmp);
		return;
	}
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("embed iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		embed_image(rsz_splited, standard, rsz_width, rsz_height);
	gettimeofday(&t2, NULL);
	float duration1 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("without simd: %f ms.\n", duration1);
	
	memset(standard->data, 0, standard->w * standard->h * nchannels * sizeof(float));

	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
#ifdef __ARM_NEON__
		embed_image_neon(rsz_splited, standard, rsz_width, rsz_height);
#endif
	gettimeofday(&t2, NULL);
	float duration2 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("with simd: %f ms.\n", duration2);
	printf("speed-up:%f\n", duration1 / duration2);
	
	char *red = calloc(standard->w * standard->h, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		free_image(standard);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < standard->w * standard->h; ++i) {
		red[i] = (char)(standard->data[i] * 255);
	}
	
	bitmap *red_bmp = create_bmp(red, standard->w, standard->h, 8);
	save_bmp(red_bmp, "standard.bmp");
	
	free(red);
	free(splited);
	free(rsz_splited);
	free_image(standard);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}

void test_standard(int argc, char *argv[])
{
	image *standard = create_image(416, 416, 3);
	FILE *fp = fopen("standard.bin", "rb");
	fread(standard->data, sizeof(float), 416 * 416 * 3, fp);
	fclose(fp);
	
	char *red = calloc(standard->w * standard->h, sizeof(char));	
	for (int i = 0; i < standard->w * standard->h; ++i) {
		red[i] = (char)(standard->data[i] * 255);
	}
	
	bitmap *bmp = create_bmp(red, standard->w, standard->h, 8);
	save_bmp(bmp, "red.bmp");
	
	free(red);
	delete_bmp(bmp);
	free_image(standard);
}

void test_list(int argc, char *argv[])
{
	list *detections = make_list();
	if (!detections) return;
	
	int bx = 5;
	int by = 6;
	int bw = 7;
	int bh = 8;
	
	srand(time(NULL));
	for (int i = 0; i < 10; ++i) {
		detection *det = list_alloc_mem(sizeof(detection));
		if (!det) break;
		det->bbox.x = (i + 1) * bx;
		det->bbox.y = (i + 1) * by;
		det->bbox.w = (i + 1) * bw;
		det->bbox.h = (i + 1) * bh;
		det->classes = 80;
		det->probabilities = calloc(det->classes, sizeof(float));
		if (!det->probabilities) break;
		for (int j = 0; j < det->classes; ++j)
			det->probabilities[j] = rand() / (double)RAND_MAX;
		det->objectness = rand() / (double)RAND_MAX;
		if (list_add_tail(detections, det)) break;
	}
		
	int count = 0;
	node *nd = detections->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		printf("%d   %.2f:%.2f:%.2f:%.2f   %d   ", ++count, det->bbox.x, det->bbox.y, det->bbox.w,
			det->bbox.h, det->classes);
		for (int i = 0; i < det->classes; ++i)
			printf("%.2f:", det->probabilities[i]);
		printf("   %f  %p\n\n\n", det->objectness, nd);
		nd = nd->next;
	}
	
	printf("head=%p, tail=%p\n", detections->head, detections->tail);
	printf("//////////////////////////////////////////////////\n");
	detection deleted[3];
	for (int i = 0; i < 3; ++i) {
		deleted[i].bbox.x = (i + 3) * bx;
		deleted[i].bbox.y = (i + 3) * by;
		deleted[i].bbox.w = (i + 3) * bw;
		deleted[i].bbox.h = (i + 3) * bh;
		list_del_node(detections, &deleted[i], equ_val, free_val);
	}
	
	count = 0;
	nd = detections->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		printf("%d   %.2f:%.2f:%.2f:%.2f   %d   ", ++count, det->bbox.x, det->bbox.y, det->bbox.w,
			det->bbox.h, det->classes);
		for (int i = 0; i < det->classes; ++i)
			printf("%.2f:", det->probabilities[i]);
		printf("   %f  %p\n\n\n", det->objectness, nd);
		nd = nd->next;
	}
	
	printf("head=%p, tail=%p\n", detections->head, detections->tail);
	nd = detections->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		if (det->probabilities) {
			free(det->probabilities);
			det->probabilities = NULL;
		}
		nd = nd->next;
	}
	
	list_clear(detections);
}

void test_split_sse(int argc, char *argv[])
{
#ifdef __INTEL_SSE__
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u, pitch %d.\n", width, height, bit_count, pitch);
	char *splited = calloc(width * height * 3, 1);
	
	split_channel_sse(data, (unsigned char *)splited, pitch, width, height);

	bitmap *red = create_bmp(splited + 0 * width * height, width, height, 8);
	save_bmp(red, "splited.bmp");
	
	free(splited);
	delete_bmp(bmp);
	delete_bmp(red);
#endif
}

void test_split_compare(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u, pitch %d.\n", width, height, bit_count, pitch);
	unsigned char *splited = calloc(width * height * 3, 1);
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("split iterations %d\n", N);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i)
		split_channel(data, splited, pitch, width, height);
	gettimeofday(&t2, NULL);
	float duration1 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("without simd: %f ms.\n", duration1);
	
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
#ifdef __INTEL_SSE__
		split_channel_sse(data, splited, pitch, width, height);
#endif
#ifdef __ARM_NEON__
		split_channel_neon(data, splited, pitch, width, height);
#endif
	}
	gettimeofday(&t2, NULL);
	float duration2 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("with simd: %f ms.\n", duration2);
	printf("speed-up:%f\n", duration1 / duration2);
	
	bitmap *red = create_bmp((char *)splited, width, height, 8);
	save_bmp(red, "splited.bmp");
	
	free(splited);
	delete_bmp(bmp);
	delete_bmp(red);
}

void test_resize_compare(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	unsigned char *data = get_bmp_data(bmp);
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int pitch = get_bmp_pitch(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	printf("bitmap: width %u, height %u, bit_count %u, pitch %d.\n", width, height, bit_count, pitch);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * 3, sizeof(unsigned char));
	split_channel(data, splited, pitch, width, height);
	bitmap *red = create_bmp((char *)splited, width, height, 8);
	save_bmp(red, "splited.bmp");

	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *resized = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("resize iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		resize_image(splited, resized, width, height, rsz_width, rsz_height, nchannels);
	}
	gettimeofday(&t2, NULL);
	float duration1 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("without simd: %f ms.\n", duration1);
	
	memset(resized, 0, rsz_width * rsz_height * nchannels * sizeof(unsigned char));
	
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
#ifdef __ARM_NEON__
		resize_image_neon((uint8_t *)splited, resized, width, height, rsz_width, rsz_height, nchannels);
#endif
	}
	gettimeofday(&t2, NULL);
	float duration2 = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("with simd: %f ms.\n", duration2);
	printf("speed-up:%f\n", duration1 / duration2);
	
	bitmap *resized_bmp = create_bmp((char *)resized, rsz_width, rsz_height, 8);
	save_bmp(resized_bmp, "resized.bmp");
	
	free(resized);
	free(splited);
	delete_bmp(bmp);
	delete_bmp(red);
	delete_bmp(resized_bmp);
}

#ifdef __ARM_NEON__
void test_resize_faster(int argc, char *argv[])
{
	bitmap *bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	int width = get_bmp_width(bmp);
	int height = get_bmp_height(bmp);
	int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	printf("bitmap: width %u, height %u, bit_count %u.\n", width, height, bit_count);
	int nchannels = bit_count >> 3;
	
	unsigned char *splited = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		delete_bmp(bmp);
		return;
	}
	
	float sx = 416.0f / width;
	float sy = 416.0f / height;
	float s = sx < sy ? sx : sy;
	int rsz_width = (int)(width * s);
	int rsz_height = (int)(height * s);
	
	unsigned char *rsz_splited = calloc(rsz_width * rsz_height * nchannels, sizeof(unsigned char));
	if (!rsz_splited) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		delete_bmp(bmp);
		return;
	}
	
	int pitch = get_bmp_pitch(bmp);	
	split_channel(data, splited, pitch, width, height);
	
	short *x_tab = calloc(rsz_width * rsz_height, sizeof(short));
	short *y_tab = calloc(rsz_width * rsz_height, sizeof(short));
	unsigned short *dx_tab = calloc(rsz_width * rsz_height, sizeof(unsigned short));
	unsigned short *dy_tab = calloc(rsz_width * rsz_height, sizeof(unsigned short));
	unsigned char *pack = calloc(rsz_width * rsz_height * 4 * nchannels, sizeof(unsigned char));
	
	make_bilinear_interp_table(width, height, rsz_width, rsz_height, x_tab, y_tab, dx_tab, dy_tab);
		
	int N = 10000;
	if (argc > 2) N = atoi(argv[2]);
	printf("faster resize iterations %d\n", N);
	
	struct timeval t1, t2; 
	gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		package_neighbor_pixle(width, height, rsz_width, rsz_height, x_tab, y_tab, splited, pack);
		resize_image_neon_faster(pack, rsz_splited, rsz_width, rsz_height, nchannels, dx_tab, dy_tab);
	}
	gettimeofday(&t2, NULL);
	float duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("duration: %f ms.\n", duration);
	
	char *red = calloc(rsz_width * rsz_height, sizeof(char));
	if (!red) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(splited);
		free(rsz_splited);
		delete_bmp(bmp);
		return;
	}
	
	for (int i = 0; i < rsz_width * rsz_height; ++i) {
		red[i] = rsz_splited[i];
	}
	
	bitmap *red_bmp = create_bmp(red, rsz_width, rsz_height, 8);
	save_bmp(red_bmp, "resized.bmp");
	
	free(x_tab);
	free(y_tab);
	free(dx_tab);
	free(dy_tab);
	free(pack);
	free(red);
	free(splited);
	free(rsz_splited);
	delete_bmp(bmp);
	delete_bmp(red_bmp);
}
#endif

void test_activate_neon(int argc, char *argv[])
{
	float X[16];
	srand(time(NULL));
	for (int i = 0; i < 16; ++i) {
		X[i] = 2 * (rand() / (float)RAND_MAX - 0.5) + 0.5;
		printf("%.5f ", X[i]);
	}
	printf("\n");
	
	activate(X, 16, LEAKY);
	for (int i = 0; i < 16; ++i)
		printf("%.5f ", X[i]);
	printf("\n");
}

#ifdef NNPACK
void test_nnpack(int argc, char *argv[])
{
	pthreadpool_t threadpool = pthreadpool_create(1);
	if (nnp_initialize() != nnp_status_success) {
		fprintf(stderr, "nnp_initialize fail!\n");
		pthreadpool_destroy(threadpool);
		return;
	}
	
	int xreso = 8;
	int yreso = 8;
	struct nnp_size input_size = {xreso, yreso};
	struct nnp_padding input_padding = {1, 1, 1, 1};
	struct nnp_size kernel_size = {3, 3};
	struct nnp_size stride = {1, 1};

	srand(time(NULL));
	
	float *input = calloc(xreso * yreso * 3, sizeof(float));
	for (int c = 0; c < 3; ++c) {
		float *ptr = input + c * xreso * yreso;
		for (int y = 0; y < yreso; ++y) {
			for (int x = 0; x < xreso; ++x) {
				ptr[y * xreso + x] = 2 * (rand() / (float)RAND_MAX - 0.5) + 0.5;
			}
		}
	}
	
	int nfilters = 64;
	float *weights = calloc(3 * 3 * 3 * nfilters, sizeof(float));
	for (int i = 0; i < 3 * 3 * 3 * nfilters; ++i) {
		weights[i] = 1;
	}
	
	float *bias = calloc(nfilters, sizeof(float));
	float *output = calloc(xreso * yreso * nfilters, sizeof(float));

	nnp_convolution_inference(
		nnp_convolution_algorithm_implicit_gemm,
		nnp_convolution_transform_strategy_tuple_based,
		3,
		nfilters,
		input_size,
		input_padding,
		kernel_size,
		stride,
		input,
		weights,
		bias,
		output,
		NULL,
		NULL,
		nnp_activation_identity,
		NULL,
		threadpool,
		NULL
	);

	FILE *fp = fopen("conv.txt", "w");
	for (int c = 0; c < 3; ++c) {
		float *ptr = input + c * xreso * yreso;
		fprintf(fp, "channel = %d\n", c);
		for (int y = 0; y < yreso; ++y) {
			for (int x = 0; x < xreso; ++x) {
				fprintf(fp, "%.5f ", ptr[y * xreso + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n\n", fp);
	}
	
	fputs("===============================================\n", fp);
	for (int c = 0; c < nfilters; ++c) {
		float *ptr = output + c * xreso * yreso;
		fprintf(fp, "channel = %d\n", c);
		for (int y = 0; y < yreso; ++y) {
			for (int x = 0; x < xreso; ++x) {
				fprintf(fp, "%.5f ", ptr[y * xreso + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n\n", fp);
	}
	
	fclose(fp);
	free(input);
	free(weights);
	free(bias);
	free(output);
	pthreadpool_destroy(threadpool);
	nnp_deinitialize();
}
#endif

void test_winograd_weight_transformation(int argc, char *argv[])
{
#ifdef OPENCL
	int save_result = 0;
	if (argc > 2) {
		save_result = atoi(argv[2]);
	}
	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
	
	const int filter_size = 3;
	const int filter_channels = 512;
	const int nfilters = 1024;
	float *weights = calloc(filter_size * filter_size * filter_channels * nfilters, sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < filter_size * filter_size * filter_channels * nfilters; ++i) {
		weights[i] = rand() / (double)RAND_MAX;
	}
	
	float *biases = calloc(nfilters, sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < nfilters; ++i) {
		biases[i] = rand() / (double)RAND_MAX;
	}
	
	weight_transform_context *context = create_weight_transform_context(F4x4_3x3, filter_channels, nfilters);
	
	int transformed_weight_image_width, transformed_weight_image_height;
	float *transformed_weights = NULL;
	if (save_result) {
		get_transformed_weight_image_size(context, &transformed_weight_image_width, &transformed_weight_image_height);
		transformed_weights = calloc((transformed_weight_image_width << 2) * transformed_weight_image_height, sizeof(float));
	}
	
	int N = 1;
	if (argc > 1) N = atoi(argv[1]);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		transform_weight(context, weights, biases, transformed_weights);
	}
	gettimeofday(&t2, NULL);
	printf("transform_weight: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	free_weight_transform_context(context);
	
	if (save_result) {
		save_volume(weights, 3, 3, filter_channels * nfilters, "weights.txt");
		save_volume(transformed_weights, transformed_weight_image_width << 2, transformed_weight_image_height, 1, "transformed_weights.txt");
		free(transformed_weights);
	}
	
	free(weights);
	free(biases);
	
	cl_destroy_wrapper(wrapper);
#endif
}

void test_winograd_input_transformation(int argc, char *argv[])
{
#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);

	const int input_width = 13;
	const int input_height = 13;
	const int input_channels = 512;
	const int stride = 1;
	const int padding = 1;
	
	float *input = calloc(input_width * input_height * input_channels, sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < input_width * input_height * input_channels; ++i) {
		input[i] = rand() / (double)RAND_MAX;
	}
	
	save_volume(input, input_width, input_height, input_channels, "input.txt");
	
	input_transform_context *context = create_input_transform_context(F4x4_3x3, input_width, input_height,
		input_channels, stride, padding);
	
	int input_image_width, input_image_height;
	get_input_image_size(context, &input_image_width, &input_image_height);
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc input_image_desc;
	memset(&input_image_desc, 0, sizeof(cl_image_desc));
	input_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	input_image_desc.image_width = input_image_width;
	input_image_desc.image_height = input_image_height;

	cl_mem d_input = clCreateImage(wrapper.context, mem_flags, &image_format, &input_image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		exit(-1);
	}

	size_t input_image_origin[] = {0, 0, 0};
	size_t input_image_region[] = {input_image_width, input_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_input = clEnqueueMapImage(wrapper.command_queue, d_input, CL_TRUE, CL_MAP_WRITE, input_image_origin,
		input_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	nchw_to_nhwc(input, h_input, input_width, input_height, input_channels, 1, input_width, image_row_pitch, 4);
	// save_volume(h_input, input_image_width << 2, input_image_height, 1, "formated_inputs.txt");
	cl_event event;
	clEnqueueUnmapMemObject(wrapper.command_queue, d_input, h_input, 0, NULL, &event);
	set_winograd_convolution_input(context, d_input);
	
	int transformed_input_width, transformed_input_height;
	get_transformed_input_image_size(context, &transformed_input_width, &transformed_input_height);
	float *transformed_input = calloc((transformed_input_width << 2) * transformed_input_height, sizeof(float));
	
	int N = 1;
	if (argc > 1) N = atoi(argv[1]);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		transform_input(context, transformed_input);
	}
	gettimeofday(&t2, NULL);
	printf("transform_input: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	save_volume(transformed_input, transformed_input_width << 2, transformed_input_height, 1, "transformed_input.txt");
	
	free_input_transform_context(context);
	free(input);
	free(transformed_input);

	clReleaseMemObject(d_input);
	cl_destroy_wrapper(wrapper);
#endif	
}

void test_winograd_convolution(int argc, char *argv[])
{
#ifdef OPENCL
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_WIDTH);
	cl_print_device_info(wrapper, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
	cl_print_device_info(wrapper, CL_DEVICE_EXTENSIONS);

	int filter_channels_test_group[] = {3,   3,  16,  32,  64, 128, 256,  512, 256, 384};
	int nfilters_test_group[] =        {4,  16,  32,  64, 128, 256, 512, 1024, 512, 256};
	int input_size_test_group[] =      {8, 416, 208, 104,  52,  26,  13,   13,  13,  26};
	
	int save_result = 0;
	if (argc > 1)
		save_result = atoi(argv[1]);
	
	float total_duration[3] = {0, 0, 0};
	for (int g = 0;  g < 1; ++g) {
		float *transformed_weights = NULL;
		float *transformed_input = NULL;
		float *output = NULL;
		float *inverse_transformed_output = NULL;
		
		const int filter_channels = filter_channels_test_group[g];
		const int nfilters = nfilters_test_group[g];
		
		float *weights = calloc(3 * 3 * filter_channels * nfilters, sizeof(float));
		srand(time(NULL));
		for (int i = 0; i < 3 * 3 * filter_channels * nfilters; ++i)
			weights[i] = rand() / (double)RAND_MAX;
		
		float *biases = calloc(nfilters, sizeof(float));
		srand(time(NULL));
		for (int i = 0; i < nfilters; ++i)
			biases[i] = rand() / (double)RAND_MAX;
		
		const int input_width = input_size_test_group[g];
		const int input_height = input_size_test_group[g];
		const int input_channels = filter_channels;
		const int stride = 1;
		const int padding = 1;
		
		float *input = calloc(input_width * input_height * input_channels, sizeof(float));
		srand(time(NULL));
		for (int i = 0; i < input_width * input_height * input_channels; ++i)
			input[i] = 0.1f * (rand() / (double)RAND_MAX - 0.5f);
				
		printf("\ntest configuration: input size %d, filter channels %d, number of filters %d\n", input_size_test_group[g],
			filter_channels_test_group[g], nfilters_test_group[g]);

		weight_transform_context *wtc = create_weight_transform_context(F4x4_3x3, filter_channels, nfilters);
		
		int transformed_weight_image_width, transformed_weight_image_height;
		if (save_result) {
			get_transformed_weight_image_size(wtc, &transformed_weight_image_width, &transformed_weight_image_height);
			transformed_weights = calloc((transformed_weight_image_width << 2) * transformed_weight_image_height, sizeof(float));
		}
		transform_weight(wtc, weights, biases, transformed_weights);
		if (save_result) {
			save_volume(weights, 3, 3, filter_channels * nfilters, "weights.txt");
			save_volume(biases, nfilters, 1, 1, "biases.txt");
			save_volume(transformed_weights, transformed_weight_image_width << 2, transformed_weight_image_height, 1, "transformed_weights.txt");
			free(transformed_weights);
		}

		input_transform_context *itc = create_input_transform_context(F4x4_3x3, input_width, input_height,
			input_channels, stride, padding);
			
		int input_image_width, input_image_height;
		get_input_image_size(itc, &input_image_width, &input_image_height);
		cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
		cl_image_format image_format = {
			.image_channel_order = CL_RGBA,
			.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
		};
		
		cl_image_desc input_image_desc;
		memset(&input_image_desc, 0, sizeof(cl_image_desc));
		input_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
		input_image_desc.image_width = input_image_width;
		input_image_desc.image_height = input_image_height;

		cl_int errcode;
		cl_mem d_input = clCreateImage(wrapper.context, mem_flags, &image_format, &input_image_desc, NULL, &errcode);
		if (CL_SUCCESS != errcode) {
			fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
			exit(-1);
		}

		size_t input_image_origin[] = {0, 0, 0};
		size_t input_image_region[] = {input_image_width, input_image_height, 1};
		size_t image_row_pitch, image_slice_pitch;
		MEM_MAP_PTR_TYPE *h_input = clEnqueueMapImage(wrapper.command_queue, d_input, CL_TRUE, CL_MAP_WRITE, input_image_origin,
			input_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
		image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
		nchw_to_nhwc(input, h_input, input_width, input_height, input_channels, 1, input_width, image_row_pitch, 4);
		if (save_result) {
			// save_volume(h_input, input_image_width << 2, input_image_height, 1, "formated_inputs.txt");
		}
		cl_event event;
		clEnqueueUnmapMemObject(wrapper.command_queue, d_input, h_input, 0, NULL, &event);
		set_winograd_convolution_input(itc, d_input);
		
		int transformed_input_width, transformed_input_height;		
		if (save_result) {
			get_transformed_input_image_size(itc, &transformed_input_width, &transformed_input_height);
			transformed_input = calloc((transformed_input_width << 2) * transformed_input_height, sizeof(float));
		}
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
		transform_input(itc, transformed_input);
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total_duration[0] += duration;
		printf("GPU & CPU, transform_input: %f ms, total %fms.\n", duration, total_duration[0]);
		if (save_result) {
			save_volume(input, input_width, input_height, input_channels, "inputs.txt");
			save_volume(transformed_input, transformed_input_width << 2, transformed_input_height, 1, "transformed_input.txt");
			free(transformed_input);
		}
		
		clReleaseMemObject(d_input);
		matrix_multiplication_context *mmc = create_matrix_multiplication_context(wtc, itc);
		
		int output_image_width, output_image_height;
		if (save_result) {
			get_transformed_output_image_size(mmc, &output_image_width, &output_image_height);
			output = calloc((output_image_width << 2) * output_image_height, sizeof(float));
		}
		gettimeofday(&t1, NULL);
		multiply_transformed_matrix(mmc, output);
		gettimeofday(&t2, NULL);
		duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total_duration[1] += duration;
		printf("GPU & CPU, multiply_transformed_matrix: %f ms, total %fms.\n", duration, total_duration[1]);
		if (save_result) {
			save_volume(output, output_image_width << 2, output_image_height, 1, "output.txt");
			free(output);
		}
	
		output_inverse_transform_context *oitc = create_output_inverse_transform_context(mmc, LEAKY);
		
		int inverse_transformed_output_image_width, inverse_transformed_output_image_height;
		if (save_result) {
			get_inverse_transformed_output_image_size(oitc, &inverse_transformed_output_image_width,
				&inverse_transformed_output_image_height);
			inverse_transformed_output = calloc((inverse_transformed_output_image_width << 2) *
				inverse_transformed_output_image_height, sizeof(float));
		}
		gettimeofday(&t1, NULL);
		inverse_transform_output(oitc, inverse_transformed_output);
		gettimeofday(&t2, NULL);
		duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total_duration[2] += duration;
		printf("GPU & CPU, inverse_transform_output: %f ms, total %fms.\n", duration, total_duration[2]);
		if (save_result) {
			save_volume(inverse_transformed_output, inverse_transformed_output_image_width << 2, inverse_transformed_output_image_height,
				1, "inverse_transformed_output.txt");
			free(inverse_transformed_output);
		}
		
		free_input_transform_context(itc);
		free_weight_transform_context(wtc);
		free_matrix_multiplication_context(mmc);
		free_output_inverse_transform_context(oitc);
		
		if (save_result) {
			dim3 input_size = {input_width, input_height, filter_channels};
			dim3 output_size;
			void *layers[] = {make_convolutional_layer(LEAKY, input_size, 3, nfilters, 1, 1, 1, 0, &output_size)};
			znet *net = znet_create(layers, 1, "coco.weights");
			convolutional_layer *layer = (convolutional_layer *)layers[0];
			memcpy(layer->weights, weights, 3 * 3 * filter_channels * nfilters * sizeof(float));
			for (int i = 0; i < nfilters; ++i) layer->biases[i] = biases[i];
			layer->input = input;
			forward_convolutional_layer(layer, net);
			save_volume(layer->output, output_size.w, output_size.h, output_size.c, "gemm_conv.txt");
			znet_destroy(net);
		}
		
		free(weights);
		free(biases);
		free(input);
	}

	cl_destroy_wrapper(wrapper);
#endif	
}

void test_normalize_image_with_gpu(int argc, char *argv[])
{
#if OPENCL
	char *program_buffer = NULL;
	cl_program program = 0;
	cl_kernel kernel = 0;
	bitmap *bmp = NULL;
	cl_mem image = 0;
	cl_mem normalized_image = 0;
	unsigned char *normalized_image_buffer = NULL;
	bitmap *normalized_bmp = NULL;

	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	size_t size = (size_t)(&BINARY_FILENAME_TO_END(utils, cl) - &BINARY_FILENAME_TO_START(utils, cl));
	program_buffer = calloc(size + 1, sizeof(char));
	if (!program_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	memcpy(program_buffer, &BINARY_FILENAME_TO_START(utils, cl), size);
	program_buffer[size] = '\0';
	
	char options[] = "";
	program = cl_make_wrapper_program(wrapper, "utils.cl", program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	kernel = cl_make_wrapper_kernel(wrapper, program, "normalize_image", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8
	};
	
	const int width = get_bmp_width(bmp);
	const int height = get_bmp_height(bmp);
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width;
	image_desc.image_height = height;
	
	image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	const int normalized_width = 416;
	const int normalized_height = 416;
	mem_flags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
	image_format.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = normalized_width;
	image_desc.image_height = normalized_height;
	
	normalized_image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	size_t origin[] = {0, 0, 0};
	size_t region[] = {width, height, 1};
	size_t row_pitch, slice_pitch;
	unsigned char *h_image = clEnqueueMapImage(wrapper.command_queue, image, CL_TRUE, CL_MAP_WRITE,
		origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &errcode);

	const unsigned char *data = get_bmp_data(bmp);
	const int src_row_pitch = get_bmp_pitch(bmp);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			h_image[(height - 1 - y) * row_pitch + (x << 2)]     = data[y * src_row_pitch + x * 3];
			h_image[(height - 1 - y) * row_pitch + (x << 2) + 1] = data[y * src_row_pitch + x * 3 + 1];
			h_image[(height - 1 - y) * row_pitch + (x << 2) + 2] = data[y * src_row_pitch + x * 3 + 2];
			h_image[(height - 1 - y) * row_pitch + (x << 2) + 3] = 0;
		}
	}
	cl_event event;
	clEnqueueUnmapMemObject(wrapper.command_queue, image, h_image, 0, NULL, &event);

	float scale;
	int resized_width, resized_height;
	if (normalized_width / (float)width < normalized_height / (float)height) {
		resized_width = normalized_width;
		resized_height = (int)(height * normalized_width / (float)width);
		scale = width / (float)normalized_width;
	} else {
		resized_width = (int)(width * normalized_height / (float)height);
		resized_height = normalized_height;
		scale = height / (float)normalized_height;
	}
	
	region[0] = normalized_width;
	region[1] = normalized_height;
	float fill_color[] = {0.5f, 0.5f, 0.5f, 0.5f};
	errcode = clEnqueueFillImage(wrapper.command_queue, normalized_image, fill_color, origin, region, 0, NULL, NULL);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clEnqueueFillImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
	errcode |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &normalized_image);
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &resized_width);
	errcode |= clSetKernelArg(kernel, 3, sizeof(int), &resized_height);
	errcode |= clSetKernelArg(kernel, 4, sizeof(float), &scale);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}

	cl_uint work_dim = 2;
	size_t global_work_size[] = {resized_width, resized_height};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	printf("GPU, normalize_image: %fms.\n", duration);
#endif
	clReleaseEvent(event);		
		
	normalized_image_buffer = calloc(normalized_width * normalized_height * 3, sizeof(unsigned char));
	if (!normalized_image_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}

	region[0] = normalized_width;
	region[1] = normalized_height;
	float *h_normalized_image = clEnqueueMapImage(wrapper.command_queue, normalized_image, CL_TRUE, CL_MAP_READ,
		origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &errcode);
	row_pitch = row_pitch >> 2;
	for (int y = 0; y < normalized_height; ++y) {
		for (int x = 0; x < normalized_width; ++x) {
			normalized_image_buffer[3 * (y * normalized_width + x)]     = (unsigned char)(h_normalized_image[y * row_pitch + (x << 2)] * 255);
			normalized_image_buffer[3 * (y * normalized_width + x) + 1] = (unsigned char)(h_normalized_image[y * row_pitch + (x << 2) + 1] * 255);
			normalized_image_buffer[3 * (y * normalized_width + x) + 2] = (unsigned char)(h_normalized_image[y * row_pitch + (x << 2) + 2] * 255);
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, normalized_image, h_normalized_image, 0, NULL, &event);
	
	normalized_bmp = create_bmp((const char *)normalized_image_buffer, normalized_width, normalized_height, 24);
	save_bmp(normalized_bmp, "gpu-normalized_image.bmp");
	
	cleanup:
	free(normalized_image_buffer);
	delete_bmp(bmp);
	delete_bmp(normalized_bmp);
	clReleaseMemObject(image);
	clReleaseMemObject(normalized_image);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	cl_destroy_wrapper(wrapper);
#endif	
}

void test_maxpool_layer_with_gpu(int argc, char *argv[])
{
#ifdef OPENCL
	float *input = NULL;
	cl_mem d_input = 0;
	maxpool_layer *layer = NULL;
	dim3 input_size = {416, 416, 16};
	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	input = calloc(input_size.w * input_size.h * input_size.c, sizeof(float));
	if (!input) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	srand(time(NULL));
	for (int i = 0; i < input_size.w * input_size.h * input_size.c; ++i) {
		input[i] = rand() / (double)RAND_MAX - 0.5;
	}
	
	save_volume(input, input_size.w, input_size.h, input_size.c, "input.txt");	
	const int channel_blocks = (input_size.c + 3) >> 2;
	const int input_image_width = input_size.w * channel_blocks;
	const int input_image_height = input_size.h;
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = input_image_width;
	image_desc.image_height = input_image_height;
	
	d_input = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	cl_event event;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {input_image_width, input_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_input = clEnqueueMapImage(wrapper.command_queue, d_input, CL_TRUE, CL_MAP_WRITE,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	nchw_to_nhwc(input, h_input, input_size.w, input_size.h, input_size.c, 1, input_size.w, image_row_pitch, 4);
	clEnqueueUnmapMemObject(wrapper.command_queue, d_input, h_input, 0, NULL, &event);

	dim3 output_size;
	layer = make_maxpool_layer(input_size, 2, 2, 1, 1, &output_size);
	set_maxpool_layer_input(layer, d_input);
	forward_maxpool_layer(layer, NULL);
	cl_mem d_output = get_maxpool_layer_output(layer);
	const int output_image_width = output_size.w * channel_blocks;
	const int output_image_height = output_size.h;
	
	region[0] = output_image_width;
	region[1] = output_image_height;
	MEM_MAP_PTR_TYPE *h_output = clEnqueueMapImage(wrapper.command_queue, d_output, CL_TRUE, CL_MAP_READ,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	// save_volume(h_output, output_size.w * output_size.c, output_size.h, 1, "maxpool.txt");
	clEnqueueUnmapMemObject(wrapper.command_queue, d_output, h_output, 0, NULL, &event);
	
	cleanup:
	clReleaseMemObject(d_input);
	free_maxpool_layer(layer);
	free(input);
	cl_destroy_wrapper(wrapper);
#endif
}

void test_direct_convolution(int argc, char *argv[])
{	
}

void test_resample_layer_with_gpu(int argc, char *argv[])
{
#ifdef OPENCL
	resample_layer *layer = NULL;
	float *input = NULL;
	cl_mem d_input = 0;
	cl_mem d_output = 0;
	int save_result = 0;
	
	if (argc > 1) save_result = atoi(argv[1]);

	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	dim3 input_size = {13, 13, 128};
	dim3 output_size;
	layer = make_resample_layer(input_size, 1, 2, &output_size);
	
	input = calloc(input_size.w * input_size.h * input_size.c, sizeof(float));
	if (!input) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	srand(time(NULL));
	for (int i = 0; i < input_size.w * input_size.h * input_size.c; ++i) {
		input[i] = rand() / (double)RAND_MAX;
	}
	
	if (save_result) save_volume(input, input_size.w, input_size.h, input_size.c, "input.txt");
	
	const int input_image_width = input_size.w * round_up_division_4(input_size.c);
	const int input_image_height = input_size.h;
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = input_image_width;
	image_desc.image_height = input_image_height;
	
	d_input = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	cl_event event;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {input_image_width, input_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_input = clEnqueueMapImage(wrapper.command_queue, d_input, CL_TRUE, CL_MAP_WRITE,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	nchw_to_nhwc(input, h_input, input_size.w, input_size.h, input_size.c, 1, input_size.w, image_row_pitch, 4);
	// if (save_result) save_volume(h_input, input_image_width << 2, input_image_height, 1, "formated_inputs.txt");
	clEnqueueUnmapMemObject(wrapper.command_queue, d_input, h_input, 0, NULL, &event);
	
	set_resample_layer_input(layer, d_input);
	forward_resample_layer(layer, NULL);
	d_output = get_resample_layer_output(layer);
	
	const int output_image_width = output_size.w * round_up_division_4(output_size.c);
	const int output_image_height = output_size.h;
	region[0] = output_image_width;
	region[1] = output_image_height;
	float *h_output = clEnqueueMapImage(wrapper.command_queue, d_output, CL_TRUE, CL_MAP_READ,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	if (save_result) save_volume(h_output, output_image_width << 2, output_image_height, 1, "output.txt");
	clEnqueueUnmapMemObject(wrapper.command_queue, d_output, h_output, 0, NULL, &event);	
	
	cleanup:
	if (input) free(input);
	free_resample_layer(layer);
	clReleaseMemObject(d_input);
	cl_destroy_wrapper(wrapper);
#endif
}

void test_nhwc_to_nchw(int argc, char *argv[])
{
#if defined(OPENCL) && defined(USE_FLOAT)
	const int w = 13;
	const int h = 13;
	const int c = 18;
	const int rounded_c = ((c + 3) >> 2) << 2;
	MEM_MAP_PTR_TYPE *input = calloc(w * h * rounded_c, sizeof(MEM_MAP_PTR_TYPE));
	float *output = calloc(w * h * c, sizeof(float));
	
	srand(time(NULL));
	for (int i = 0; i < w * h * rounded_c; ++i) {
		input[i] = rand() / (double)RAND_MAX;
	}
	
	save_volume(input, w * rounded_c, h, 1, "nhwc.txt");
	nhwc_to_nchw(input, output, w, h, c, 1, w * rounded_c, w, 4);
	save_volume(output, w, h, c, "nchw.txt");
	
	free(input);
	free(output);
#endif
}

void test_image_standardizer(int argc, char *argv[])
{
#ifdef OPENCL	
	bitmap *bmp = NULL;
	bitmap *standard_bmp = NULL;
	image_standardizer *standardizer = NULL;
	unsigned char *standard_image_buffer = NULL;
	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	const int width = get_bmp_width(bmp);
	const int height = get_bmp_height(bmp);
	const unsigned char *data = get_bmp_data(bmp);
	const int standard_width = 416;
	const int standard_height = 416;
	
	standardizer = create_image_standardizer(width, height, standard_width, standard_height, 3);
	if (!standardizer) goto cleanup;
	
	int N = 1;
	if (argc > 2) N = atoi(argv[2]);
	
	cl_mem d_standard_image = get_standardizer_output_ptr(standardizer);
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
		standardize_image(standardizer, data, width, height, 0, 0, width, height, d_standard_image);
	}
	gettimeofday(&t2, NULL);
	printf("time: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	standard_image_buffer = calloc(standard_width * standard_height * 3, sizeof(unsigned char));
	if (!standard_image_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}

	cl_event event;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {standard_width, standard_height, 1};
	size_t row_pitch, slice_pitch;
	MEM_MAP_PTR_TYPE *h_standard_image = clEnqueueMapImage(wrapper.command_queue, d_standard_image, CL_TRUE, CL_MAP_READ,
		origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &errcode);
	row_pitch = row_pitch / sizeof(MEM_MAP_PTR_TYPE);
#ifndef CHANNEL_BLOCK_SIZE8
	printf("channel block size 4!\n");
	const int shift = 2;
#else
	printf("channel block size 8!\n");
	const int shift = 3;
#endif
	for (int y = 0; y < standard_height; ++y) {
		for (int x = 0; x < standard_width; ++x) {
			float red = DEVICE_TO_HOST(h_standard_image[y * row_pitch + (x << shift)]);
			float green = DEVICE_TO_HOST(h_standard_image[y * row_pitch + (x << shift) + 1]);
			float blue = DEVICE_TO_HOST(h_standard_image[y * row_pitch + (x << shift) + 2]);
			standard_image_buffer[3 * (y * standard_width + x)]     = (unsigned char)(red * 255);
			standard_image_buffer[3 * (y * standard_width + x) + 1] = (unsigned char)(green * 255);
			standard_image_buffer[3 * (y * standard_width + x) + 2] = (unsigned char)(blue * 255);
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_standard_image, h_standard_image, 0, NULL, &event);
	
	standard_bmp = create_bmp((const char *)standard_image_buffer, standard_width, standard_height, 24);
	save_bmp(standard_bmp, "gpu-standard_image.bmp");
	
	cleanup:
	delete_bmp(bmp);
	delete_bmp(standard_bmp);
	if (standard_image_buffer) free(standard_image_buffer);
	free_image_standardizer(standardizer);
	cl_destroy_wrapper(wrapper);
#endif
}

void test_yolov3_tiny_with_cpu_or_gpu(int argc, char *argv[])
{
	bitmap *bmp = NULL;
	image_standardizer *standardizer = NULL;
	znet *net = NULL;
#ifdef OPENCL	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
#endif	
	bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	const int width = get_bmp_width(bmp);
	const int height = get_bmp_height(bmp);
	const unsigned char *data = get_bmp_data(bmp);
	const int standard_width = 416;
	const int standard_height = 416;
	const int roiw = 1000;
	const int roih = 1000;
	const int roix = (width - roiw) >> 1;
	const int roiy = (height - roih) >> 1;
	
	standardizer = create_image_standardizer(width, height, standard_width, standard_height, 3);
	if (!standardizer) goto cleanup;
	
	void *standard_image = get_standardizer_output_ptr(standardizer);
	standardize_image(standardizer, data, width, height, roix, roiy, roiw, roih, standard_image);

	void *layers[24];
	int nlayers = sizeof(layers) / sizeof(layers[0]);
	dim3 output_size;
	
	int bigger_mask[] = {3, 4, 5};
	int smaller_mask[] = {0, 1, 2};
	int anchor_boxes[] = {61,117, 62,191, 199,118, 128,195, 92,293, 191,291};
	const int scales = 3;
	const int classes = 1;
	const int object_tensor_depth = (4 + 1 + classes) * scales;
	
	dim3 input_size = {416, 416, 3};
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

	net = znet_create(layers, nlayers, "agriculture.weights");
	if (!net) goto cleanup;

	system("rm -f *.cl.bin");
	znet_architecture(net);
	
	int N = 1;
	if (argc > 3) N = atoi(argv[3]);
	printf("inference iterations %d\n", N);
	
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int i = 0; i < N; ++i) {
#ifdef NDEBUG
		printf("------------------------------------------------------\n");
#endif
		znet_inference(net, standard_image);
	}
	gettimeofday(&t2, NULL);
	printf("time: %f ms.\n", ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0);
	
	float thresh = 0.5f;
	if (argc > 2) thresh = atof(argv[2]);
	
	list *detections = get_detections(net, thresh, roiw, roih);	
	list *bests = soft_nms(detections, 2.5);

	draw_detections(bmp, bests, names, thresh, roix, roiy, roiw, roih);
	save_bmp(bmp, "detections.bmp");
	
	free_detections(bests);
	free_detections(detections);
	
	cleanup:
	znet_destroy(net);
	delete_bmp(bmp);
	free_image_standardizer(standardizer);
#ifdef OPENCL
	cl_destroy_wrapper(wrapper);
#endif	
}

void test_ion_image_standardizer(int argc, char *argv[])
{
	bitmap *bmp = NULL;
	image_standardizer *standardizer = NULL;
	unsigned char *red = NULL;
	bitmap *red_bmp = NULL;

	bmp = read_bmp(argv[1]);
	if (!bmp) {
		fprintf(stderr, "read_bmp fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}

	const int width = get_bmp_width(bmp);
	const int height = get_bmp_height(bmp);
	const int standard_width = 416;
	const int standard_height = 416;

	standardizer = create_image_standardizer(width, height, standard_width, standard_height, 3);
	if (!standardizer) goto cleanup;

	unsigned char *data = get_bmp_data(bmp);
	float *output = calloc(standard_width * standard_height * 3, sizeof(float));
	standardize_image(standardizer, data, width, height, 0, 0, width, height, output);

	red = calloc(standard_width * standard_height, sizeof(unsigned char));
	
	for (int i = 0; i < standard_width * standard_height; ++i)
		red[i] = (unsigned char)(output[i] * 255);

	free(output);
	red_bmp = create_bmp((const char *)red, standard_width, standard_height, 8);
	save_bmp(red_bmp, "standard.bmp");

	cleanup:
	delete_bmp(red_bmp);
	free(red);
	delete_bmp(bmp);
	free_image_standardizer(standardizer);
}

void test_half(int argc, char *argv[])
{
#ifdef OPENCL
	enum {N = 12168};
	float float_number[N];
	srand(time(NULL));
	for (int i = 0; i < N; ++i) {
		float_number[i] = rand() / (double)RAND_MAX - 0.5;
	}
	
	float_number[0] = 0;
	float_number[1] = HUGE_VALF;
	float_number[2] = -HUGE_VALF;
	float_number[3] = NAN;
	
	int iters = 1;
	if (argc > 1) iters = atoi(argv[1]);
	
	cl_half half_number[N];
	struct timeval t1, t2; 
    gettimeofday(&t1, NULL);
	for (int k = 0; k < iters; ++k) {
		for (int i = 0; i < N; ++i) {
			half_number[i] = to_half(float_number[i]);
		}
	}
	gettimeofday(&t2, NULL);
	double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("to_half time: %f ms.\n", duration / iters);
	
	cl_float float_number_new[N];
	gettimeofday(&t1, NULL);
	for (int k = 0; k < iters; ++k) {
		for (int i = 0; i < N; ++i) {
			float_number_new[i] = to_float(half_number[i]);
		}
	}
	gettimeofday(&t2, NULL);
	duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
	printf("to_float time: %f ms.\n", duration / iters);
	
	printf("float: ");
	for (int i = 0; i < 16; ++i) {
		printf("%.7f ", float_number[i]);
	}
	
	printf("\nhalf: ");
	for (int i = 0; i < 16; ++i) {
		printf("%.7f ", float_number_new[i]);
	}
	printf("\n");
#endif	
}

void test_read_half_value_from_float_image(int argc, char *argv[])
{
#ifdef OPENCL
	const int width = 16;
	const int height = 16;
	cl_float float_data[width * height];
	cl_half half_data[width * height];
	cl_mem d_image = 0;
	cl_program program = 0;
	cl_kernel kernel = 0;
	cl_mem d_half_image = 0;
	
	struct stat statbuf;
	stat("test.cl", &statbuf);
	char program_buffer[statbuf.st_size + 1];
	
	cl_int errcode;
	wrapper = cl_create_wrapper(&errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_create_wrapper[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	srand(time(NULL));
	for (int i = 0; i < width * height; ++i) {
		float_data[i] = rand() / (double)RAND_MAX - 0.5;
	}
	
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			printf("%f ", float_data[y * width + x]);
		}
		printf("\n");
	}
	printf("\n");
	
	for (int i = 0; i < width * height; ++i) {
		half_data[i] = to_half(float_data[i]);
	}
	
	char *half_ptr = (char *)half_data;
	const int half_row_pitch = width * sizeof(cl_half);
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT
	};

	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width >> 3;
	image_desc.image_height = height;
	
	d_image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	size_t origin[] = {0, 0, 0};
	size_t region[] = {width >> 3, height, 1};
	size_t image_row_pitch, image_slice_pitch;
	char *h_image = (char *)clEnqueueMapImage(wrapper.command_queue, d_image, CL_TRUE, CL_MAP_WRITE, origin, region,
		&image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	for (int y = 0; y < height; ++y) {
		memcpy(h_image + y * image_row_pitch, half_ptr + y * half_row_pitch, half_row_pitch);
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_image, h_image, 0, NULL, NULL);
	
	mem_flags = CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
	image_format.image_channel_order = CL_RGBA;
	image_format.image_channel_data_type = CL_HALF_FLOAT;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width >> 2;
	image_desc.image_height = height;
	d_half_image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	FILE *fp = fopen("test.cl", "rb");
	fread(program_buffer, sizeof(char), statbuf.st_size, fp);
	fclose(fp);
	program_buffer[statbuf.st_size] = '\0';
	
	char options[] = "";
	program = cl_make_wrapper_program(wrapper, "test.cl", program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	kernel = cl_make_wrapper_kernel(wrapper, program, "test", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	errcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_image);
	errcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_half_image);
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {width >> 3, height};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
	clFinish(wrapper.command_queue);
	
	region[1] = width >> 2;
	cl_half *h_half_image = clEnqueueMapImage(wrapper.command_queue, d_half_image, CL_TRUE, CL_MAP_READ, origin, region,
		&image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(cl_half);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			printf("%f ", to_float(h_half_image[y * image_row_pitch + x]));
		}
		printf("\n");
	}
	printf("\n");
	clEnqueueUnmapMemObject(wrapper.command_queue, d_image, h_image, 0, NULL, NULL);
	
	cleanup:
	clReleaseMemObject(d_image);
	clReleaseMemObject(d_half_image);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	cl_destroy_wrapper(wrapper);
#endif	
}

void test_set_color(int argc, char *argv[])
{
	enum {nchannels = 4};
	float color[nchannels];
	// set_fill_color(&color, nchannels);
#ifndef CHANNEL_BLOCK_SIZE8
	for (int i = 0; i < nchannels; ++i) printf("%f ", color[i]);
	printf("\n");
#else
	const int half_channels = nchannels << 1;
	cl_half *ptr = (cl_half *)color;
	for (int i = 0; i < half_channels; ++i) printf("%f ", to_float(ptr[i]));
	printf("\n");
#endif
}