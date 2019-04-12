#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "znet.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "route_layer.h"
#include "resample_layer.h"
#include "yolo_layer.h"

extern char BINARY_FILENAME_TO_START(agriculture, weights);

typedef void (*print_layer_info_t)(void *layer, int id);
typedef void (*set_layer_input_t)(void *layer, void *input);
typedef void *(*get_layer_output_t)(void *layer);
typedef void (*forward_t)(void *layer, znet *net);
typedef void (*free_layer_t)(void *layer);

struct znet {
	WORK_MODE work_mode;
	int nlayers;
	void **layers;
	void *input;
	float *output;
	int width;
	int height;
	print_layer_info_t *print_layer_info;
	set_layer_input_t *set_layer_input;
	get_layer_output_t *get_layer_output;
	forward_t *forward;
	free_layer_t *free_layer;
	int *is_output_layer;
	char weight_filename[256];
#ifdef NNPACK
	pthreadpool_t threadpool;
#endif
};

static int convnet_parse_input_size(znet *net);
static int convnet_parse_layer(znet *net);
static int __attribute__((unused)) convnet_parse_weights(znet *net);
static int convnet_parse_weights_from_buffer(znet *net);

znet *znet_create(void *layers[], int nlayers, const char *weight_filename)
{
	znet *net = calloc(1, sizeof(znet));
	if (!net) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return net;
	}
	
	net->work_mode = INFERENCE;
	net->nlayers = nlayers;
	net->layers = layers;
	net->input = NULL;
	net->output = NULL;
	net->print_layer_info = NULL;
	net->set_layer_input = NULL;
	net->get_layer_output = NULL;
	net->forward = NULL;
	net->free_layer = NULL;
	net->is_output_layer = NULL;
	strcpy(net->weight_filename, weight_filename);
	
	net->print_layer_info = calloc(nlayers, sizeof(print_layer_info_t));
	if (!net->print_layer_info) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->set_layer_input = calloc(nlayers, sizeof(set_layer_input_t));
	if (!net->set_layer_input) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->get_layer_output = calloc(nlayers, sizeof(get_layer_output_t));
	if (!net->get_layer_output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->forward = calloc(nlayers, sizeof(forward_t));
	if (!net->forward) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->free_layer = calloc(nlayers, sizeof(free_layer_t));
	if (!net->free_layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	net->is_output_layer = calloc(nlayers, sizeof(int));
	if (!net->is_output_layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	if (convnet_parse_input_size(net)) goto cleanup;
	if (convnet_parse_layer(net)) goto cleanup;
	
	if (convnet_parse_weights_from_buffer(net)) {
		cleanup:znet_destroy(net);
		return NULL;
	}
	
#ifdef NNPACK
	net->threadpool = pthreadpool_create(8);
	nnp_initialize();
#endif	
	
	return net;
}

void znet_train(znet *net, data_store *ds, train_options *opts)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

float *znet_inference(znet *net, void *input)
{
	net->work_mode = INFERENCE;
	net->input = input;
	
	for (int i = 0; i < net->nlayers; ++i) {
		net->set_layer_input[i](net->layers[i], net->input);
		net->forward[i](net->layers[i], net);
		net->input = net->get_layer_output[i](net->layers[i]);
	}

	return 0;
}

void znet_destroy(znet *net)
{
	if (!net) return;
	
	for (int i = 0; i < net->nlayers; ++i) {		
		net->free_layer[i](net->layers[i]);
	}
	
	if (net->print_layer_info) {
		free(net->print_layer_info);
		net->print_layer_info = NULL;
	}
	
	if (net->set_layer_input) {
		free(net->set_layer_input);
		net->set_layer_input = NULL;
	}
	
	if (net->get_layer_output) {
		free(net->get_layer_output);
		net->get_layer_output = NULL;
	}
	
	if (net->forward) {
		free(net->forward);
		net->forward = NULL;
	}
	
	if (net->free_layer) {
		free(net->free_layer);
		net->free_layer = NULL;
	}
	
	if (net->is_output_layer) {
		free(net->is_output_layer);
		net->is_output_layer = NULL;
	}
	
#ifdef NNPACK
	pthreadpool_destroy(net->threadpool);
	nnp_deinitialize();
#endif

	free(net);
	net = NULL;
}

void znet_architecture(znet *net)
{
	printf("id\tlayer\t\t\t   input\t  size/stride\t     filters\t\t\t  output\n");
	for (int i = 0; i < net->nlayers; i++) {
		net->print_layer_info[i](net->layers[i], i + 1);
	}
}

WORK_MODE znet_workmode(znet *net)
{
	return net->work_mode;
}

void **znet_layers(znet *net)
{
	return net->layers;
}

#ifdef NNPACK
pthreadpool_t znet_threadpool(znet *net)
{
	return net->threadpool;
}
#endif

int znet_input_width(znet *net)
{
	return net->width;
}

int znet_input_height(znet *net)
{
	return net->height;
}

list *get_detections(znet *net, float thresh, int width, int height)
{
	list *l = make_list();
	if (!l) return l;
	
	for (int i = 0; i < net->nlayers; ++i) {
		if (!net->is_output_layer[i]) continue;
		get_yolo_layer_detections((yolo_layer *)net->layers[i], net, width, height, thresh, l);
	}
	
	return l;
}

void free_detections(list *l)
{
	free_yolo_layer_detections(l);
}

int convnet_parse_input_size(znet *net)
{
	LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[0]);
	if (type != CONVOLUTIONAL) {
		fprintf(stderr, "the first layer isn't convolutional layer!");
		return -1;
	}
	
	convolutional_layer *layer = (convolutional_layer *)net->layers[0];
	net->width = layer->input_size.w;
	net->height = layer->input_size.h;
	
	return 0;
}

int convnet_parse_layer(znet *net)
{
	for (int i = 0; i < net->nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			net->print_layer_info[i] = print_convolutional_layer_info;
			net->set_layer_input[i] = set_convolutional_layer_input;
			net->get_layer_output[i] = get_convolutional_layer_output;
			net->forward[i] = forward_convolutional_layer;
			net->free_layer[i] = free_convolution_layer;
		} else if (type == MAXPOOL) {
			net->print_layer_info[i] = print_maxpool_layer_info;
			net->set_layer_input[i] = set_maxpool_layer_input;
			net->get_layer_output[i] = get_maxpool_layer_output;
			net->forward[i] = forward_maxpool_layer;
			net->free_layer[i] = free_maxpool_layer;
		} else if (type == ROUTE) {
			net->print_layer_info[i] = print_route_layer_info;
			net->set_layer_input[i] = set_route_layer_input;
			net->get_layer_output[i] = get_route_layer_output;
			net->forward[i] = forward_route_layer;
			net->free_layer[i] = free_route_layer;
		} else if (type == RESAMPLE) {
			net->print_layer_info[i] = print_resample_layer_info;
			net->set_layer_input[i] = set_resample_layer_input;
			net->get_layer_output[i] = get_resample_layer_output;
			net->forward[i] = forward_resample_layer;
			net->free_layer[i] = free_resample_layer;
		} else if (type == YOLO) {
			net->print_layer_info[i] = print_yolo_layer_info;
			net->set_layer_input[i] = set_yolo_layer_input;
			net->get_layer_output[i] = get_yolo_layer_output;
			net->forward[i] = forward_yolo_layer;
			net->free_layer[i] = free_yolo_layer;
			net->is_output_layer[i] = 1;
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
			return -1;
		}
	}
	
	return 0;
}

int convnet_parse_weights(znet *net)
{
	FILE *fp = fopen(net->weight_filename, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	int major;
	int minor;
	int revision;
	unsigned long long seen;
	
	fread(&major, sizeof(int), 1, fp);
	fread(&minor, sizeof(int), 1, fp);
	fread(&revision, sizeof(int), 1, fp);
	if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
		fread(&seen, sizeof(unsigned long long), 1, fp);
	} else {
		int iseen = 0;
		fread(&iseen, sizeof(int), 1, fp);
		seen = iseen;
	}
	
	printf("version %d.%d.%d, seen %u.\n", major, minor, revision, (unsigned int)seen);
	for (int i = 0; i < net->nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			load_convolutional_layer_weights(layer, fp);
		}
	}
	
	fclose(fp);
	
	return 0;
}

int convnet_parse_weights_from_buffer(znet *net)
{	
	int major;
	int minor;
	int __attribute__((unused)) revision;
	unsigned long long __attribute__((unused)) seen;
	
	char *ptr = &BINARY_FILENAME_TO_START(agriculture, weights);
	
	major = *((int *)ptr);
	ptr += sizeof(int);
	minor = *((int *)ptr);
	ptr += sizeof(int);
	revision = *((int *)ptr);
	ptr += sizeof(int);
	
	if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
		seen = *((unsigned long long *)ptr);
		ptr += sizeof(unsigned long long);
	} else {
		seen = *((int *)ptr);
		ptr += sizeof(int);
	}

	for (int i = 0; i < net->nlayers; ++i) {
		LAYER_TYPE type = *(LAYER_TYPE *)(net->layers[i]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *layer = (convolutional_layer*)net->layers[i];
			load_convolutional_layer_weights_from_buffer(layer, &ptr);
		}
	}
	
	return 0;
}