#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <string.h>
#ifdef __linux__
#	include <termios.h>
#	include <sys/types.h>
#	include <unistd.h>
#	include <fcntl.h>
#endif
#ifdef _WIN32
#	include <windows.h>
#	include <conio.h>
#endif
#include "aicore.h"
#include "bitmap.h"
#if defined(__linux__) && defined(ION)
#	include "cl_wrapper.h"
#endif

static int quit = 0;	// 退出应用程序的标志

void print_help();
int create_object_process_thread(pthread_t *tid);
void *object_process_thread(void *param);
void draw_bounding_box(bitmap *bmp, int x, int y, int w, int h);
void wait_for_thread_dead(pthread_t tid);
#ifdef __linux__
int kbhit(void);
#ifdef ION
extern cl_wrapper wrapper;
cl_ion_context create_ion_image(int width, int height);
void set_ion_image_value(cl_ion_context ion_context, bitmap *bmp);
#endif
#endif

int main(int argc, char *argv[])
{
	if (argc < 3){
		print_help();
		return 0;
	}

	pthread_t tid;
	int keyboard = 0;
	char filename[128];
	int counter = 0;
	
	const int w = atoi(argv[1]);
	const int h = atoi(argv[2]);
	int ret = ai_core_init(w, h);
	if (ret != AIC_OK) {
		fprintf(stderr, "ai_core_init fail, error code %d.\n", ret);
		goto cleanup;
	}
#if defined(__linux__) && defined(ION)
	cl_ion_context ion_context = create_ion_image(w, h);
#endif
	if (create_object_process_thread(&tid)) goto cleanup;

	int delay = 30000000;	// 30ms
	if (argc > 2) delay = atoi(argv[3]);
	const struct timespec req = {0, delay};
	double duration = 0;
	struct timeval t1, t2;
	
	while (!quit) {
		// 检测是否按下'q'键想退出测试程序
#ifdef __linux__
		if (kbhit()) {
			keyboard = getchar();
#else
		if (_kbhit()) {
			keyboard = _getch();
#endif
			if (keyboard == 'q') {
				quit = 1;
				break;
			}
		}
		
		sprintf(filename, "dataset/%04d.bmp", counter);
		bitmap *bmp = read_bmp(filename);
		if (!bmp) {
			fprintf(stderr, "read_bmp fail[%s:%d:%s], restart.\n", __FILE__, __LINE__, filename);
			counter = 0;
			continue;
		}
		
		const int width = get_bmp_width(bmp);
		const int height = get_bmp_height(bmp);
#if !defined(__linux__) || !defined(ION)
		const char *data = (char *)get_bmp_data(bmp);
#endif
		
#if !defined(__linux__) || !defined(ION)
		int ret = ai_core_send_image(data, width * height);
#else
		set_ion_image_value(ion_context, bmp);
		int ret = ai_core_send_ion_image(ion_context.ion_mem.ion_filedesc, ion_context.ion_mem.ion_hostptr, width, height);
#endif			
		if (ret != AIC_OK) {
			if (ret == AIC_FRAME_DISCARD) fprintf(stderr, "ai_core_send_image: discard!\n");	// 抽帧检测
			else fprintf(stderr, "ai_core_send_image fail! error code %d.\n", ret);
		}
		
		if (0 == counter) {
			gettimeofday(&t1, NULL);
		} else {
			gettimeofday(&t2, NULL);
			duration = ((double)t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
			printf("send frame rate %ffps.\n", counter / duration);
		}
		
		++counter;
		if (bmp) delete_bmp(bmp);
		nanosleep(&req, NULL);
	}
	
	cleanup:
	wait_for_thread_dead(tid);	// 等待接收物体坐标的线程结束
	ai_core_free();
#if defined(__linux__) && defined(ION)
	cl_free_ion_context(wrapper, ion_context);
#endif
	return 0;
}

void print_help()
{
	printf("Usage:\n");
	printf("      ./test_aicore [width] [height] [delay(nanosecond)]\n");
}

int create_object_process_thread(pthread_t *tid)
{
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	
	int ret = pthread_create(tid, &attr, object_process_thread, NULL);
	if (0 != ret) {
		fprintf(stderr, "pthread_create[%s:%d].\n", __FILE__, __LINE__);
		return -1;
	}
	
	return 0;
}

void *object_process_thread(void *param)
{
	float threshold = 0.25f;
	const struct timespec req = {0, 1000000};
	object_t object;
	int counter = 0;
	char filename[128];
	double duration = 0;
	struct timeval t1, t2;
	double frame_rate = 0;
	
	while (!quit) {
		int ret = ai_core_fetch_object(&object, 1, threshold);
		if (ret < 0) {
			continue;
		}
		
		if (0 == counter) {
			gettimeofday(&t1, NULL);
		} else {
			gettimeofday(&t2, NULL);
			duration = ((double)t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0;
			frame_rate = counter / duration;
		}
		
		sprintf(filename, "dataset/%04d.bmp", counter);
		bitmap *bmp = read_bmp(filename);
		if (!bmp) {
			fprintf(stderr, "read_bmp fail[%s:%d:%s], restart.\n", __FILE__, __LINE__, filename);
			counter = 0;
			continue;
		}

		if (ret > 0) {
			printf("detected object[%d,%d,%d,%d;%f,%f],", object.x, object.y, object.w, object.h,
				object.objectness, object.probability);
			draw_bounding_box(bmp, object.x, object.y, object.w, object.h);
		} else {
			printf("no object,");
		}
		
		printf("display frame rate %ffps.\n", frame_rate);
		sprintf(filename, "detection/%04d.bmp", counter);
		save_bmp(bmp, filename);
		++counter;
		if (bmp) delete_bmp(bmp);
		nanosleep(&req, NULL);
	}

	return (void *)(0);
}

void draw_bounding_box(bitmap *bmp, int x, int y, int w, int h)
{
	const int height = get_bmp_height(bmp);
	const int pitch = get_bmp_pitch(bmp);
	const int bit_count = get_bmp_bit_count(bmp);
	unsigned char *data = get_bmp_data(bmp);
	const int bpp = bit_count >> 3;
	const int color[3] = {0, 255, 255};
	const int left = x;
	const int right = x + w - 1;
	const int bottom = height - 1 - y;
	const int top = height - 1 - (y + h - 1);
	const int lw = 3;
	
	for (int c = 0; c < bpp; ++c) {
		for (int y = top; y < bottom; ++y) {
			for (int l = 0; l < lw; ++l) {
				data[y * pitch + (left + l) * bpp + c] = color[c];
				data[y * pitch + (right - l) * bpp + c] = color[c];
			}
		}
		
		for (int x = left; x < right; ++x) {
			for (int l = 0; l < lw; ++l) {
				data[(top + l) * pitch + x * bpp + c] = color[c];
				data[(bottom - l) * pitch + x * bpp + c] = color[c];
			}
		}
	}
}

void wait_for_thread_dead(pthread_t tid)
{
	int timer = 1000;
	struct timespec req = {0, 10000000};
	while (timer--) {
		int ret = pthread_kill(tid, 0);
		if (ESRCH == ret) {
			fprintf(stderr, "the thread didn't exists or already quit[%s:%d].\n", __FILE__, __LINE__);
			return;
		} else if (EINVAL == ret) {
			fprintf(stderr, "signal is invalid[%s:%d].\n", __FILE__, __LINE__);
			return;
		} else {
			nanosleep(&req, NULL);
			continue;
		}
	}
}

#ifdef __linux__
int kbhit(void)
{
	struct termios oldt, newt;
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	
	int oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
	
	int ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF) {
		ungetc(ch, stdin);
		return 1;
	}
	
	return 0;
}

#ifdef ION
cl_ion_context create_ion_image(int width, int height)
{	
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
	
	cl_ion_context ion_context = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_desc);
	return ion_context;
}

void set_ion_image_value(cl_ion_context ion_context, bitmap *bmp)
{
	const int width = get_bmp_width(bmp);
	const int height = get_bmp_height(bmp);
	const int pitch = get_bmp_pitch(bmp);
	const unsigned char *data = get_bmp_data(bmp);
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
	cl_mem d_image = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, &ion_context.ion_mem, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		exit(-1);
	}
	
	size_t origin[] = {0, 0, 0};
	size_t region[] = {width, height, 1};
	size_t row_pitch, slice_pitch;
	unsigned char *h_image = clEnqueueMapImage(wrapper.command_queue, d_image, CL_TRUE, CL_MAP_WRITE,
		origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &errcode);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			h_image[(height - 1 - y) * row_pitch + (x << 2)]     = data[y * pitch + x * 3 + 2];
			h_image[(height - 1 - y) * row_pitch + (x << 2) + 1] = data[y * pitch + x * 3 + 1];
			h_image[(height - 1 - y) * row_pitch + (x << 2) + 2] = data[y * pitch + x * 3 + 0];
			h_image[(height - 1 - y) * row_pitch + (x << 2) + 3] = 0;
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, d_image, h_image, 0, NULL, NULL);
	clReleaseMemObject(d_image);
}
#endif
#endif