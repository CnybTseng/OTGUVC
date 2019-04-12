#pragma once
extern "C"
{
	#include <libavutil/opt.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/channel_layout.h>
	#include <libavutil/common.h>
	#include <libavutil/imgutils.h>
	#include <libavutil/mathematics.h>
	#include <libavutil/samplefmt.h>
	#include <libavformat/avformat.h>
}

#define LINE_SZ 1024

void ffmpeg_av_log_default_callback(void* ptr, int level, const char* fmt, va_list vl);
static void format_line(void *ptr, int level, const char *fmt, va_list vl, char part[3][LINE_SZ], int part_size, int *print_prefix, int type[2]);
static void sanitize(char *line);
static int get_category(void *ptr);
static void colored_fputs(int level, const char *str);