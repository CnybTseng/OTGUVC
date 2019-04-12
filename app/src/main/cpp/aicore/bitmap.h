#ifndef _BITMAP_H_
#define _BITMAP_H_

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef AICORE_BUILD_DLL
#ifdef _WIN32
#	define BITMAP_EXPORT __declspec(dllexport)
#else
#	define BITMAP_EXPORT __attribute__ ((visibility("default"))) extern
#endif
#else
#ifdef _WIN32
#	define BITMAP_EXPORT __declspec(dllimport)
#else
#	define BITMAP_EXPORT __attribute__ ((visibility("default")))
#endif
#endif

struct bitmap;
typedef struct bitmap bitmap;

BITMAP_EXPORT bitmap *read_bmp(const char *filename);
BITMAP_EXPORT bitmap *create_bmp(const char *const data, int width, int height, int bits_per_pixel);
BITMAP_EXPORT unsigned char *get_bmp_data(bitmap *bmp);
BITMAP_EXPORT int get_bmp_width(bitmap *bmp);
BITMAP_EXPORT int get_bmp_height(bitmap *bmp);
BITMAP_EXPORT int get_bmp_bit_count(bitmap *bmp);
BITMAP_EXPORT int get_bmp_pitch(bitmap *bmp);
BITMAP_EXPORT void save_bmp(bitmap *bmp, const char *filename);
BITMAP_EXPORT void delete_bmp(bitmap *bmp);

#ifdef __cplusplus
}
#endif

#endif