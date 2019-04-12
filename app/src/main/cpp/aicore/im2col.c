#include "im2col.h"

/** @brief 图像转换为矩阵.按照从左到右,从上到下的顺序移动卷积核,将卷积核覆盖的二维像素矩阵降维成列向量,
 **        并按照从左到右的顺序填充到matrix的一列.
 ** @param image 图像数据.对于多通道图像,同一通道内的数据是连续存储的.
 ** @param width 图像宽度.
 ** @param height 图像高度.
 ** @param channels 图像通道数.
 ** @param fsize 卷积核大小,目前只支持方形卷积核.
 ** @param stride 卷积核移动步长,目前只支持水平和垂直方向等步长移动卷积.
 ** @param padding 填充量,目前只支持水平和垂直方向等量填充.
 ** @param matrix 矩阵数据.该矩阵的行数等于图像通道数*卷积核高*卷积核宽,矩阵的列数等于卷积后
 **        图像高*卷积后图像宽.
 **/
void im2col_cpu(float *image, int width, int height, int channels, int fsize,
                int stride, int padding, float *matrix)
{
	int convw = (width  + 2 * padding - fsize) / stride + 1;	// 卷积后图像宽
	int convh = (height + 2 * padding - fsize) / stride + 1;	// 卷积后图像高
	int channel_size = width * height;		// 每个通道的图像大小
	int submatrix_width = convw * convh;	// 一个子矩阵对应一个通道的图像
	int submatrix_height = fsize * fsize;
	for (int c = 0; c < channels; ++c, image += channel_size) {
		for (int y = 0; y < submatrix_height; ++y) {
			int dx = y % fsize;		// 核窗元素水平偏移量
			int dy = y / fsize;		// 核窗元素垂直偏移量
			for (int x = 0; x < submatrix_width; ++x) {
				int ix = (x % convw) * stride + dx - padding;	// 图像像素横坐标
				int iy = (x / convw) * stride + dy - padding;	// 图像像素纵坐标
				if (ix > -1 && ix < width && iy > -1 && iy < height) {
					*(matrix++) = image[iy * width + ix];
				} else {
					*(matrix++) = 0;	// 越界元素填充零
				}
			}
		}
	}
}