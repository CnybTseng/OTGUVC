__kernel
void normalize_image(__read_only image2d_t image, __write_only image2d_t normalized_image,
	__private const int resized_width, __private const int resized_height, __private const float scale,
	__private const int roix, __private const int roiy)
{
	int resized_x = get_global_id(0);
	int resized_y = get_global_id(1);
#if 0
	float2 xy = (float2)(resized_x / (float)resized_width, resized_y / (float)resized_height);
#else
	float2 xy = (float2)((scale * (resized_x + 0.5f) - 0.5f + roix) / (float)get_image_width(image),
		(scale * (resized_y + 0.5f) - 0.5f + roiy) / (float)get_image_height(image));
#endif
	const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
	DATA_TYPE4 pixel = READ_IMAGES(image, sampler, xy);

	const int dx = (get_image_width(normalized_image) - resized_width) >> 1;
	const int dy = (get_image_height(normalized_image) - resized_height) >> 1;
	WRITE_IMAGE(normalized_image, (int2)(dx + resized_x, dy + resized_y), pixel);
}


