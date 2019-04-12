__kernel
void upsampleB1(__read_only image2d_t input, __write_only image2d_t output,
	__private const int stride, __private const int input_width, __private const int output_width)
{
	int gx = get_global_id(0);
	int output_pixel_y = get_global_id(1);

	const int channel_block_id = gx / output_width;
	const int output_pixel_x = gx - mul24(channel_block_id, output_width);
	
	int input_pixel_x = output_pixel_x / stride;
	const int input_pixel_y = output_pixel_y / stride;
	
	input_pixel_x += channel_block_id * input_width;
	DATA_TYPE4 val = READ_IMAGE(input, (int2)(input_pixel_x, input_pixel_y));

	WRITE_IMAGE(output, (int2)(gx, output_pixel_y), val);
}


