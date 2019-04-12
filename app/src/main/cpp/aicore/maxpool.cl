__kernel
void maxpool_2x2(__read_only image2d_t input, __write_only image2d_t output,
	__private const int input_width, __private const int input_height,
	__private const int output_width, __private const int output_height,
	__private const int padding, __private const int stride)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	const int offset = -(padding >> 1);
	const int channel_block_id = gx / output_width;
	const int lx = mad24(gx - mul24(channel_block_id, output_width), stride, offset);
	const int ly = mad24(gy, stride, offset);

	DATA_TYPE4 pixels[4];
#ifdef FLOAT
	DATA_TYPE4 maximum = -FLT_MAX;
#else
	DATA_TYPE4 maximum = -HALF_MAX;
#endif	
	const int x = mad24(channel_block_id, input_width, lx);
	const int y = ly;
	const int4 flag1 = lx < 0 || lx > input_width - 1;
	const int4 flag2 = ly < 0 || ly > input_height - 1;
	const int4 flag3 = lx + 1 < 0 || lx + 1 >input_width - 1;
	const int4 flag4 = ly + 1 < 0 || ly + 1 > input_height - 1;
	
	pixels[0] = select(READ_IMAGE(input, (int2)(x, y)), -FLT_MAX, flag1 || flag2);
	pixels[1] = select(READ_IMAGE(input, (int2)(x + 1, y)), -FLT_MAX, flag3 || flag2);
	pixels[2] = select(READ_IMAGE(input, (int2)(x, y + 1)), -FLT_MAX, flag1 || flag4);
	pixels[3] = select(READ_IMAGE(input, (int2)(x + 1, y + 1)), -FLT_MAX, flag3 || flag4);
	
	maximum = max(maximum, pixels[0]);
	maximum = max(maximum, pixels[1]);
	maximum = max(maximum, pixels[2]);
	maximum = max(maximum, pixels[3]);
	
	WRITE_IMAGE(output, (int2)(gx, gy), maximum);
}


