__kernel
void weight_transform_f4x4_3x3(__read_only image2d_t weight, __write_only image2d_t transformed_weight)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	const int channel_groups = get_global_size(0);
	
	DATA_TYPE4 g[9];
	DATA_TYPE4 Gg[18];
	DATA_TYPE4 GgGT[36];

	#pragma unroll
	for (int i = 0; i < 9; i++) {
		g[i] = READ_IMAGE(weight, (int2)(mul24(gx, 9) + i, gy));
	}
	
	#pragma unroll
	for (int x = 0; x < 3; x++) {
		Gg[x] = 0.25f * g[x];
		Gg[3 + x] = -0.1666667f * (g[x] + g[3 + x] + g[6 + x]);
		Gg[6 + x] = -0.1666667f * (g[x] - g[3 + x] + g[6 + x]);
		Gg[9 + x] = 0.0416667f * (g[x] + 2 * g[3 + x] + 4 * g[6 + x]);
		Gg[12 + x] = 0.0416667f * (g[x] - 2 * g[3 + x] + 4 * g[6 + x]);
		Gg[15 + x] = g[6 + x];
	}
	
	#pragma unroll
	for (int y = 0; y < 6; y++) {
		GgGT[y * 6] = 0.25f * Gg[y * 3];
		GgGT[y * 6 + 1] = -0.1666667f * (Gg[y * 3] + Gg[y * 3 + 1] + Gg[y * 3 + 2]);
		GgGT[y * 6 + 2] = -0.1666667f * (Gg[y * 3] - Gg[y * 3 + 1] + Gg[y * 3 + 2]);
		GgGT[y * 6 + 3] = 0.0416667f * (Gg[y * 3] + 2 * Gg[y * 3 + 1] + 4 * Gg[y * 3 + 2]);
		GgGT[y * 6 + 4] = 0.0416667f * (Gg[y * 3] - 2 * Gg[y * 3 + 1] + 4 * Gg[y * 3 + 2]);
		GgGT[y * 6 + 5] = Gg[y * 3 + 2];
	}
	
	#pragma unroll
	for (int i = 0; i < 36; i++) {
		WRITE_IMAGE(transformed_weight, (int2)(gx + i * channel_groups, gy), GgGT[i]);
	}
}

__kernel
void input_transform_f4x4_3x3(__read_only image2d_t input, __write_only image2d_t transformed_input,
	__private const int input_width, __private const int input_height, __private const int input_channels,
	__private const int ntilesX)
{
	int tile_id = get_global_id(0);
	int channel_block_id = get_global_id(1);
	
	const int channel_blocks = get_global_size(1);
	const int tile_y = tile_id / ntilesX;
	const int tile_x = tile_id - mul24(tile_y, ntilesX);
	const int pixel_x = (tile_x << 2) - 1;
	const int pixel_y = (tile_y << 2) - 1;
	const int pos = channel_block_id * input_width + pixel_x;
	
	DATA_TYPE4 d[36];
	DATA_TYPE4 BTd[36];
	DATA_TYPE4 BTdB[36];

	int y = select(pixel_y, -1, pixel_y < 0 || pixel_y > input_height - 1);
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		int x = select(pos + i, -1, pixel_x + i < 0 || pixel_x + i > input_width - 1);
		d[i] = READ_IMAGE(input, (int2)(x, y));
	}
	
	y = select(pixel_y + 1, -1, pixel_y + 1 < 0 || pixel_y + 1 > input_height - 1);
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		int x = select(pos + i, -1, pixel_x + i < 0 || pixel_x + i > input_width - 1);
		d[6 + i] = READ_IMAGE(input, (int2)(x, y));
	}
	
	y = select(pixel_y + 2, -1, pixel_y + 2 < 0 || pixel_y + 2 > input_height - 1);
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		int x = select(pos + i, -1, pixel_x + i < 0 || pixel_x + i > input_width - 1);
		d[12 + i] = READ_IMAGE(input, (int2)(x, y));
	}
	
	y = select(pixel_y + 3, -1, pixel_y + 3 < 0 || pixel_y + 3 > input_height - 1);
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		int x = select(pos + i, -1, pixel_x + i < 0 || pixel_x + i > input_width - 1);
		d[18 + i] = READ_IMAGE(input, (int2)(x, y));
	}
	
	y = select(pixel_y + 4, -1, pixel_y + 4 < 0 || pixel_y + 4 > input_height - 1);
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		int x = select(pos + i, -1, pixel_x + i < 0 || pixel_x + i > input_width - 1);
		d[24 + i] = READ_IMAGE(input, (int2)(x, y));
	}
	
	y = select(pixel_y + 5, -1, pixel_y + 5 < 0 || pixel_y + 5 > input_height - 1);
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		int x = select(pos + i, -1, pixel_x + i < 0 || pixel_x + i > input_width - 1);
		d[30 + i] = READ_IMAGE(input, (int2)(x, y));
	}
	
	#pragma unroll
	for (int x = 0; x < 6; x++) {
		BTd[x] = 4 * d[x] - 5 * d[12 + x] + d[24 + x];
		BTd[6 + x] = -4 * d[6 + x] - 4 * d[12 + x] + d[18 + x] + d[24 + x];
		BTd[12 + x] = 4 * d[6 + x] - 4 * d[12 + x] - d[18 + x] + d[24 + x];
		BTd[18 + x] = -2 * d[6 + x] - 1 * d[12 + x] + 2 * d[18 + x] + d[24 + x];
		BTd[24 + x] = 2 * d[6 + x] - 1 * d[12 + x] - 2 * d[18 + x] + d[24 + x];
		BTd[30 + x] = 4 * d[6 + x] - 5 * d[18 + x] + d[30 + x];
	}
	
	#pragma unroll
	for (int y = 0; y < 6; y++) {
		BTdB[6 * y] = 4 * BTd[6 * y] - 5 * BTd[6 * y + 2] + BTd[6 * y + 4];
		BTdB[6 * y + 1] = -4 * BTd[6 * y + 1] - 4 * BTd[6 * y + 2] + BTd[6 * y + 3] + BTd[6 * y + 4];
		BTdB[6 * y + 2] = 4 * BTd[6 * y + 1] - 4 * BTd[6 * y + 2] - BTd[6 * y + 3] + BTd[6 * y + 4];
		BTdB[6 * y + 3] = -2 * BTd[6 * y + 1] - 1 * BTd[6 * y + 2] + 2 * BTd[6 * y + 3] + BTd[6 * y + 4];
		BTdB[6 * y + 4] = 2 * BTd[6 * y + 1] - 1 * BTd[6 * y + 2] - 2 * BTd[6 * y + 3] + BTd[6 * y + 4];
		BTdB[6 * y + 5] = 4 * BTd[6 * y + 1] - 5 * BTd[6 * y + 3] + BTd[6 * y + 5];
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		WRITE_IMAGE(transformed_input, (int2)(tile_id, channel_block_id), BTdB[i]);
		channel_block_id += channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		WRITE_IMAGE(transformed_input, (int2)(tile_id, channel_block_id), BTdB[6 + i]);
		channel_block_id += channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		WRITE_IMAGE(transformed_input, (int2)(tile_id, channel_block_id), BTdB[12 + i]);
		channel_block_id += channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		WRITE_IMAGE(transformed_input, (int2)(tile_id, channel_block_id), BTdB[18 + i]);
		channel_block_id += channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		WRITE_IMAGE(transformed_input, (int2)(tile_id, channel_block_id), BTdB[24 + i]);
		channel_block_id += channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		WRITE_IMAGE(transformed_input, (int2)(tile_id, channel_block_id), BTdB[30 + i]);
		channel_block_id += channel_blocks;
	}
}

__kernel
void matrix_multiply(__read_only image2d_t transformed_weight, __read_only image2d_t transformed_input,
	__write_only image2d_t output, __private const int input_channel_blocks, __private const int output_channel_blocks,
	__private const int ntiles)
{
	int tile_id = get_global_id(0) << 2;
	int output_channel_block_global_id = get_global_id(1);
	
	const int batch = output_channel_block_global_id / output_channel_blocks;
	const int output_channel_block_id = output_channel_block_global_id - mul24(batch, output_channel_blocks);
	const int batch_pos = mul24(batch, input_channel_blocks);
	const int output_channel_id = output_channel_block_id << 2;
	
	DATA_TYPE4 a0, a1, a2, a3;
	DATA_TYPE4 b0, b1, b2, b3;
	DATA_TYPE4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
	
	for (int i = 0; i < input_channel_blocks; i++) {
		a0 = READ_IMAGE(transformed_weight, (int2)(i + batch_pos, output_channel_id));
		a1 = READ_IMAGE(transformed_weight, (int2)(i + batch_pos, output_channel_id + 1));
		a2 = READ_IMAGE(transformed_weight, (int2)(i + batch_pos, output_channel_id + 2));
		a3 = READ_IMAGE(transformed_weight, (int2)(i + batch_pos, output_channel_id + 3));
		
		b0 = READ_IMAGE(transformed_input, (int2)(tile_id, batch_pos + i));
		b1 = READ_IMAGE(transformed_input, (int2)(tile_id + 1, batch_pos + i));
		b2 = READ_IMAGE(transformed_input, (int2)(tile_id + 2, batch_pos + i));
		b3 = READ_IMAGE(transformed_input, (int2)(tile_id + 3, batch_pos + i));
		
		c0 += (DATA_TYPE4)(dot(a0, b0), dot(a1, b0), dot(a2, b0), dot(a3, b0));
		c1 += (DATA_TYPE4)(dot(a0, b1), dot(a1, b1), dot(a2, b1), dot(a3, b1));
		c2 += (DATA_TYPE4)(dot(a0, b2), dot(a1, b2), dot(a2, b2), dot(a3, b2));
		c3 += (DATA_TYPE4)(dot(a0, b3), dot(a1, b3), dot(a2, b3), dot(a3, b3));
	}

	WRITE_IMAGE(output, (int2)(tile_id, output_channel_block_global_id), c0);
	if ((tile_id + 1) >= ntiles) return;
	WRITE_IMAGE(output, (int2)(tile_id + 1, output_channel_block_global_id), c1);
	if ((tile_id + 2) >= ntiles) return;
	WRITE_IMAGE(output, (int2)(tile_id + 2, output_channel_block_global_id), c2);
	if ((tile_id + 3) >= ntiles) return;
	WRITE_IMAGE(output, (int2)(tile_id + 3, output_channel_block_global_id), c3);
}

__kernel
void inverse_output_transform_f4x4_3x3(__read_only image2d_t output, __read_only image1d_t biases,
	__write_only image2d_t inverse_transformed_output, __private const int ntilesX,
	__private const int output_width, __private const int output_height)
{
	int tile_id = get_global_id(0);
	int output_channel_block_id = get_global_id(1);
	
	const int tile_y = tile_id / ntilesX;
	const int tile_x = tile_id - mul24(tile_y, ntilesX);
	const int pixel_x = tile_x << 2;
	const int pixel_y = tile_y << 2;
	const int output_channel_blocks = get_global_size(1);
	const int pos = mad24(output_channel_block_id, output_width, pixel_x);
	
	DATA_TYPE4 out[36];
	DATA_TYPE4 b;
	DATA_TYPE4 ATout[24];
	DATA_TYPE4 AToutA[16];
	
	int output_channel_block_global_id = output_channel_block_id;
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		out[i] = READ_IMAGE(output, (int2)(tile_id, output_channel_block_global_id));
		output_channel_block_global_id += output_channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		out[6 + i] = READ_IMAGE(output, (int2)(tile_id, output_channel_block_global_id));
		output_channel_block_global_id += output_channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		out[12 + i] = READ_IMAGE(output, (int2)(tile_id, output_channel_block_global_id));
		output_channel_block_global_id += output_channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		out[18 + i] = READ_IMAGE(output, (int2)(tile_id, output_channel_block_global_id));
		output_channel_block_global_id += output_channel_blocks;
	}
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		out[24 + i] = READ_IMAGE(output, (int2)(tile_id, output_channel_block_global_id));
		output_channel_block_global_id += output_channel_blocks;
	}
	
	b = READ_IMAGE(biases, output_channel_block_id);
	
	#pragma unroll
	for (int i = 0; i < 6; i++) {
		out[30 + i] = READ_IMAGE(output, (int2)(tile_id, output_channel_block_global_id));
		output_channel_block_global_id += output_channel_blocks;
	}
	
	#pragma unroll
	for (int x = 0; x < 6; x++) {
		ATout[x] = out[x] + out[6 + x] + out[12 + x] + out[18 + x] + out[24 + x];
		ATout[6 + x] = out[6 + x] - out[12 + x] + 2 * out[18 + x] - 2 * out[24 + x];
		ATout[12 + x] = out[6 + x] + out[12 + x] + 4 * out[18 + x] + 4 * out[24 + x];
		ATout[18 + x] = out[6 + x] - out[12 + x] + 8 * out[18 + x] - 8 * out[24 + x] + out[30 + x];
	}
	
	#pragma unroll
	for (int y = 0; y < 4; y++) {
		AToutA[y * 4] = ATout[6 * y] + ATout[6 * y + 1] + ATout[6 * y + 2] + ATout[6 * y + 3] + ATout[6 * y + 4];
		AToutA[y * 4 + 1] = ATout[6 * y + 1] - ATout[6 * y + 2] + 2 * ATout[6 * y + 3] - 2 * ATout[6 * y + 4];
		AToutA[y * 4 + 2] = ATout[6 * y + 1] + ATout[6 * y + 2] + 4 * ATout[6 * y + 3] + 4 * ATout[6 * y + 4];
		AToutA[y * 4 + 3] = ATout[6 * y + 1] - ATout[6 * y + 2] + 8 * ATout[6 * y + 3] - 8 * ATout[6 * y + 4] + ATout[6 * y + 5];
	}
	
	#pragma unroll
	for (int i = 0; i < 16; i++) {
		AToutA[i] += b;
		AToutA[i] = select(0.1f * AToutA[i], AToutA[i], AToutA[i] > (DATA_TYPE4)(0));
	}

	const int still_left_x = min(4, output_width - pixel_x);
	const int still_left_y = output_height - pixel_y;
	
	if (still_left_y < 1) return;
	
	#pragma unroll
	for (int i = 0; i < still_left_x; i++) {
		WRITE_IMAGE(inverse_transformed_output, (int2)(pos + i, pixel_y), AToutA[i]);
	}
	
	if (still_left_y < 2) return;

	#pragma unroll
	for (int i = 0; i < still_left_x; i++) {
		WRITE_IMAGE(inverse_transformed_output, (int2)(pos + i, pixel_y + 1), AToutA[4 + i]);
	}
	
	if (still_left_y < 3) return;

	#pragma unroll
	for (int i = 0; i < still_left_x; i++) {
		WRITE_IMAGE(inverse_transformed_output, (int2)(pos + i, pixel_y + 2), AToutA[8 + i]);
	}
	
	if (still_left_y < 4) return;

	#pragma unroll
	for (int i = 0; i < still_left_x; i++) {
		WRITE_IMAGE(inverse_transformed_output, (int2)(pos + i, pixel_y + 3), AToutA[12 + i]);
	}
}

__kernel
void direct_convolution_2d_1x1(__read_only image2d_t weight, __read_only image2d_t input,
	__read_only image1d_t biases, write_only image2d_t output, __private const int width,
	__private const int input_channel_blocks, __private const int leaky_or_linear)
{
	int output_channel_block_id = get_global_id(0);
	int output_pixel_block_id = get_global_id(1);
	int pixel_y = get_global_id(2);

	DATA_TYPE4 w[4];
	DATA_TYPE4 x[4];
	DATA_TYPE4 y[4];
	
	y[0] = READ_IMAGE(biases, output_channel_block_id);
	y[1] = y[0];
	y[2] = y[0];
	y[3] = y[0];
	
	int4 input_pixel_block_x;
	input_pixel_block_x.x = output_pixel_block_id << 2;
	input_pixel_block_x.y = input_pixel_block_x.x + 1;
	input_pixel_block_x.z = input_pixel_block_x.y + 1;
	input_pixel_block_x.w = input_pixel_block_x.z + 1;
	
	input_pixel_block_x.x = select(input_pixel_block_x.x, -1, input_pixel_block_x.x >= width);
	input_pixel_block_x.y = select(input_pixel_block_x.y, -1, input_pixel_block_x.y >= width);
	input_pixel_block_x.z = select(input_pixel_block_x.z, -1, input_pixel_block_x.z >= width);
	input_pixel_block_x.w = select(input_pixel_block_x.w, -1, input_pixel_block_x.w >= width);
	
	int input_row_start = 0;
	int weight_row_start = 0;
	for (int i = 0; i < input_channel_blocks; i++) {		
		w[0] = READ_IMAGE(weight, (int2)(weight_row_start, output_channel_block_id));
		w[1] = READ_IMAGE(weight, (int2)(weight_row_start + 1, output_channel_block_id));
		w[2] = READ_IMAGE(weight, (int2)(weight_row_start + 2, output_channel_block_id));
		w[3] = READ_IMAGE(weight, (int2)(weight_row_start + 3, output_channel_block_id));
		
		x[0] = READ_IMAGE(input, (int2)(input_row_start + input_pixel_block_x.x, pixel_y));
		x[1] = READ_IMAGE(input, (int2)(input_row_start + input_pixel_block_x.y, pixel_y));
		x[2] = READ_IMAGE(input, (int2)(input_row_start + input_pixel_block_x.z, pixel_y));
		x[3] = READ_IMAGE(input, (int2)(input_row_start + input_pixel_block_x.w, pixel_y));
#if 0		
		y[0] += (DATA_TYPE4)(dot(w[0], x[0]), dot(w[1], x[0]), dot(w[2], x[0]), dot(w[3], x[0]));
		y[1] += (DATA_TYPE4)(dot(w[0], x[1]), dot(w[1], x[1]), dot(w[2], x[1]), dot(w[3], x[1]));
		y[2] += (DATA_TYPE4)(dot(w[0], x[2]), dot(w[1], x[2]), dot(w[2], x[2]), dot(w[3], x[2]));
		y[3] += (DATA_TYPE4)(dot(w[0], x[3]), dot(w[1], x[3]), dot(w[2], x[3]), dot(w[3], x[3]));
#else
		y[0] = mad(x[0].x, w[0], y[0]);
		y[0] = mad(x[0].y, w[1], y[0]);
		y[0] = mad(x[0].z, w[2], y[0]);
		y[0] = mad(x[0].w, w[3], y[0]);
		
		y[1] = mad(x[1].x, w[0], y[1]);
		y[1] = mad(x[1].y, w[1], y[1]);
		y[1] = mad(x[1].z, w[2], y[1]);
		y[1] = mad(x[1].w, w[3], y[1]);
		
		y[2] = mad(x[2].x, w[0], y[2]);
		y[2] = mad(x[2].y, w[1], y[2]);
		y[2] = mad(x[2].z, w[2], y[2]);
		y[2] = mad(x[2].w, w[3], y[2]);
		
		y[3] = mad(x[3].x, w[0], y[3]);
		y[3] = mad(x[3].y, w[1], y[3]);
		y[3] = mad(x[3].z, w[2], y[3]);
		y[3] = mad(x[3].w, w[3], y[3]);
#endif		
		input_row_start += width;
		weight_row_start += 4;
	}
	
	const int output_row_start = output_channel_block_id * width;
	
	int output_pixel_x = output_pixel_block_id << 2;
	if (output_pixel_x >= width) return;
	if (leaky_or_linear) y[0] = select(0.1f * y[0], y[0], y[0] > (DATA_TYPE4)(0));
	WRITE_IMAGE(output, (int2)(output_row_start + output_pixel_x, pixel_y), y[0]);
	
	output_pixel_x++;
	if (output_pixel_x >= width) return;
	if (leaky_or_linear) y[1] = select(0.1f * y[1], y[1], y[1] > (DATA_TYPE4)(0));
	WRITE_IMAGE(output, (int2)(output_row_start + output_pixel_x, pixel_y), y[1]);
	
	output_pixel_x++;
	if (output_pixel_x >= width) return;
	if (leaky_or_linear) y[2] = select(0.1f * y[2], y[2], y[2] > (DATA_TYPE4)(0));
	WRITE_IMAGE(output, (int2)(output_row_start + output_pixel_x, pixel_y), y[2]);
	
	output_pixel_x++;
	if (output_pixel_x >= width) return;
	if (leaky_or_linear) y[3] = select(0.1f * y[3], y[3], y[3] > (DATA_TYPE4)(0));
	WRITE_IMAGE(output, (int2)(output_row_start + output_pixel_x, pixel_y), y[3]);
}


