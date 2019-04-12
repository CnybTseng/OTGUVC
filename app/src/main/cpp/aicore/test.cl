#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void test(__read_only image2d_t float_image, __write_only image2d_t half_image)
{
	const int gx = get_global_id(0);
	const int gy = get_global_id(1);
	float4 fval = read_imagef(float_image, (int2)(gx, gy));
	half8 hval = as_half8(fval);
	write_imageh(half_image, (int2)((gx << 1), gy), hval.lo);
	write_imageh(half_image, (int2)((gx << 1) + 1, gy), hval.hi);
}

