__kernel void
sgemm_nn_common(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{	
	const int gx = get_global_id(0);
	const int gy = get_global_id(1);
	
	float sum = beta * C[gy * ldc + gx];
	for (int l = 0; l < k; ++l) {
		sum += alpha * A[gy * lda + l] * B[l * ldb + gx];
	}
	
	C[gy * ldc + gx] = sum;
}

__kernel void
sgemm_nn_8x4(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int gx = get_global_id(0) << 2;
	const int gy = get_global_id(1) << 3;

	float  a[8];
	float4 b;
	float4 c[8];
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		c[j] = beta * vload4(0, C + (gy + j) * ldc + gx);
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			a[j] = A[(gy + j) * lda + i];
		}
		
		b = vload4(0, B + i * ldb + gx);
		
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		vstore4(c[j], 0, C + (gy + j) * ldc + gx);
	}
}

__kernel void
sgemm_nn_8x8(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int gx = get_global_id(0) << 3;
	const int gy = get_global_id(1) << 3;
	
	float  a[8];
	float8 b;
	float8 c[8];
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		c[j] = beta * vload8(0, C + (gy + j) * ldc + gx);
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			a[j] = A[(gy + j) * lda + i];
		}
		
		b = vload8(0, B + i * ldb + gx);
		
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		vstore8(c[j], 0, C + (gy + j) * ldc + gx);
	}
}

__kernel void
sgemm_nn_8x16(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int gx = get_global_id(0) << 4;
	const int gy = get_global_id(1) << 3;
	
	float   a[8];
	float16 b;
	float16 c[8];
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		c[j] = beta * vload16(0, C + (gy + j) * ldc + gx);
	}
	
	for (int i = 0; i < k; ++i) {
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			a[j] = A[(gy + j) * lda + i];
		}
		
		b = vload16(0, B + i * ldb + gx);
		
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			c[j] += alpha * a[j] * b;
		}
	}
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		vstore16(c[j], 0, C + (gy + j) * ldc + gx);
	}
}

__kernel void
sgemm_nn_8x4_tp(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global const float *A, __private const int lda,
	__read_only image2d_t B, __private const float beta, __global float *C, __private const int ldc)
{
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	float4 a[8];
	float4 b[4];
	float4 c[8];
	
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		c[i] = beta * vload4(0, C + ((gy << 3) + i) * ldc + (gx << 2));
	}
	
	for (int pos = 0; pos < k; pos += 4) {
		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			a[i] = vload4(0, A + ((gy << 3) + i) * lda + pos);
		}
		
		#pragma unroll
		for (int i = 0; i < 4; ++i) {
			b[i] = read_imagef(B, (int2)(gx, pos + i));
		}
		
		#pragma unroll
		for (int i = 0; i < 8; ++i) {
			c[i] += alpha * (a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3]);
		}
	}
	
	#pragma unroll
	for (int i = 0; i < 8; ++i) {
		vstore4(c[i], 0, C + ((gy << 3) + i) * ldc + (gx << 2));
	}
}

__kernel void
sgemm_nn_sm(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int ly = get_local_id(0);
	const int lx = get_local_id(1);

	__local float tile_A[16][16];
	__local float tile_B[16][16];

	const int gy = (get_group_id(0) << 4) + ly;
	const int gx = (get_group_id(1) << 4) + lx;
	
	float acc = 0;
	const int ntiles = k >> 4;
	for (int i = 0; i < ntiles; ++i) {
		tile_A[ly][lx] = A[gy * lda + (i << 4) + lx];
		tile_B[ly][lx] = B[((i << 4) + ly) * ldb + gx];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		#pragma unroll
		for (int j = 0; j < 16; ++j) {
			acc += alpha * tile_A[ly][j] * tile_B[j][lx];
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	float c = beta * C[gy * ldc + gx];
	C[gy * ldc + gx] = acc + c;
}

__kernel void
sgemm_nt_common(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int gx = get_global_id(0);
	const int gy = get_global_id(1);
	
	float sum = beta * C[gy * ldc + gx];
	for (int l = 0; l < k; ++l) {
		sum += alpha * A[gy * lda + l] * B[gx * ldb + l];
	}
	
	C[gy * ldc + gx] = sum;
}

__kernel void
sgemm_nt_1x4(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int gx = get_global_id(0) << 2;
	const int gy = get_global_id(1);
	
	float4 a;
	float4 b;
	float4 c4[4] = {0, 0, 0, 0};
	float4 c = {0, 0, 0, 0};
	
	c = beta * vload4(0, C + gy * ldc + gx);
	
	for (int i = 0; i < k; i += 4) {
		a = vload4(0, A + gy * lda + i);
		for (int j = 0; j < 4; ++j) {
			b = vload4(0, B + (gx + j) * ldb + i);
			c4[j] += alpha * a * b;
		}
	}
	
	c.x += c4[0].x + c4[0].y + c4[0].z + c4[0].w;
	c.y += c4[1].x + c4[1].y + c4[1].z + c4[1].w;
	c.z += c4[2].x + c4[2].y + c4[2].z + c4[2].w;
	c.w += c4[3].x + c4[3].y + c4[3].z + c4[3].w;
	
	vstore4(c, 0, C + gy * ldc + gx);
}

__kernel void
sgemm_nt_8x4(__private const int m, __private const int n, __private const int k,
	__private const float alpha, __global float *A, __private const int lda, __global float *B,
	__private const int ldb, __private const float beta, __global float *C, __private const int ldc)
{
	const int gx = get_global_id(0) << 2;
	const int gy = get_global_id(1) << 3;
	
	float4 a[8];
	float4 b[4];
	float4 c84[8][4] = {
		{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
		{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
	float4 c[8] = {0, 0, 0, 0};
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		c[j] = beta * vload4(0, C + (gy + j) * ldc + gx);
	}
	
	for (int i = 0; i < k; i += 4) {
		#pragma unroll
		for (int j = 0; j < 8; ++j) {
			a[j] = vload4(0, A + (gy + j) * lda + i);
		}
		
		#pragma unroll
		for (int j = 0; j < 4; ++j) {
			b[j] = vload4(0, B + (gx + j) * ldb + i);
		}
		
		#pragma unroll
		for (int j = 0; j < 8; j++) {
			c84[j][0] += alpha * a[j] * b[0];
			c84[j][1] += alpha * a[j] * b[1];
			c84[j][2] += alpha * a[j] * b[2];
			c84[j][3] += alpha * a[j] * b[3];
		}
	}
	
	#pragma unroll
	for (int j = 0; j < 8; ++j) {
		c[j].x += c84[j][0].x + c84[j][0].y + c84[j][0].z + c84[j][0].w;
		c[j].y += c84[j][1].x + c84[j][1].y + c84[j][1].z + c84[j][1].w;
		c[j].z += c84[j][2].x + c84[j][2].y + c84[j][2].z + c84[j][2].w;
		c[j].w += c84[j][3].x + c84[j][3].y + c84[j][3].z + c84[j][3].w;
		
		vstore4(c[j], 0, C + (gy + j) * ldc + gx);
	}
}


