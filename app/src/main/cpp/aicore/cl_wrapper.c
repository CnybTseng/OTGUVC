#ifdef OPENCL
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <fcntl.h>
#ifdef __linux__
#	include <sys/ioctl.h>
#	include <sys/mman.h>
#	include <unistd.h>
#	include <errno.h>
#	include <linux/ion.h>
#	include "msm_ion.h"
#endif
#include "cl_wrapper.h"

static cl_program __attribute__((unused)) cl_create_program_with_source(cl_device_id device, cl_context context, const char *filename,
	const char *options, cl_int *errcode);
static cl_program cl_create_program_from_binary(cl_device_id device, cl_context context, const char *filename,
	const char *options, cl_int *errcode);
static cl_program cl_make_wrapper_program_from_buffer(cl_wrapper wrapper, char *buffer, const char *options, cl_int *errcode);
static cl_int cl_save_binary_program(cl_device_id device, cl_program program, const char *filename);
#ifdef __linux__
static cl_ion_context cl_make_ion_buffer_internal(cl_wrapper wrapper, size_t size, unsigned int ion_allocation_flags,
	cl_uint host_cache_policy);
static cl_ion_context cl_make_ion_buffer(cl_wrapper wrapper, size_t size);
#endif
static void cl_host_mem_free(int n, ...);

cl_wrapper cl_create_wrapper(cl_int *errcode)
{
	cl_wrapper wrapper;
	
	const cl_uint num_entries = 1;
	*errcode = clGetPlatformIDs(num_entries, &wrapper.platform, NULL);
	if (CL_SUCCESS != *errcode) return wrapper;

	*errcode = clGetDeviceIDs(wrapper.platform, CL_DEVICE_TYPE_GPU, num_entries, &wrapper.device, NULL);
	if (CL_SUCCESS != *errcode) return wrapper;

	cl_context_properties context_properties[] = {		
		CL_CONTEXT_PLATFORM, (cl_context_properties)wrapper.platform, 0
	};
	
	wrapper.context = clCreateContext(context_properties, num_entries, &wrapper.device, NULL, NULL, errcode);
	if (!wrapper.context || CL_SUCCESS != *errcode) return wrapper;
	
	cl_command_queue_properties command_queue_properties = 0;
#ifdef NDEBUG
	command_queue_properties = CL_QUEUE_PROFILING_ENABLE;
#endif
	
	wrapper.command_queue = clCreateCommandQueue(wrapper.context, wrapper.device, command_queue_properties, errcode);
	if (!wrapper.command_queue || CL_SUCCESS != *errcode) return wrapper;
#ifdef __linux__	
	wrapper.ion_device_fd = open("/dev/ion", O_RDONLY);
	if (wrapper.ion_device_fd < 0) {
		*errcode = errno;
		return wrapper;
	}
#endif	
	return wrapper;
}

cl_program cl_make_wrapper_program(cl_wrapper wrapper, const char *filename, char *buffer, const char *options, cl_int *errcode)
{
	char binary_filename[1024];
	strcpy(binary_filename, filename);
	strcat(binary_filename, ".bin");

	cl_program program = cl_create_program_from_binary(wrapper.device, wrapper.context, binary_filename, options, errcode);
	if (!program) {
		program = cl_make_wrapper_program_from_buffer(wrapper, buffer, options, errcode);
		if (!program) return program;
		*errcode = cl_save_binary_program(wrapper.device, program, binary_filename);
	}
	
	return program;
}

cl_kernel cl_make_wrapper_kernel(cl_wrapper wrapper, cl_program program, const char *kername, cl_int *errcode)
{
	return clCreateKernel(program, kername, errcode);
}

void cl_destroy_wrapper(cl_wrapper wrapper)
{
	clReleaseCommandQueue(wrapper.command_queue);
	clReleaseContext(wrapper.context);
#ifdef __linux__
	close(wrapper.ion_device_fd);
#endif
}

void cl_print_platform_info(cl_wrapper wrapper, cl_platform_info param_name)
{
	switch (param_name) {
	case CL_PLATFORM_PROFILE:
		break;
	case CL_PLATFORM_VERSION: {
		char version[1024];
		clGetPlatformInfo(wrapper.platform, param_name, sizeof(version), version, NULL);
		printf("%s\n", version);
	}	break;
	case CL_PLATFORM_NAME:
		break;
	case CL_PLATFORM_VENDOR:
		break;
	case CL_PLATFORM_EXTENSIONS: {
		char extensions[1024] = {'\0'};
		clGetPlatformInfo(wrapper.platform, param_name, sizeof(extensions), extensions, NULL);
		if (strlen(extensions) <= 1) printf("couldn't identify available OpenCL platform extensions\n");
		else printf("platform extensions: %s\n", extensions);
	}	break;
	default:
		break;
	}
}

void cl_print_device_info(cl_wrapper wrapper, cl_device_info param_name)
{
	switch (param_name) {
	case CL_DEVICE_IMAGE2D_MAX_WIDTH: {
		size_t image2d_max_width;
		clGetDeviceInfo(wrapper.device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &image2d_max_width, NULL);
		printf("image2d_max_width: %d\n", (int)image2d_max_width);
	}	break;
	case CL_DEVICE_IMAGE2D_MAX_HEIGHT: {
		size_t image2d_max_height;
		clGetDeviceInfo(wrapper.device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &image2d_max_height, NULL);
		printf("image2d_max_height: %d\n", (int)image2d_max_height);
	}	break;
	case CL_DEVICE_IMAGE3D_MAX_WIDTH: {
		size_t image3d_max_width;
		clGetDeviceInfo(wrapper.device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &image3d_max_width, NULL);
		printf("image3d_max_width: %d\n", (int)image3d_max_width);
	} 	break;
	case CL_DEVICE_IMAGE3D_MAX_HEIGHT: {
		size_t image3d_max_height;
		clGetDeviceInfo(wrapper.device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &image3d_max_height, NULL);
		printf("image3d_max_height: %d\n", (int)image3d_max_height);
	} 	break;
	case CL_DEVICE_IMAGE3D_MAX_DEPTH: {
		size_t image3d_max_depth;
		clGetDeviceInfo(wrapper.device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &image3d_max_depth, NULL);
		printf("image3d_max_depth: %d\n", (int)image3d_max_depth);
	} 	break;
	case CL_DEVICE_EXTENSIONS: {
		char extensions[1024] = {'\0'};
		clGetDeviceInfo(wrapper.device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
		if (strlen(extensions) <= 1) printf("couldn't identify available OpenCL device extensions\n");
		else printf("device extensions: %s\n", extensions);
	}	break;
	default:
		break;
	}
}

#ifdef __linux__
size_t cl_get_ion_image_row_pitch(cl_wrapper wrapper, cl_image_format image_format, cl_image_desc image_desc)
{
	size_t image_row_pitch = 0;
	cl_int errcode = clGetDeviceImageInfoQCOM(wrapper.device, image_desc.image_width, image_desc.image_height,
		&image_format, CL_IMAGE_ROW_PITCH, sizeof(size_t), &image_row_pitch, NULL);
    if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clGetDeviceImageInfoQCOM fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return 0;
	}
	
	return image_row_pitch;
}

cl_ion_context cl_make_ion_buffer_for_nonplanar_image(cl_wrapper wrapper, cl_image_desc image_desc)
{
	cl_int errcode;
	size_t padding_in_bytes = 0;
	
	errcode = clGetDeviceInfo(wrapper.device, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(size_t), &padding_in_bytes, NULL);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clGetDeviceImageInfoQCOM fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		exit(errcode);
	}
	
	const size_t total_bytes = image_desc.image_row_pitch * image_desc.image_height + padding_in_bytes;
    return cl_make_ion_buffer(wrapper, total_bytes);
}

void cl_free_ion_context(cl_wrapper wrapper, cl_ion_context ion_context)
{
	munmap(ion_context.ion_mem.ion_hostptr, ion_context.allocation_data.len);
	close(ion_context.fd_data.fd);
	ioctl(wrapper.ion_device_fd, ION_IOC_FREE, &ion_context.handle_data);
}
#endif

cl_program cl_create_program_with_source(cl_device_id device, cl_context context, const char *filename,
	const char *options, cl_int *errcode)
{		
	struct stat stbuf;
	int ret = stat(filename, &stbuf);
	if (ret) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	char *strings = calloc(stbuf.st_size + 1, sizeof(char));
	if (!strings) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	strings[stbuf.st_size] = '\0';
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		free(strings);
		return 0;
	}
	
	fread(strings, sizeof(char), stbuf.st_size, fp);
	fclose(fp);
	
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&strings, NULL, errcode);
	free(strings);
	
	if (!program || CL_SUCCESS != *errcode) return program;
	
	*errcode = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if (CL_SUCCESS != *errcode) {
		char buildinfo[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "clGetProgramBuildInfo:\n%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

cl_program cl_create_program_from_binary(cl_device_id device, cl_context context, const char *filename,
	const char *options, cl_int *errcode)
{	
	struct stat stbuf;
	int ret = stat(filename, &stbuf);
	if (ret) {
		fprintf(stderr, "stat[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	unsigned char *binaries = calloc(stbuf.st_size + 1, sizeof(unsigned char));
	if (!binaries) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
	
	binaries[stbuf.st_size] = '\0';
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		free(binaries);
		return 0;
	}
	
	const size_t length = stbuf.st_size;
	fread(binaries, sizeof(unsigned char), length, fp);
	fclose(fp);
	
	cl_int binary_status;
	cl_program program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char **)&binaries,
		&binary_status, errcode);
	free(binaries);
	
	if (!program || CL_SUCCESS != *errcode) return program;
	
	*errcode = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if (CL_SUCCESS != *errcode) {
		char buildinfo[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "clGetProgramBuildInfo:\n%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

cl_program cl_make_wrapper_program_from_buffer(cl_wrapper wrapper, char *buffer, const char *options, cl_int *errcode)
{
	cl_program program = clCreateProgramWithSource(wrapper.context, 1, (const char **)&buffer, NULL, errcode);
	
	if (!program || CL_SUCCESS != *errcode) return program;
	
	*errcode = clBuildProgram(program, 1, &wrapper.device, options, NULL, NULL);
	if (CL_SUCCESS != *errcode) {
		char buildinfo[16384];
		clGetProgramBuildInfo(program, wrapper.device, CL_PROGRAM_BUILD_LOG, sizeof(buildinfo), buildinfo, NULL);
		fprintf(stderr, "clGetProgramBuildInfo:\n%s\n", buildinfo);
		clReleaseProgram(program);
	}
	
	return program;
}

cl_int cl_save_binary_program(cl_device_id device, cl_program program, const char *filename)
{
	cl_uint ndevices;
	cl_int errcode = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &ndevices, NULL);
	if (CL_SUCCESS != errcode) return errcode;
	
	cl_device_id *devices = calloc(ndevices, sizeof(cl_device_id));
	if (!devices) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return CL_WRAPPER_CALLOC_FAIL;
	}
	
	errcode = clGetProgramInfo(program, CL_PROGRAM_DEVICES, ndevices * sizeof(cl_device_id), devices, NULL);
	if (CL_SUCCESS != errcode) {
		free(devices);
		return errcode;
	}
	
	size_t *sizes = calloc(ndevices, sizeof(size_t));
	if (!sizes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		free(devices);
		return CL_WRAPPER_CALLOC_FAIL;
	}
	
	errcode = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, ndevices * sizeof(size_t), sizes, NULL);
	if (CL_SUCCESS != errcode) {
		cl_host_mem_free(2, devices, sizes);
		return errcode;
	}
	
	unsigned char **binaries = calloc(ndevices, sizeof(unsigned char *));
	if (!binaries) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cl_host_mem_free(2, devices, sizes);
		return CL_WRAPPER_CALLOC_FAIL;
	}
		
	for (cl_uint i = 0; i < ndevices; i++) {
		binaries[i] = calloc(sizes[i], 1);
		if (!binaries[i]) {
			fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
			for (cl_uint j = 0; j < i; j++) free(binaries[j]);
			cl_host_mem_free(3, binaries, devices, sizes);
			return CL_WRAPPER_CALLOC_FAIL;
		}
	}
	
	errcode = clGetProgramInfo(program, CL_PROGRAM_BINARIES, ndevices * sizeof(unsigned char *), binaries, NULL);
	if (CL_SUCCESS != errcode) {
		for (cl_uint j = 0; j < ndevices; j++) free(binaries[j]);
		cl_host_mem_free(3, binaries, devices, sizes);
		return errcode;
	}
	
	for (cl_uint i = 0; i < ndevices; i++) {
		if (devices[i] != device) continue;
		FILE *fp = fopen(filename, "wb");
		if (fp) {
			fwrite(binaries[i], sizeof(unsigned char), sizes[i], fp);
			fclose(fp);
		} else fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		break;
	}
	
	for (cl_uint j = 0; j < ndevices; j++) free(binaries[j]);
	cl_host_mem_free(3, binaries, devices, sizes);
	
	return CL_SUCCESS;
}

#ifdef __linux__
cl_ion_context cl_make_ion_buffer_internal(cl_wrapper wrapper, size_t size, unsigned int ion_allocation_flags,
	cl_uint host_cache_policy)
{
	cl_int  errcode;
    cl_uint device_page_size;
	cl_ion_context ion_context;

    errcode = clGetDeviceInfo(wrapper.device, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(device_page_size), &device_page_size, NULL);
    if (errcode != CL_SUCCESS) {
		fprintf(stderr, "CL_DEVICE_PAGE_SIZE_QCOM fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		exit(errcode);
    }
	
    ion_context.allocation_data.len          = size;
    ion_context.allocation_data.align        = device_page_size;
    ion_context.allocation_data.heap_id_mask = ION_HEAP(ION_IOMMU_HEAP_ID);
    ion_context.allocation_data.flags        = ion_allocation_flags;
	int ret = ioctl(wrapper.ion_device_fd, ION_IOC_ALLOC, &ion_context.allocation_data);
    if (ret) {
		fprintf(stderr, "allocating ion memory fail:%d\n", errno);
		exit(errno);
    }

    ion_context.handle_data.handle = ion_context.allocation_data.handle;
    ion_context.fd_data.handle     = ion_context.allocation_data.handle;
	ret = ioctl(wrapper.ion_device_fd, ION_IOC_MAP, &ion_context.fd_data);
    if (ret) {
        ioctl(wrapper.ion_device_fd, ION_IOC_FREE, &ion_context.handle_data);
		fprintf(stderr, "mapping ion memory to cpu-addressable fd fail:%d.\n", errno);
		exit(errno);
    }

    void *host_addr = mmap(NULL, ion_context.allocation_data.len, PROT_READ | PROT_WRITE, MAP_SHARED, ion_context.fd_data.fd, 0);
    if (MAP_FAILED == host_addr) {
        close(ion_context.fd_data.fd);
        ioctl(wrapper.ion_device_fd, ION_IOC_FREE, &ion_context.handle_data);
		fprintf(stderr, "mmapping fd to pointer fail:%d.\n", errno);
		exit(errno);
    }

    ion_context.ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
    ion_context.ion_mem.ext_host_ptr.host_cache_policy = host_cache_policy;
    ion_context.ion_mem.ion_filedesc                   = ion_context.fd_data.fd;
    ion_context.ion_mem.ion_hostptr                    = host_addr;
	
	return ion_context;
}

cl_ion_context cl_make_ion_buffer(cl_wrapper wrapper, size_t size)
{
	return cl_make_ion_buffer_internal(wrapper, size, 0, CL_MEM_HOST_UNCACHED_QCOM);
}
#endif

void cl_host_mem_free(int n, ...)
{
	va_list ap;
	va_start(ap, n);
	
	for (int i = 0; i < n; ++i) {
		void *p = va_arg(ap, void *);
		if (p) {
			free(p);
			p = NULL;
		}
	}
	
	va_end(ap);
}
#endif