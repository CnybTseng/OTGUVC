ARCH?=arm
GPU?=1
WINOGRAD?=1
HALF?=1
NNPACK?=0
CLBLAST=0
DEBUG?=0

RM=rm -f
EXE_SUFFIX=

CS=$(wildcard *.c)
CNDS=$(notdir $(CS))
COBJS=$(patsubst %.c,%.o,$(CNDS))

CPPS=$(wildcard *.cpp)
CPPNDS=$(notdir $(CPPS))
CPPOBJS=$(patsubst %.cpp,%.o,$(CPPNDS))

ifeq ($(GPU),1)
	CLS=$(wildcard *.cl)
	CLNDS=$(notdir $(CLS))
	CLOBJS=$(patsubst %.cl,%.o,$(CLNDS))
	CLNAMES=$(patsubst %.cl,%,$(CLNDS))
endif

EXEOBJ=test_znet.o test_aicore.o
CL_KERNEL_FILES=$(CLNAMES)
BINOBJ=agriculture.o cl_common.o $(CLOBJS)
ALLOBJS=$(COBJS) $(CPPOBJS) $(BINOBJ)
OBJS=$(filter-out $(EXEOBJ) $(BINOBJ),$(ALLOBJS))

ifeq ($(ARCH),x86)
	SLIB=aicore.dll
	ALIB=libaicore.a
else ifeq ($(ARCH),arm)
	SLIB=libaicore.so
	ALIB=libaicore.a
else ifeq ($(ARCH),arm64)
	SLIB=libaicore.so.1.0.0
	ALIB=libaicore.a
endif

_EXEC=test_znet test_aicore
ifeq ($(ARCH),x86)
	EXE_SUFFIX=.exe
	EXEC=$(addsuffix $(EXE_SUFFIX),$(_EXEC))
else ifeq ($(ARCH),arm)
	EXEC=$(_EXEC)
else ifeq ($(ARCH),arm64)
	EXEC=$(_EXEC)
endif

ifeq ($(ARCH),x86)
	CC=gcc
else ifeq ($(ARCH),arm)
	CC=$(ANDROID_TOOLCHAIN_PATH)/bin/arm-linux-androideabi-gcc
else ifeq ($(ARCH),arm64)
	CC=$(ANDROID_TOOLCHAIN_PATH)/bin/aarch64-linux-android-gcc
endif

ifeq ($(ARCH),x86)
	AR=ar
	ARFLAGS=rcs
else ifeq ($(ARCH),arm)
	AR=$(ANDROID_TOOLCHAIN_PATH)/arm-linux-androideabi/bin/ar
	ARFLAGS=rcs
else ifeq ($(ARCH),arm64)
	AR=$(ANDROID_TOOLCHAIN_PATH)/aarch64-linux-android/bin/ar
	ARFLAGS=rcs
endif

ifeq ($(ARCH),x86)
	OBJCOPY=objcopy
	OT=pe-i386
	BA=i386
else ifeq ($(ARCH),arm)
	OBJCOPY=$(ANDROID_TOOLCHAIN_PATH)/arm-linux-androideabi/bin/objcopy
	OT=elf32-littlearm
	BA=arm
else ifeq ($(ARCH),arm64)
	OBJCOPY=$(ANDROID_TOOLCHAIN_PATH)/aarch64-linux-android/bin/objcopy
	OT=elf64-littleaarch64
	BA=aarch64
endif

INC=
ifeq ($(ARCH),x86)
	INC+= -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include" \
	-I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include/CL" \
	-I../thirdparty/x86/pthreads-2.9.1/include
else ifeq ($(ARCH),arm)
	INC+= -I../thirdparty/armeabi-v7a/opencl/include \
	-I../thirdparty/armeabi-v7a/NNPACK/include \
	-I../thirdparty/armeabi-v7a/clblast/include
else ifeq ($(ARCH),arm64)
	INC+= -I../thirdparty/arm64-v8a/opencl/include \
	-I../thirdparty/arm64-v8a/NNPACK/include \
	-I../thirdparty/arm64-v8a/clblast/include
endif

CFLAGS=$(INC) -Wall -fPIC -O3 -DCL_TARGET_OPENCL_VERSION=120 -g  -fopenmp \
-DMERGE_BATCHNORM_TO_CONV -DAICORE_BUILD_DLL -DUSE_CL_PROGRAM_BINARY
ifeq ($(GPU),1)
	CFLAGS+= -DOPENCL
	ifeq ($(CLBLAST),1)
		CFLAGS+= -DCLBLAST
	endif
	ifeq ($(WINOGRAD),1)
		CFLAGS+= -DWINOGRAD_CONVOLUTION
	endif
	ifeq ($(HALF),0)
		CFLAGS+= -DUSE_FLOAT
	endif
	ifeq ($(ARCH),arm)
		CFLAGS+= -DION
	endif
	ifeq ($(ARCH),arm64)
		CFLAGS+= -DION
	endif
else ifeq ($(NNPACK),1)
	CFLAGS+= -DNNPACK
endif
ifeq ($(ARCH),x86)
	CFLAGS+= -msse2 -mssse3 -D__INTEL_SSE__ -D_TIMESPEC_DEFINED
else ifeq ($(ARCH),arm)
	CFLAGS+= -march=armv7-a -mfloat-abi=softfp -mfpu=neon -std=c99 -D__ANDROID_API__=24 -pie -fPIE
else ifeq ($(ARCH),arm64)
	CFLAGS+= -march=armv8-a -std=c99 -D__ANDROID_API__=24 -pie -fPIE
endif
ifeq ($(DEBUG),1)
	CFLAGS+= -DNDEBUG
endif

LIB= -L.
LIBS=
ifeq ($(ARCH),x86)
	LIB+= -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/Win32" \
	-L../thirdparty/x86/pthreads-2.9.1/lib/x86
	LIBS+= -lpthread
else ifeq ($(ARCH),arm)
	LIB+= -L../thirdparty/armeabi-v7a/opencl/lib \
	-L../thirdparty/armeabi-v7a/NNPACK/lib \
	-L../thirdparty/armeabi-v7a/clblast/lib
	LIBS+= -lm -llog
else ifeq ($(ARCH),arm64)
	LIB+= -L../thirdparty/arm64-v8a/opencl/lib \
	-L../thirdparty/arm64-v8a/NNPACK/lib \
	-L../thirdparty/arm64-v8a/clblast/lib
	LIBS+= -lm -llog
endif
ifeq ($(GPU),1)
	LIBS+= -lOpenCL
	ifeq ($(CLBLAST),1)
		LIBS+= -lclblast
	endif
else ifeq ($(NNPACK),1)
	LIBS+= -lpthreadpool -lnnpack -lcpuinfo -lclog -llog
endif

LDFLAGS=$(LIB) $(LIBS)
ifeq ($(ARCH),arm)
	LDFLAGS+= -march=armv7-a -Wl,--fix-cortex-a8
endif
ifeq ($(ARCH),arm64)
	LDFLAGS+= -march=armv8-a -Wl,--fix-cortex-a53-835769 \
	-Wl,-rpath-link=../thirdparty/arm64-v8a/system/lib64 \
	-Wl,-rpath-link=../thirdparty/arm64-v8a/system/vendor/lib64
endif

.PHONY:all
all:info bin2obj $(SLIB) $(EXEC)

test_znet$(EXE_SUFFIX):test_znet.o $(OBJS) agriculture.o cl_common.o $(CLOBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_aicore$(EXE_SUFFIX):test_aicore.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -laicore
	
$(ALIB): $(OBJS) agriculture.o cl_common.o $(CLOBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS) agriculture.o cl_common.o $(CLOBJS)
ifeq ($(ARCH),arm)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)
else ifeq ($(ARCH),arm64)
	$(CC) $(CFLAGS) -shared -Wl,-soname,libaicore.so.1 -o $@ $^ $(LDFLAGS)
else
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS) -Wl,--out-implib,$(ALIB)
endif

%.o:%.c
	$(CC) $(CFLAGS) -c $^

bin2obj:
	$(OBJCOPY) --input-target binary --output-target $(OT) --binary-architecture $(BA) agriculture.weights agriculture.o
	$(OBJCOPY) --input-target binary --output-target $(OT) --binary-architecture $(BA) cl_common.h cl_common.o
	@for target in $(CL_KERNEL_FILES);	\
	do	\
	$(OBJCOPY) --input-target binary --output-target $(OT) --binary-architecture $(BA) $$target.cl $$target.o;	\
	done
	
info:
	@echo objects:$(ALLOBJS)
	
.PHONY:clean
clean:
	$(RM) $(ALLOBJS) $(EXEC) $(SLIB) $(ALIB) *.cl.bin