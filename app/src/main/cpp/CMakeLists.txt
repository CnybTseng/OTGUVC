# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.4.1)

set(JNI_LIB_DIR ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_SOURCE_DIR}/v4l2)
include_directories(${CMAKE_SOURCE_DIR}/ffmpeg)
include_directories(${CMAKE_SOURCE_DIR}/ffmpeg/include)
include_directories(${CMAKE_SOURCE_DIR}/libusb/include)
include_directories(${CMAKE_SOURCE_DIR}/libuvc/include)
include_directories(${CMAKE_SOURCE_DIR}/aicore)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


add_library(native SHARED native.cpp utils/lock.cpp v4l2/v4l2.cpp
		libusb/src/core.c libusb/src/descriptor.c libusb/src/hotplug.c libusb/src/io.c libusb/src/strerror.c libusb/src/sync.c
		libusb/src/os/android_netlink.c libusb/src/os/android_usbfs.c libusb/src/os/poll_posix.c libusb/src/os/threads_posix.c
		libuvc/src/ctrl.c libuvc/src/device.c libuvc/src/diag.c libuvc/src/frame.c
		libuvc/src/init.c libuvc/src/misc.c libuvc/src/stream.c

		#ffmpeg/local_ffmpeg.cpp ffmpeg/ffmpeg_log.cpp

		#aicore/maxpool.cl aicore/convolution.cl aicore/blas.cl aicore/resample.cl aicore/test.cl aicore/test.cl
		#aicore/aicore.c aicore/activation.c aicore/batchnorm_layer.c aicore/bitmap.c
		#aicore/box.c aicore/cl_wrapper.c aicore/list.c
		#aicore/convolutional_layer.c aicore/fifo.c aicore/gemm.c aicore/half.c
		#aicore/im2col.c aicore/image.c aicore/maxpool_layer.c
		#aicore/neon_math.c aicore/resample_layer.c aicore/route_layer.c aicore/sse_math.c
		#aicore/test_aicore.c aicore/test_znet.c aicore/winograd_convolution.c
		#aicore/yolo_layer.c aicore/znet.c aicore/zutils.c
		)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/utils)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/v4l2)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/ffmpeg)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/ffmpeg/include)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/libusb/include)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/libuvc/include)
target_include_directories(native PRIVATE ${CMAKE_SOURCE_DIR}/aicore)

find_library(log-lib log)

add_library(aicore SHARED IMPORTED)
set_target_properties(aicore PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libaicore.so)
set_property(TARGET aicore PROPERTY IMPORTED_NO_SONAME 1)

#add_library(avdevice SHARED IMPORTED)
#set_target_properties(avdevice PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libavdevice.so)
#
#add_library(avformat SHARED IMPORTED)
#set_target_properties(avformat PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libavformat.so)
#
#add_library(avcodec SHARED IMPORTED)
#set_target_properties(avcodec PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libavcodec.so)
#
#add_library(avutil SHARED IMPORTED)
#set_target_properties(avutil PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libavutil.so)
#
#add_library(avfilter SHARED IMPORTED)
#set_target_properties(avfilter PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libavfilter.so)
#
#add_library(swresample SHARED IMPORTED)
#set_target_properties(swresample PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libswresample.so)
#
#add_library(swscale SHARED IMPORTED)
#set_target_properties(swscale PROPERTIES IMPORTED_LOCATION ${JNI_LIB_DIR}/libswscale.so)

target_link_libraries(
        native
        #---------split-------------#
		aicore
		#---------split-------------#
		#avformat
        #avdevice
        #avfilter
        #avcodec
        #avutil
        #swresample
        #swscale
        android
        ${log-lib}
)