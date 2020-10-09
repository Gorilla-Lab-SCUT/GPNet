project(PointNet2)
cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${SOURCE_DIR}/cinclude")
cuda_include_directories("${SOURCE_DIR}/cinclude")

set(BUILD_CMD python "${SOURCE_DIR}/build_ffi.py")
file(GLOB wrapper_headers "${SOURCE_DIR}/cinclude/matrix_k_minimums_cpu_wrapper.h")
file(GLOB wrapper_sources "${SOURCE_DIR}/csrs/matrix_k_minimums_cpu.c")
add_custom_command(OUTPUT "${SOURCE_DIR}/_ext/pointnet2/_pointnet2.so"
           WORKING_DIRECTORY ${SOURCE_DIR}
           COMMAND ${BUILD_CMD} --build --objs ${cuda_kernels}
           DEPENDS ${cuda_kernels}
           DEPENDS ${wrapper_headers}
           DEPENDS ${wrapper_sources}
           VERBATIM)

add_custom_target(pointnet2_ext ALL
          DEPENDS "${SOURCE_DIR}/_ext/pointnet2/_pointnet2.so")

