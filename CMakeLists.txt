cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project (cuda-gs LANGUAGES C CUDA)

if ("${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    #message(${PROJECT_BINARY_DIR})
    #message(${PROJECT_SOURCE_DIR})
    message(FATAL_ERROR "You cannot build in the source directory. Please run cmake from a subdirectory called 'build'")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_library(gs SHARED gs.c)
add_library(gs_gpu SHARED gs_gpu.cu)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP was found")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_C_FLAGS}")
    target_link_libraries(gs OpenMP::OpenMP_C)
    target_link_libraries(gs_gpu ${OpenMP_C_LIBRARIES})
else ()
    message("OpenMP was not found")
endif()

find_library(NCCL nccl)
if (NCCL)
    message("NCCL was found")
    add_definitions(-DHAS_NCCL)
    target_link_libraries(gs_gpu ${NCCL})
else ()
    message("NCCL was not found")
endif ()

install(TARGETS gs gs_gpu DESTINATION lib)
