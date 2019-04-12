cmake_minimum_required (VERSION 3.9 FATAL_ERROR)
project (cuda-gs LANGUAGES C CUDA)

if ("${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    #message(${PROJECT_BINARY_DIR})
    #message(${PROJECT_SOURCE_DIR})
    message(FATAL_ERROR "You cannot build in the source directory. Please run cmake from a subdirectory called 'build'")
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_library(gs SHARED gs.c)
add_library(gs_gpu SHARED gs_gpu.cu)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_C_FLAGS}")
    target_link_libraries(gs OpenMP::OpenMP_C)
    target_link_libraries(gs_gpu ${OpenMP})
endif()

find_library(NCCL nccl)
if (NCCL)
    message("NCCL found")
    add_definitions(-DHAS_NCCL)
    target_link_libraries(gs_gpu ${NCCL})
endif ()

install(TARGETS gs gs_gpu DESTINATION lib)
