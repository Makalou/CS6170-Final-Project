cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(ParallelContourTree LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

#Find OpenGL
find_package(OpenGL REQUIRED)

# Add CUDA files to the project
file(GLOB CUDA_SOURCES src/*.cu)

# Specify CUDA compute capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_61)

# Add executable
add_executable(parallel_contour_tree src/main.cpp ${CUDA_SOURCES})

# Link CUDA libraries if necessary
# target_link_libraries(parallel_contour_tree CUDA::cudart)
add_subdirectory(ext)

target_include_directories(parallel_contour_tree PRIVATE ext)
target_include_directories(parallel_contour_tree PRIVATE ext/glfw/include)
target_include_directories(parallel_contour_tree PRIVATE ext/glad/include)

target_link_libraries(parallel_contour_tree stb)
target_link_libraries(parallel_contour_tree glfw)
target_link_libraries(parallel_contour_tree OpenGL::GL)
target_link_libraries(parallel_contour_tree glad)


