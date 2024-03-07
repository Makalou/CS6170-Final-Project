cmake_minimum_required(VERSION 3.20)
# Change template to your own project name
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(stb C CXX)

add_library(stb STATIC stb_image_write.cpp stb_image.cpp)
