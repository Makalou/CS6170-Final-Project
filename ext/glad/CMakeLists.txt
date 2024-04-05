cmake_minimum_required(VERSION 3.22)
project(glad)

set(CMAKE_CXX_STANDARD 17)

add_library(glad src/glad.c)

target_include_directories(glad SYSTEM PRIVATE include)