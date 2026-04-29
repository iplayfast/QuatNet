cmake_minimum_required(VERSION 3.10)
project(quaternary_nn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O3)

include_directories(include)

add_executable(quaternary_nn
    src/QuaternaryNN.cpp
    src/GGUFHelper.cpp
    src/main.cpp
)
