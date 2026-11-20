cmake_minimum_required(VERSION 3.20)
project(CustomGPT VERSION 1.0)

if(NOT APPLE)
    message(FATAL_ERROR "CustomGPT inference currently requires macOS Apple Accelerate (cblas/vDSP).")
endif()

# Mandatory for modern features like std::unordered_map and smart pointers.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Tell the compiler where the header files (.hpp) live
include_directories(include)

# Create the main executable and link your source files
# Add any new .cpp files you create in the src/ folder to this list.
add_executable(custom_gpt_engine src/main.cpp
        src/tokenizer.cpp
        src/embedding.cpp
        src/positional_encoding.cpp
        src/attention.cpp
        src/feed_forward.cpp
        src/transformer_block.cpp)

target_compile_definitions(custom_gpt_engine PRIVATE ACCELERATE_NEW_LAPACK ACCELERATE_LAPACK_ILP64)
target_compile_options(custom_gpt_engine PRIVATE -O3 -Wall -Wextra)
# Include Apple Accelerate for faster speeds
target_link_libraries(custom_gpt_engine "-framework Accelerate")

option(BUILD_PYTHON_BINDINGS "Build pybind11 custom_tokenizer module" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(custom_tokenizer
        tokenizer_bindings.cpp
        src/tokenizer.cpp)
    target_include_directories(custom_tokenizer PRIVATE include)
    target_compile_features(custom_tokenizer PRIVATE cxx_std_20)
endif()
