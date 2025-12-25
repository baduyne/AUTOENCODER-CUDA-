cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dl_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bật/tắt GPU phase
option(USE_CUDA "Build with CUDA support" OFF)

# Core library - đã chuẩn hóa chỉ dùng các file thuần C++ float*
add_library(dl_core
    src/cpu/src/autoencoder_cpu.cpp
    src/cpu/src/conv2d_cpu.cpp
    src/cpu/src/cifar10_dataset.cpp
    src/cpu/src/layers_cpu.cpp
    src/cpu/src/svm_trainer.cpp
    src/cpu/src/svm_evaluator.cpp
    src/cpu/src/training_pipeline_cpu.cpp
    src/cpu/src/main.cpp
    third_party/libsvm/svm.cpp
)

target_include_directories(dl_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cpu/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/libsvm
)

target_link_libraries(dl_core PUBLIC)

set_property(TARGET dl_core PROPERTY CXX_STANDARD 17)

# Nếu build với CUDA → thêm file .cu & define USE_CUDA
if (USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET dl_core PROPERTY CUDA_ARCHITECTURES 86)
    target_sources(dl_core PRIVATE 
        src/gpu/gpu_main_wrapper.cu
        src/gpu/gpu_autoencoder.cu
        src/gpu/utils.cpp
        src/gpu/main.cu
    )
else()
    # Stub implementation khi không có CUDA
    target_sources(dl_core PRIVATE src/gpu/gpu_phase_stub.cpp)
endif()

# Executable
# Root launcher (in-process)
add_executable(main main.cpp)
target_link_libraries(main PRIVATE dl_core)
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# Standalone CPU phase executable (wrapper calls cpu_phase_main)
add_executable(cpu_main src/cpu/cpu_main_wrapper.cpp)
target_link_libraries(cpu_main PRIVATE dl_core)
set_property(TARGET cpu_main PROPERTY CXX_STANDARD 17)

# Standalone GPU phase executable (wrapper calls gpu_phase_main)
if (USE_CUDA)
    add_executable(gpu_main src/gpu/gpu_main_wrapper.cu)
    target_link_libraries(gpu_main PRIVATE dl_core)
    set_property(TARGET gpu_main PROPERTY CXX_STANDARD 17)
    set_property(TARGET gpu_main PROPERTY CUDA_STANDARD 17)
else()
    add_executable(gpu_main src/gpu/gpu_main_wrapper.cpp)
    target_link_libraries(gpu_main PRIVATE dl_core)
    set_property(TARGET gpu_main PROPERTY CXX_STANDARD 17)
endif()
