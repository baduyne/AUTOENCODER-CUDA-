cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dl_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bật/tắt GPU phase
option(USE_CUDA "Build with CUDA support" OFF)

# Đường dẫn libtorch (trên WSL).
# Sau này trên Colab bạn có thể override bằng -DCMAKE_PREFIX_PATH=/content/libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")

find_package(Torch REQUIRED)

message(STATUS "Found Torch version: ${TORCH_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# Khuyến nghị từ PyTorch: thêm TORCH_CXX_FLAGS vào CXX_FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Core library
add_library(dl_core
    src/device.cpp
    src/autoencoder.cpp
    src/conv2d_cpu.cpp
    src/cifar10_dataset.cpp
    src/layers_cpu.cpp
)

target_include_directories(dl_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(dl_core PUBLIC "${TORCH_LIBRARIES}")

set_property(TARGET dl_core PROPERTY CXX_STANDARD 17)

# Nếu build với CUDA → thêm file .cu & define USE_CUDA
if (USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Thêm conv2d_cuda.cu vào lib
    target_sources(dl_core PRIVATE src/conv2d_cuda.cu)
endif()

# Executable
add_executable(dl_main src/main.cpp)
target_link_libraries(dl_main PRIVATE dl_core "${TORCH_LIBRARIES}")
set_property(TARGET dl_main PROPERTY CXX_STANDARD 17)
