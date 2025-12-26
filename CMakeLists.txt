cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dl_pipeline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

# Enable/disable GPU phase
option(USE_CUDA "Build with CUDA support" OFF)

# =============================================================================
# BLAS library (required by liblinear)
# =============================================================================
add_library(blas STATIC
    libs/liblinear/blas/daxpy.c
    libs/liblinear/blas/ddot.c
    libs/liblinear/blas/dnrm2.c
    libs/liblinear/blas/dscal.c
)
target_include_directories(blas PUBLIC ${CMAKE_SOURCE_DIR}/libs/liblinear/blas)
set_property(TARGET blas PROPERTY LINKER_LANGUAGE C)

# =============================================================================
# LIBLINEAR library (replaces libsvm - much faster for linear SVM)
# =============================================================================
add_library(liblinear STATIC
    libs/liblinear/linear.cpp
    libs/liblinear/newton.cpp
)

target_include_directories(liblinear PUBLIC
    ${CMAKE_SOURCE_DIR}/libs/liblinear
    ${CMAKE_SOURCE_DIR}/libs/liblinear/blas
)

target_link_libraries(liblinear PUBLIC blas)
set_property(TARGET liblinear PROPERTY CXX_STANDARD 17)

# =============================================================================
# Core library - Autoencoder and dataset handling
# =============================================================================
add_library(dl_core
    src/cpu/src/autoencoder_cpu.cpp
    src/cpu/src/conv2d_cpu.cpp
    src/cpu/src/cifar10_dataset.cpp
    src/cpu/src/layers_cpu.cpp
    src/cpu/src/training_pipeline_cpu.cpp
    src/cpu/src/main.cpp
)

target_include_directories(dl_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cpu/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/gpu/include
    ${CMAKE_SOURCE_DIR}/libs/liblinear
)

target_link_libraries(dl_core PUBLIC)

set_property(TARGET dl_core PROPERTY CXX_STANDARD 17)

# If building with CUDA, add .cu files
if (USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET dl_core PROPERTY CUDA_ARCHITECTURES 89)
    target_sources(dl_core PRIVATE 
        src/gpu/src/gpu_main_wrapper.cu
        src/gpu/src/gpu_autoencoder.cu
        src/gpu/src/utils.cpp
        src/gpu/src/main.cu
    )
else()
    # Stub implementation when CUDA is not available
    target_sources(dl_core PRIVATE src/gpu/gpu_phase_stub.cpp)
endif()

# =============================================================================
# Executables
# =============================================================================

# Root launcher (in-process)
add_executable(main main.cpp)
target_link_libraries(main PRIVATE dl_core)
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# Standalone CPU phase executable
add_executable(cpu_main src/cpu/cpu_main_wrapper.cpp)
target_link_libraries(cpu_main PRIVATE dl_core)
set_property(TARGET cpu_main PROPERTY CXX_STANDARD 17)

# Standalone GPU phase executable
if (USE_CUDA)
    add_executable(gpu_main src/gpu/src/gpu_main_wrapper.cu)
    target_link_libraries(gpu_main PRIVATE dl_core)
    set_property(TARGET gpu_main PROPERTY CXX_STANDARD 17)
    set_property(TARGET gpu_main PROPERTY CUDA_STANDARD 17)
else()
    add_executable(gpu_main src/gpu/src/gpu_main_wrapper.cpp)
    target_link_libraries(gpu_main PRIVATE dl_core)
    set_property(TARGET gpu_main PROPERTY CXX_STANDARD 17)
endif()

# =============================================================================
# SVM Phase executable (uses liblinear, works with both CPU and GPU features)
# =============================================================================
add_executable(svm_main
    svm_main.cpp
    svm/linear_svm.cpp
)

target_include_directories(svm_main PRIVATE
    ${CMAKE_SOURCE_DIR}/svm
    ${CMAKE_SOURCE_DIR}/libs/liblinear
)

target_link_libraries(svm_main PRIVATE liblinear)
set_property(TARGET svm_main PROPERTY CXX_STANDARD 17)