cmake_minimum_required(VERSION 3.18)
project(llamacu LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")

find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development.Module REQUIRED)
message (STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
execute_process(COMMAND ${Python_EXECUTABLE} "-c"
    "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
message (STATUS "PYBIND11_CMAKE_DIR: ${PYBIND11_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

pybind11_add_module(C ${SOURCES} ${CUDA_SOURCES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(C PRIVATE
    "-Wl,-Bsymbolic"
    "-Wl,-Bsymbolic-functions"
    CUDA::cublas
)

target_include_directories(C
    PRIVATE "src" "src/model"
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


