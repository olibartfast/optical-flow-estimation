cmake_minimum_required(VERSION 3.12)
project(torchscript-raft-optical-flow)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

set(SOURCES main.cpp)

# Set Torch DIR and find package
set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch/)
message(STATUS "Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find CUDA
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA found: Enabling CUDA support")
    add_compile_definitions(TORCH_CUDA_AVAILABLE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CUDA_FLAGS}")
else()
    message(STATUS "CUDA not found: Building CPU-only version")
endif()

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link against OpenCV and the selected framework libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# Optional: Print build type and flags for debugging
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS}")
if(CUDA_FOUND)
    message(STATUS "CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()