cmake_minimum_required(VERSION 3.12)
project(torchscript-raft-optical-flow)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

set(SOURCES main.cpp)

set(Torch_DIR $ENV{HOME}/libtorch/share/cmake/Torch/)
message(STATUS "${Torch_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link against OpenCV and the selected framework libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}  ${TORCH_LIBRARIES}
)
