# Face Detection using Haar Cascade - CMake Build
cmake_minimum_required(VERSION 3.15)
project(FaceDetection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV (only core modules needed, no contrib)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Build executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
