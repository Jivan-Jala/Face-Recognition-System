# =============================================================================
# Face Recognition System - CMake Build Configuration
# =============================================================================
# Minimum CMake version required
cmake_minimum_required(VERSION 3.15)

# Project name and language
project(FaceRecognitionSystem LANGUAGES CXX)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Find OpenCV package (must be installed on the system)
# =============================================================================
# On Windows, you may need to set OpenCV_DIR:
#   cmake -DOpenCV_DIR="C:/opencv/build" ..
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect face)

# Print OpenCV version for verification
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# =============================================================================
# Source files
# =============================================================================
set(SOURCES
    src/main.cpp
    src/FaceDetector.cpp
    src/FaceRecognizer.cpp
)

set(HEADERS
    src/FaceDetector.h
    src/FaceRecognizer.h
)

# =============================================================================
# Build the executable
# =============================================================================
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# =============================================================================
# Create output directories at build time
# =============================================================================
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/dataset"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/saved_faces"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/models"
    COMMENT "Creating data directories..."
)

# =============================================================================
# Copy Haar Cascade file to build directory (convenience)
# =============================================================================
# The cascade file ships with OpenCV; we reference it from OpenCV's data dir.
# If OpenCV_DIR is set, cascades are typically at:
#   ${OpenCV_DIR}/etc/haarcascades/haarcascade_frontalface_default.xml
