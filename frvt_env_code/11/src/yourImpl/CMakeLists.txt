cmake_minimum_required(VERSION 3.10)
project(essiImpl)

# ---------------------------------------------------------------------------
# EDIT THIS ONE LINE to match where you unpacked ONNX Runtime
# ---------------------------------------------------------------------------
set(ORT_DIR /home/ubuntu/onnxruntime-linux-x64-1.20.1)
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -Wall")

# NIST's API headers
include_directories(${PROJECT_SOURCE_DIR}/../include)
include_directories(${PROJECT_SOURCE_DIR}/../../../common/src/include)

# ONNX Runtime headers
include_directories(${ORT_DIR}/include)

add_library(frvt_11_essi_000 SHARED
    essiimplfrvt11.cpp
    face_engine.cpp)

target_link_libraries(frvt_11_essi_000
    ${ORT_DIR}/lib/libonnxruntime.so)

# $ORIGIN makes the loader look for libonnxruntime.so in the SAME directory
# as our own .so at runtime. Without this, NIST's machine cannot find it.
set_target_properties(frvt_11_essi_000 PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../../lib)
