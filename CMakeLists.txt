cmake_minimum_required(VERSION 3.20)
project(MetalIRPipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── LLVM ──────────────────────────────────────────────────────────────────
# Point to Triton's bundled LLVM or system LLVM.
# Usage: cmake -B build -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(${CMAKE_SOURCE_DIR}/include)

# Match LLVM's build flags (no RTTI, no exceptions)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
add_compile_options(-fno-exceptions)

# ── Libraries ─────────────────────────────────────────────────────────────
add_subdirectory(lib/Transforms)
add_subdirectory(lib/Serialization)
add_subdirectory(lib/Bridge)

# ── Tools ─────────────────────────────────────────────────────────────────
add_subdirectory(tools)

# ── Tests ─────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(test)
