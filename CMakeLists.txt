cmake_minimum_required(VERSION 3.16)
project(NexusInfinite VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for performance
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /W4 /permissive- /arch:AVX2 /DUSE_AVX2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter -fconstexpr-ops-limit=1000000000")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -flto=auto -DNDEBUG -DUSE_AVX2 -DUSE_BMI2 -funroll-loops -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address,undefined")
    
    # Profile-Guided Optimization support
    set(CMAKE_CXX_FLAGS_PGO "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate")
    set(CMAKE_EXE_LINKER_FLAGS_PGO "-fprofile-generate")
endif()

# Enable specific instruction sets if available
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
CHECK_CXX_COMPILER_FLAG("-mbmi2" COMPILER_SUPPORTS_BMI2)
CHECK_CXX_COMPILER_FLAG("-mpopcnt" COMPILER_SUPPORTS_POPCNT)

if(COMPILER_SUPPORTS_AVX2)
    add_compile_definitions(USE_AVX2)
endif()
if(COMPILER_SUPPORTS_BMI2)
    add_compile_definitions(USE_BMI2)
endif()
if(COMPILER_SUPPORTS_POPCNT)
    add_compile_definitions(USE_POPCNT)
endif()

# Source files
set(SOURCES
    src/bitboard.cpp
    src/board.cpp
    src/move.cpp
    src/movegen.cpp
    src/eval.cpp
    src/tt.cpp
    src/search.cpp
    src/timeman.cpp
    src/uci.cpp
    src/see.cpp
    src/reductions.cpp
    src/history.cpp
    src/simd.cpp
    src/nnue.cpp
    src/syzygy.cpp
    src/thread.cpp
    src/test_suite.cpp
    src/training.cpp
    src/datapipeline.cpp
    src/main.cpp
)

# Engine executable
add_executable(nexus ${SOURCES})

target_include_directories(nexus PRIVATE src)

if(NOT MSVC)
    target_link_libraries(nexus PRIVATE pthread)
endif()

# Tests
enable_testing()
add_executable(nexus_tests tests/test_bitboard.cpp tests/test_perft.cpp 
    src/bitboard.cpp
    src/board.cpp
    src/move.cpp
    src/movegen.cpp
    src/eval.cpp
    src/tt.cpp
    src/see.cpp
    src/reductions.cpp
    src/history.cpp
    src/simd.cpp
    src/nnue.cpp
    src/syzygy.cpp
    src/thread.cpp
    src/test_suite.cpp
    src/training.cpp
    src/datapipeline.cpp)
target_include_directories(nexus_tests PRIVATE src)
if(NOT MSVC)
    target_link_libraries(nexus_tests PRIVATE pthread)
endif()
add_test(NAME PerftSuite COMMAND nexus_tests)
