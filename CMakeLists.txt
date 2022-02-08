cmake_minimum_required(VERSION 3.15)
project(GraphStreamingCC)

include (FetchContent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Make the default build type Release. If user or another
# project sets a different value than use that
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to default -- Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
endif()
message(STATUS "GraphStreamingCC Build Type: ${CMAKE_BUILD_TYPE}")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message(STATUS "Adding GNU compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
elseif(STATUS "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message("Adding MSVC compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  message(STATUS "${CMAKE_CXX_COMPILER_ID} not recognized, no flags added")
endif()

# Get xxHash
FetchContent_Declare(
  xxhash

  GIT_REPOSITORY "https://github.com/Cyan4973/xxHash.git"
  GIT_TAG        "v0.8.0"
)

# Get GutterTree Project
FetchContent_Declare(
  GutterTree

  GIT_REPOSITORY  "https://github.com/GraphStreamingProject/GutterTree.git"
  GIT_TAG         "query_handle"

  UPDATE_DISCONNECTED OFF
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

FetchContent_MakeAvailable(xxHash GutterTree)
#####
# Some additional steps for xxHash as it is unofficial
#####
#xxHash messes with BUILD_SHARED_LIBS if it is empty
set(SAVED_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")

add_subdirectory(
  "${xxhash_SOURCE_DIR}/cmake_unofficial"
  "${xxhash_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
#Restore BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS "${SAVED_BUILD_SHARED_LIBS}" CACHE BOOL "" FORCE)


# AVAILABLE COMPILATION DEFINITIONS:
# VERIFY_SAMPLES_F   Use a deterministic connected-components 
#                    algorithm to verify post-processing.

add_library(GraphStreamingCC
  src/graph.cpp
  src/supernode.cpp
  src/graph_worker.cpp
  src/l0_sampling/sketch.cpp
  src/l0_sampling/update.cpp
  src/util.cpp)
add_dependencies(GraphStreamingCC GutterTree)
target_link_libraries(GraphStreamingCC PUBLIC xxhash GutterTree)
target_include_directories(GraphStreamingCC PUBLIC include/ include/l0_sampling/)
target_compile_options(GraphStreamingCC PUBLIC -fopenmp)
target_link_options(GraphStreamingCC PUBLIC -fopenmp)
target_compile_definitions(GraphStreamingCC PUBLIC XXH_INLINE_ALL)

add_library(GraphStreamingVerifyCC
  src/graph.cpp
  src/supernode.cpp
  src/graph_worker.cpp
  src/l0_sampling/sketch.cpp
  src/l0_sampling/update.cpp
  src/util.cpp
  test/util/file_graph_verifier.cpp
  test/util/mat_graph_verifier.cpp)
add_dependencies(GraphStreamingVerifyCC GutterTree)
target_link_libraries(GraphStreamingVerifyCC PUBLIC xxhash GutterTree)
target_include_directories(GraphStreamingVerifyCC PUBLIC include/ include/l0_sampling/ include/test/)
target_compile_options(GraphStreamingVerifyCC PUBLIC -fopenmp)
target_link_options(GraphStreamingVerifyCC PUBLIC -fopenmp)
target_compile_definitions(GraphStreamingVerifyCC PUBLIC XXH_INLINE_ALL VERIFY_SAMPLES_F)

add_executable(tests
  test/test_runner.cpp
  test/graph_test.cpp
  test/sketch_test.cpp
  test/supernode_test.cpp
  test/util_test.cpp
  test/util/file_graph_verifier.cpp
  test/util/graph_gen.cpp
  test/util/graph_gen_test.cpp
  test/util/graph_verifier_test.cpp)
add_dependencies(tests GraphStreamingVerifyCC)
target_link_libraries(tests PRIVATE GraphStreamingVerifyCC)

add_executable(statistical_test
  test/statistical_testing/graph_testing.cpp
  test/util/file_graph_verifier.cpp
  test/util/graph_gen.cpp)
add_dependencies(statistical_test GraphStreamingVerifyCC)
target_link_libraries(statistical_test PRIVATE GraphStreamingVerifyCC)

# executables for experiment/benchmarking
add_executable(efficient_gen
  test/util/efficient_gen/edge_gen.cpp
  test/util/efficient_gen/efficient_gen.cpp)
target_link_libraries(efficient_gen PRIVATE xxhash GraphZeppelinCommon)
