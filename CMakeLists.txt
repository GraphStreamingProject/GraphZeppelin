#TODO: write FindxxHash for libxxhash-dev compatibility?
#Cmake for xxHash is unofficial, so libxxhash-dev doesn't have xxHashConfig.cmake
cmake_minimum_required(VERSION 3.15)
project(GraphStreamingCC)

include (ExternalProject)
include (FetchContent)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

#Uncomment to enable debug
#set(ENV{DEBUG} ON)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message("Adding GNU compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message("Adding MSVC compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
else()
  message("${CMAKE_CXX_COMPILER_ID} not recognized, no flags added")
endif()

message("${CMAKE_MODULE_PATH}")
include("DownloadProject.cmake")

#Find or download xxHash
find_package(xxHash 0.7 CONFIG)
if(NOT xxHash_FOUND)
  if(NOT AUTODOWNLOAD)
    message(FATAL_ERROR "xxHash config not found.\n"
      "Set xxHash_DIR to a directory containing one of \n"
      "  xxHashConfig.cmake\n"
      "  xxhash-config.cmake\n"
      "or use -DAUTODOWNLOAD:BOOL=ON to automatically download")
  else()
    #Download xxHash
    download_project(
      PROJ xxHash
      GIT_REPOSITORY "https://github.com/Cyan4973/xxHash.git"
      GIT_TAG "v0.8.0"
    )

    #xxHash messes with BUILD_SHARED_LIBS if it is empty
    set(SAVED_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")

    add_subdirectory(
      "${xxHash_SOURCE_DIR}/cmake_unofficial"
      "${xxHash_BINARY_DIR}"
      EXCLUDE_FROM_ALL
    )

    #Restore BUILD_SHARED_LIBS
    set(BUILD_SHARED_LIBS "${SAVED_BUILD_SHARED_LIBS}" CACHE BOOL "" FORCE)

    #Add aliases to targets
    add_library(xxHash::xxhash ALIAS xxhash)
  endif()
endif()

# GutterTree Project
FetchContent_Declare(
  GutterTree

  GIT_REPOSITORY  "https://github.com/GraphStreamingProject/GutterTree.git"
  GIT_TAG         "cmake_updates"
)

FetchContent_MakeAvailable(GutterTree)

# include the relevant h files from the other projects
include_directories(${CMAKE_CURRENT_BINARY_DIR}/GraphZeppelinCommon/prefix/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/GutterTree/prefix/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/GutterTree/prefix/lib)


# AVAILABLE COMPILATION DEFINITIONS:
# VERIFY_SAMPLES_F   Use a deterministic connected-components algorithm to
#                    verify post-processing.

add_library(GraphStreamingCC
  src/graph.cpp
  src/supernode.cpp
  src/graph_worker.cpp
  src/l0_sampling/sketch.cpp
  src/l0_sampling/update.cpp
  src/util.cpp)
add_dependencies(GraphStreamingCC GutterTree)
target_link_libraries(GraphStreamingCC PUBLIC xxHash::xxhash GutterTree)
target_compile_options(GraphStreamingCC PUBLIC -fopenmp)
target_link_options(GraphStreamingCC PUBLIC -fopenmp)

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
target_link_libraries(GraphStreamingVerifyCC PUBLIC xxHash::xxhash GutterTree)
target_compile_options(GraphStreamingVerifyCC PUBLIC -fopenmp)
target_link_options(GraphStreamingVerifyCC PUBLIC -fopenmp)
target_compile_definitions(GraphStreamingVerifyCC PUBLIC VERIFY_SAMPLES_F)

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
target_link_libraries(efficient_gen PRIVATE xxHash::xxhash GraphZeppelinCommon)

#uncomment if manually installing project
#without specifying INSTALL_PREFIX elsewhere
#set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/GraphStreamingCC/prefix)

# for use when building as an external project
# The Directory commands are used to copy over our headers and the GutterTree headers
install(TARGETS GraphStreamingCC GraphStreamingVerifyCC
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/GutterTree/prefix/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GutterTree/
  FILES_MATCHING PATTERN "*.h"
)
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/GraphZeppelinCommon/prefix/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GraphZeppelinCommon/
  FILES_MATCHING PATTERN "*.h"
)
