#TODO: write FindxxHash for libxxhash-dev compatibility?
#Cmake for xxHash is unofficial, so libxxhash-dev doesn't have xxHashConfig.cmake
cmake_minimum_required(VERSION 3.15)
project(GraphStreamingCC)

include (ExternalProject)

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

#Find or download GTest
find_package(GTest CONFIG)
if(NOT GTest_FOUND)
  if(NOT AUTODOWNLOAD)
    message(FATAL_ERROR "GTest config not found.\n"
      "Set GTest_DIR to a directory containing one of \n"
      "  GTestConfig.cmake\n"
      "  gtest-config.cmake\n"
      "or use -DAUTODOWNLOAD:BOOL=ON to automatically download")
  else()
    #Download googletest
    download_project(
      PROJ googletest
      GIT_REPOSITORY "https://github.com/google/googletest.git"
      GIT_TAG "release-1.10.0"
    )

    #As specified in the googletest docs, need to set
    #gtest_force_shared_crt for windows visual studio.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()

    add_subdirectory(
      "${googletest_SOURCE_DIR}"
      "${googletest_BINARY_DIR}"
      EXCLUDE_FROM_ALL
    )

    #Add aliases to targets
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gmock_main ALIAS gmock_main)
  endif()
endif()

#Find or download boost
find_package(Boost 1.66.0)
if(NOT Boost_FOUND)
  if(NOT AUTODOWNLOAD)
    message(FATAL_ERROR "Boost not found.\n"
      "Set BOOST_ROOT to a directory where Boost is installed\n"
      "or use -DAUTODOWNLOAD:BOOL=ON to automatically download")
  else()
    message("Downloading boost")
    #Download boost
    download_project(
      PROJ boost
      GIT_REPOSITORY "https://github.com/boostorg/boost.git"
      GIT_TAG "boost-1.66.0"
    )

    #Run bootstrap according to OS version
    if (WIN32)
      set(boost_bootstrap_script ".\\bootstrap.bat")
    else()
      set(boost_bootstrap_script "./bootstrap.sh")
    endif()
    execute_process(
      COMMAND "${boost_bootstrap_script}"
      RESULT_VARIABLE result
      WORKING_DIRECTORY "${boost_SOURCE_DIR}"
    )
    if(result)
      message(FATAL_ERROR "Failed to run bootstrap script")
    endif()

    #Generate header files with b2
    execute_process(
      COMMAND "./b2" "headers"
      RESULT_VARIABLE result
      WORKING_DIRECTORY "${boost_SOURCE_DIR}"
    )
    if(result)
      message(FATAL_ERROR "Failed to run b2 headers")
    endif()

    #Set BOOST_ROOT in cache to what we just downloaded, FindBoost again
    set(ENV{BOOST_LIBRARYDIR} ${boost_SOURCE_DIR}/../boost-install/lib)
    set(BOOST_ROOT "${boost_SOURCE_DIR}" CACHE STRING "" FORCE)
    find_package(Boost 1.66.0 REQUIRED)
  endif()
endif()
include_directories("${Boost_INCLUDE_DIR}")

# Include common components
ExternalProject_Add(
    GraphZeppelinCommon

    PREFIX          GraphZeppelinCommon
    GIT_REPOSITORY  "https://github.com/GraphStreamingProject/GraphZeppelinCommon.git"
    GIT_TAG         "main"

    UPDATE_DISCONNECTED false
    BUILD_ALWAYS    OFF
    INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}/GraphZeppelinCommon/prefix

    CMAKE_CACHE_ARGS
    -DAUTODOWNLOAD:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>

    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/GraphZeppelinCommon/prefix/include)

# Install GutterTree Project
ExternalProject_Add(
  GutterTree

  PREFIX          GutterTree
  GIT_REPOSITORY  "https://github.com/GraphStreamingProject/GutterTree.git"
  GIT_TAG         "commons"

  UPDATE_DISCONNECTED false
  BUILD_ALWAYS    OFF
  INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}/GutterTree/prefix

  CMAKE_CACHE_ARGS
        -DAUTODOWNLOAD:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>

  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)
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
target_link_libraries(GraphStreamingCC PUBLIC xxHash::xxhash GTest::gtest)
add_dependencies(GraphStreamingCC GutterTree GraphZeppelinCommon)
target_link_libraries(GraphStreamingCC PUBLIC libGutterTree.a)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(GraphStreamingCC PRIVATE -g -fopenmp)
  target_link_options(GraphStreamingCC PRIVATE -fopenmp)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(GraphStreamingCC PRIVATE -O3 -fopenmp)
    target_link_options(GraphStreamingCC PRIVATE -fopenmp)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(GraphStreamingCC PRIVATE /O2)
  endif()
endif()

add_library(GraphStreamingVerifyCC
  src/graph.cpp
  src/supernode.cpp
  src/graph_worker.cpp
  src/l0_sampling/sketch.cpp
  src/l0_sampling/update.cpp
  src/util.cpp
  test/util/file_graph_verifier.cpp
  test/util/mat_graph_verifier.cpp)
target_link_libraries(GraphStreamingVerifyCC PUBLIC xxHash::xxhash GTest::gtest)
target_compile_definitions(GraphStreamingVerifyCC PRIVATE VERIFY_SAMPLES_F)
add_dependencies(GraphStreamingVerifyCC GutterTree GraphZeppelinCommon)
target_link_libraries(GraphStreamingVerifyCC PUBLIC libGutterTree.a)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(GraphStreamingVerifyCC PRIVATE -g -fopenmp)
  target_link_options(GraphStreamingVerifyCC PRIVATE -fopenmp)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(GraphStreamingVerifyCC PRIVATE -O3 -fopenmp)
    target_link_options(GraphStreamingVerifyCC PRIVATE -fopenmp)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(GraphStreamingVerifyCC PRIVATE /O2)
  endif()
endif()

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
target_link_libraries(tests PRIVATE xxHash::xxhash GTest::gtest GraphStreamingVerifyCC)
target_compile_definitions(tests PRIVATE VERIFY_SAMPLES_F)
# dependencies
add_dependencies(tests GutterTree GraphZeppelinCommon)
target_link_libraries(tests PRIVATE libGutterTree.a)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(tests PRIVATE -g -fopenmp)
  target_link_options(tests PRIVATE -fopenmp)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(tests PRIVATE -O3 -fopenmp)
    target_link_options(tests PRIVATE -fopenmp)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(tests PRIVATE /O2)
  endif()
endif ()

add_executable(statistical_test
  test/statistical_testing/graph_testing.cpp
  test/util/file_graph_verifier.cpp
  test/util/graph_gen.cpp)
target_link_libraries(statistical_test PRIVATE xxHash::xxhash GTest::gtest GraphStreamingVerifyCC)
target_compile_definitions(statistical_test PRIVATE VERIFY_SAMPLES_F)
# dependencies
add_dependencies(statistical_test GutterTree GraphZeppelinCommon)
target_link_libraries(statistical_test PRIVATE libGutterTree.a)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(statistical_test PRIVATE -g -fopenmp)
  target_link_options(statistical_test PRIVATE -fopenmp)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(statistical_test PRIVATE -O3 -fopenmp)
    target_link_options(statistical_test PRIVATE -fopenmp)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(statistical_test PRIVATE /O2)
  endif()
endif ()

# executables for experiment/benchmarking
add_executable(efficient_gen
  test/util/efficient_gen/edge_gen.cpp
  test/util/efficient_gen/efficient_gen.cpp)
target_link_libraries(efficient_gen PRIVATE xxHash::xxhash)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(efficient_gen PRIVATE -O3)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(efficient_gen PRIVATE /O2)
endif()

add_executable(cont_test
  test/test_runner.cpp
  test/cont_test.cpp
  test/util/mat_graph_verifier.cpp
  test/util/graph_gen.cpp)
target_link_libraries(cont_test PRIVATE xxHash::xxhash GTest::gtest GraphStreamingVerifyCC)
target_compile_definitions(cont_test PRIVATE VERIFY_SAMPLES_F)
# dependencies
add_dependencies(cont_test GutterTree GraphZeppelinCommon)
target_link_libraries(cont_test PRIVATE libGutterTree.a)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(cont_test PRIVATE -g -fopenmp)
  target_link_options(cont_test PRIVATE -fopenmp)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(cont_test PRIVATE -O3 -fopenmp)
    target_link_options(cont_test PRIVATE -fopenmp)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(cont_test PRIVATE /O2)
  endif()
endif ()

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
