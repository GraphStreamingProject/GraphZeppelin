#TODO: write FindxxHash for libxxhash-dev compatibility?
#Cmake for xxHash is unofficial, so libxxhash-dev doesn't have xxHashConfig.cmake

cmake_minimum_required(VERSION 3.15)
include (ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

#Uncomment to enable debug
#set(ENV{DEBUG} ON)

project(test)

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
    set(BOOST_ROOT "${boost_SOURCE_DIR}" CACHE STRING "" FORCE)
    find_package(Boost 1.66.0)
  endif()
endif()
include_directories("${Boost_INCLUDE_DIR}")

#add_executable(main
#  main.cpp
#  graph.cpp
#  supernode.cpp
#  l0_sampling/sketch.cpp
#  l0_sampling/update.cpp
#  l0_sampling/bucket.cpp
#  util.cpp
#  l0_sampling/int127.cpp)
#target_link_libraries(main PRIVATE xxHash::xxhash)
#if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#  target_compile_options(main PRIVATE -O3)
#elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#  target_compile_options(main PRIVATE /O2)
#endif()

#add_executable(l0sampling
#  l0sampling.cpp
#  l0_sampling/sketch.cpp
#  l0_sampling/update.cpp
#  l0_sampling/bucket.cpp
#  util.cpp
#  l0_sampling/int127.cpp)
#target_link_libraries(l0sampling PRIVATE xxHash::xxhash)
#if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#  target_compile_options(l0sampling PRIVATE -O3)
#elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#  target_compile_options(l0sampling PRIVATE /O2)
#endif()

# Install BufferTree Project
ExternalProject_Add(
  BufferTree

  PREFIX          BufferTree
  GIT_REPOSITORY  "https://github.com/Victor-C-Zhang/FastBufferTree.git"
  GIT_TAG         "main"

  BUILD_ALWAYS    OFF
  INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}/BufferTree/prefix

  CMAKE_CACHE_ARGS
        -DAUTODOWNLOAD:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>

  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target install
)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/BufferTree/prefix/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/BufferTree/prefix/lib)

add_executable(tests
  test/test_runner.cpp
  test/graph_test.cpp
  test/sketch_test.cpp
  test/supernode_test.cpp
  test/util_test.cpp
  test/util/graph_verifier.cpp
  test/util/graph_gen.cpp
  test/util/graph_gen_test.cpp
  test/util/graph_verifier_test.cpp
  graph.cpp
  supernode.cpp
  l0_sampling/sketch.cpp
  util.cpp)
target_link_libraries(tests PRIVATE xxHash::xxhash GTest::gtest)
target_compile_definitions(tests PRIVATE VERIFY_SAMPLES_F)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(tests PRIVATE -g)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(tests PRIVATE -O3)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(tests PRIVATE /O2)
  endif()
endif ()

add_executable(native_tests
  test/test_runner.cpp
  test/graph_test.cpp
  test/sketch_test.cpp
  test/supernode_test.cpp
  test/util_test.cpp
  test/util/graph_verifier.cpp
  test/util/graph_gen.cpp
  test/util/graph_gen_test.cpp
  test/util/graph_verifier_test.cpp
  graph.cpp
  supernode.cpp
  l0_sampling/sketch.cpp
  util.cpp)
target_link_libraries(native_tests PRIVATE xxHash::xxhash GTest::gtest)
target_compile_definitions(native_tests PRIVATE VERIFY_SAMPLES_F USE_NATIVE_F)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(native_tests PRIVATE -g)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(native_tests PRIVATE -O3)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(native_tests PRIVATE /O2)
  endif()
endif ()

add_executable(wods_test
  test/test_runner.cpp
  test/graph_test.cpp
  test/sketch_test.cpp
  test/supernode_test.cpp
  test/util_test.cpp
  test/util/graph_verifier.cpp
  test/util/graph_gen.cpp
  test/util/graph_gen_test.cpp
  test/util/graph_verifier_test.cpp
  graph.cpp
  supernode.cpp
  GraphWorker.cpp
  l0_sampling/sketch.cpp
  l0_sampling/update.cpp
  util.cpp)
target_link_libraries(wods_test PRIVATE xxHash::xxhash GTest::gtest)
target_compile_definitions(wods_test PRIVATE VERIFY_SAMPLES_F USE_NATIVE_F)
# toku dependencies
target_compile_definitions(wods_test PRIVATE -DBUF_PROTOTYPE)
add_dependencies(wods_test BufferTree)
target_link_libraries(wods_test PRIVATE libFastBufferTree.a)

# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(wods_test PRIVATE -g)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(wods_test PRIVATE -O3)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(wods_test PRIVATE /O2)
  endif()
endif ()

# executables for experiment/benchmarking
add_executable(efficient_gen
  test/util/efficient_gen/edge_gen.cpp
  test/util/efficient_gen/cum_gen.cpp
  test/util/efficient_gen/efficient_gen.cpp)
target_link_libraries(efficient_gen PRIVATE xxHash::xxhash)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(efficient_gen PRIVATE -O3)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(efficient_gen PRIVATE /O2)
endif()

add_executable(experiment
  test/test_runner.cpp
  test/experiment/BCHMK_2_toku.cpp
  test/util/graph_verifier.cpp
  test/util/graph_gen.cpp
  graph.cpp
  supernode.cpp
  l0_sampling/sketch.cpp
  l0_sampling/update.cpp
  util.cpp)
target_link_libraries(experiment PRIVATE xxHash::xxhash GTest::gtest)
target_compile_definitions(experiment PRIVATE VERIFY_SAMPLES_F USE_NATIVE_F)
# optimize
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(tests PRIVATE -O3)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  target_compile_options(tests PRIVATE /O2)
endif()

add_executable(buffertree_experiment
  test/test_runner.cpp
  test/experiment/BCHMK_2_toku.cpp
  test/util/graph_verifier.cpp
  test/util/graph_gen.cpp
  graph.cpp
  supernode.cpp
  GraphWorker.cpp
  l0_sampling/sketch.cpp
  l0_sampling/update.cpp
  util.cpp)
target_link_libraries(buffertree_experiment PRIVATE xxHash::xxhash GTest::gtest)
target_compile_definitions(buffertree_experiment PRIVATE VERIFY_SAMPLES_F USE_NATIVE_F)
# dependencies
target_compile_definitions(buffertree_experiment PRIVATE -DBUF_PROTOTYPE)
add_dependencies(buffertree_experiment BufferTree)
target_link_libraries(buffertree_experiment PRIVATE libFastBufferTree.a)
# optimize unless debug
if (DEFINED ENV{DEBUG})
  message("Disabling optimizations and enabling debug symbols")
  target_compile_options(buffertree_experiment PRIVATE -g)
else ()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(buffertree_experiment PRIVATE -O3)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(buffertree_experiment PRIVATE /O2)
  endif()
endif ()
