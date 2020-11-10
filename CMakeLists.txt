cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(test)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message("Adding GNU compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Adding MSVC compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /O2")
else()
  message("${CMAKE_CXX_COMPILER_ID} not recognized, no flags added")
endif()

include(FindOrDownloadPackage.cmake)

find_or_download_package(
  PKG xxHash
  VERSION 0.7
  SOURCE_DIR cmake_unofficial
  TARGETS xxhash
  DOWNLOAD
    GIT_REPOSITORY "https://github.com/Cyan4973/xxHash.git"
    GIT_TAG "v0.8.0"
)

#As specified in the googletest docs, need to set
#gtest_force_shared_crt for windows visual studio.
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

find_or_download_package(
  PKG GTest
  TARGETS gtest gtest_main gmock gmock_main
  DOWNLOAD
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG "release-1.10.0"
)

#add_executable(main main.cpp)
add_executable(l0sampling
  l0sampling.cpp
  l0_sampling/sketch.cpp
  l0_sampling/update.cpp
)
target_link_libraries(l0sampling PRIVATE xxHash::xxhash)

add_executable(tests
  test/test_runner.cpp
  test/sketch_test.cpp
  l0_sampling/sketch.cpp
  l0_sampling/update.cpp
)
target_link_libraries(tests PRIVATE xxHash::xxhash GTest::gtest)
