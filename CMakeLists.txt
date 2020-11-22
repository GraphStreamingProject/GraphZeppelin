#TODO: write FindxxHash for libxxhash-dev compatibility?
#Cmake for xxHash is unofficial, so libxxhash-dev doesn't have xxHashConfig.cmake

cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(test)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message("Adding GNU compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message("Adding MSVC compiler flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /O2")
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