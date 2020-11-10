cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(test)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message("Adding GNU flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Adding MSVC flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /O2")
else()
  message("${CMAKE_CXX_COMPILER_ID} not recognized, no flags added")
endif()

find_package(xxHash 0.7 CONFIG QUIET)
if (TARGET xxHash::xxhash)
  message("Found installation of xxHash, using it")
else ()
  message("No installation of xxHash found, automatically downloading")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/xxHash-download/CMakeLists.txt
    "cmake_minimum_required(VERSION 2.8.11)\n"
    "project(xxHash-download NONE)\n"
    "include(ExternalProject)\n"
    "ExternalProject_Add(xxHash\n"
    "  GIT_REPOSITORY    https://github.com/Cyan4973/xxHash.git\n"
    "  GIT_TAG           v0.8.0\n"
    "  SOURCE_DIR        \"${CMAKE_CURRENT_BINARY_DIR}/xxHash-src\"\n"
    "  BINARY_DIR        \"${CMAKE_CURRENT_BINARY_DIR}/xxHash-build\"\n"
    "  CONFIGURE_COMMAND \"\"\n"
    "  BUILD_COMMAND     \"\"\n"
    "  INSTALL_COMMAND   \"\"\n"
    "  TEST_COMMAND      \"\"\n"
    ")"
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xxHash-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for xxHash failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xxHash-download
  )
  if(result)
    message(FATAL_ERROR "Build step for xxHash failed: ${result}")
  endif()

  # Add xxHash directly to our build. This defines the xxhash target.
  add_subdirectory(
    ${CMAKE_CURRENT_BINARY_DIR}/xxHash-src/cmake_unofficial
    ${CMAKE_CURRENT_BINARY_DIR}/xxHash-build
    EXCLUDE_FROM_ALL
  )
  #Add aliases to target
  add_library(xxHash::xxhash ALIAS xxhash)
endif()

find_package(GTest CONFIG QUIET)
if (TARGET GTest::gtest_main)
  message("Found installation of GTest, using it")
else ()
  message("No installation of GTest found, automatically downloading")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/googletest-download/CMakeLists.txt
    "cmake_minimum_required(VERSION 2.8.11)\n"
    "project(googletest-download NONE)\n"
    "include(ExternalProject)\n"
    "ExternalProject_Add(googletest\n"
    "  GIT_REPOSITORY    https://github.com/google/googletest.git\n"
    "  GIT_TAG           release-1.10.0\n"
    "  SOURCE_DIR        \"${CMAKE_CURRENT_BINARY_DIR}/googletest-src\"\n"
    "  BINARY_DIR        \"${CMAKE_CURRENT_BINARY_DIR}/googletest-build\"\n"
    "  CONFIGURE_COMMAND \"\"\n"
    "  BUILD_COMMAND     \"\"\n"
    "  INSTALL_COMMAND   \"\"\n"
    "  TEST_COMMAND      \"\"\n"
    ")"
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  # Prevent overriding the parent project's compiler/linker
  # settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # Add googletest directly to our build. This defines the gtest and gtest_main targets.
  add_subdirectory(
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-src
    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL
  )
  #Add aliases to targets
  add_library(GTest::gtest ALIAS gtest)
  add_library(GTest::gtest_main ALIAS gtest_main)
  add_library(GTest::gmock ALIAS gmock)
  add_library(GTest::gmock_main ALIAS gmock_main)
endif()

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
target_link_libraries(tests xxHash::xxhash GTest::gtest pthread)
