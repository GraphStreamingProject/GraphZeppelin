cmake_minimum_required(VERSION 3.0)
project(test)

find_package(xxHash 0.7 CONFIG REQUIRED)
#add_executable(main main.cpp)
add_executable(l0sampling l0sampling.cpp sketch.cpp update.cpp)
target_link_libraries(l0sampling PRIVATE xxHash::xxhash)
# set compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3 -std=c++11")

find_package(xxHash 0.7 CONFIG REQUIRED)

add_executable(tests
    tests.cpp
    test/sketch_test.cpp
    sketch.cpp
    update.cpp
    )
target_link_libraries(tests PRIVATE xxHash::xxhash criterion)
