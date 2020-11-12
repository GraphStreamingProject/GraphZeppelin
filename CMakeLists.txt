cmake_minimum_required(VERSION 3.0)
project(test)

find_package(xxHash 0.7 CONFIG REQUIRED)

# set compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3 -std=c++11 -g")

add_executable(main 
    main.cpp
    graph.cpp
    supernode.cpp
    l0_sampling/sketch.cpp
    l0_sampling/update.cpp
    )
target_link_libraries(main PRIVATE xxHash::xxhash)


# set compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3 -std=c++11")

add_executable(l0sampling
    l0sampling.cpp
    l0_sampling/sketch.cpp
    l0_sampling/update.cpp
    )
target_link_libraries(l0sampling PRIVATE xxHash::xxhash)


find_package(xxHash 0.7 CONFIG REQUIRED)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(tests
    test/test_runner.cpp
    test/sketch_test.cpp
    l0_sampling/sketch.cpp
    l0_sampling/update.cpp
    )
target_link_libraries(tests xxHash::xxhash ${GTEST_LIBRARIES} pthread)
