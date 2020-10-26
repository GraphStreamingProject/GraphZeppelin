cmake_minimum_required(VERSION 3.0)
project(test)

find_package(xxHash 0.7 CONFIG REQUIRED)
#add_executable(main main.cpp)
add_executable(l0sampling l0sampling.cpp)
target_link_libraries(l0sampling PRIVATE xxHash::xxhash)
# set compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -O3 -std=c++11")

find_package(xxHash 0.7 CONFIG REQUIRED)
#add_executable(main main.cpp)
add_executable(l0sampling l0sampling.cpp)
add_executable(tests tests.cpp)
target_link_libraries(l0sampling PRIVATE xxHash::xxhash)
target_link_libraries(tests criterion)
>>>>>>> d1ab12581f7e9708c5507070b367163e2d5765aa
