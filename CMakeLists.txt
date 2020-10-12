cmake_minimum_required(VERSION 3.0)
project(test)

find_package(xxHash 0.7 CONFIG REQUIRED)
#add_executable(main main.cpp)
add_executable(l0sampling l0sampling.cpp)
target_link_libraries(l0sampling PRIVATE xxHash::xxhash)
