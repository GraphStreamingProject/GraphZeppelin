# C++ Library for Solving the Connected Components Problem on a Graph Stream

## Installation
1. Clone this repository
2. Create a `build` sub directory at the project root dir.
3. Initialize cmake by running `cmake ..` in the build dir.
4. Build the library and executables for testing by running `cmake --build .` in the build dir.

This library can easily be included with other cmake projects using FetchContent or ExternalProject.

## Configuration
GraphStreamingCC has a few parameters set via a configuration file. These include the number of cpu threads to use, and which datastructure to buffer updates in. The file `example_streaming.conf` gives an example of one such configuration and provides explanations of the various parameters.

To define your own configuration, copy `example_streaming.conf` into the `build` directory as `streaming.conf`. If using GraphStreamingCC as an external library the process for defining the configuration is the same. Once you make changes to the configuration, you should see them reflected in the configuration displayed at the beginning of the program.

## Debugging
You can enable the symbol table and turn off compiler optimizations for debugging with tools like `gdb` or `valgrind` by performing the following steps
1. Re-initialize cmake by running `cmake -DCMAKE_BUILD_TYPE=Debug ..` in the build directory
2. Re-build by running `cmake --build .` in the build dir. Symbol table information is now included in the libraries and executables.

To switch back to the optimized version of the code without the symbol table redo the two above steps except specify `Release` as the CMAKE_BUILD_TYPE.
Other build types are available as well, but these should be the only two you need.

## Other Information
https://github.com/GraphStreamingProject/GraphStreamingCC/wiki
