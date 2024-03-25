# GraphZeppelin: A C++ Library for Solving the Connected Components Problem on Large, Dense Graph Streams
This is the source code of GraphZeppelin: a compact, fast, and scalable graph processing system. Graph Zeppelin is described in detail in our paper published in [SIGMOD2022](https://dl.acm.org/doi/10.1145/3514221.3526146).

The full experiments for our SIGMOD paper can be found in our [Experiments Repository](https://github.com/GraphStreamingProject/ZeppelinExperiments). Our experiments were replicated by the SIGMOD reproducibility committee, details can be found in the [reproducibility report](https://reproducibility.sigmod.org/rep_rep/2023/Dayan-SIGMODReproReport26.pdf).

Since submitting to SIGMOD, GraphZeppelin has been continually updated improve robustness, performance, and reduce memory consumption.

## Installing and Running GraphZeppelin
### Requirements
- Unix OS (not Mac, tested on Ubuntu)
- cmake>=3.15

### Installation
1. Clone this repository
2. Create a `build` sub directory at the project root dir.
3. Initialize cmake by running `cmake ..` in the build dir.
4. Build the library and executables for testing by running `cmake --build .` in the build dir.

This library can easily be included with other cmake projects using FetchContent or ExternalProject.

## Minimal Example
```
#include <binary_file_stream.h>
#include <cc_sketch_alg.h>
#include <graph_sketch_driver.h>
#include <time.h>

std::string file_name = "/path/to/binary/stream";

int main() {
  BinaryFileStream stream(file_name);           // Create a stream object for parsing a graph stream 'file_name'
  node_id_t num_vertices = stream.vertices();   // Extract the number of graph vertices from the stream
  CCSketchAlg cc_alg{                           // Create connected components sketch algorithm
    num_vertices,                                  // vertices in graph
    size_t(time(NULL)),                            // seed
    CCAlgConfiguration()                           // configuration
  }; 
  GraphSketchDriver<CCSketchAlg> driver{        // Create a driver to manage the CC algorithm
    &cc_alg,                                       // algorithm to update
    &stream,                                       // stream to read
    DriverConfiguration()                          // configuration
  };
  driver.process_stream_until(END_OF_STREAM);   // Tell the driver to process the entire graph stream
  driver.prep_query(CONNECTIVITY);              // Ensure algorithm is ready for a connectivity query
  auto CC = cc_alg.connected_components();      // Extract the connected components
}
```
A more detailed example can be found in `tools/process_stream.cpp`.

## Configuration
GraphZeppelin has a number of parameters both for the driver and the sketch algorithm. Examples of these parameters include the number of threads and which GutteringSystem to run for the driver and the desired batch size for the algorithm.
To achieve high performance, it is important to set these parameters correctly. See `tools/process_stream.cpp`.

The driver options are set with the `DriverConfiguration` object (see `include/driver_configuration.h`).
The algorithm configuration is allowed to vary by algorithm. The connected components algorithm options is managed with the `CCAlgConfiguration` object (see `include/cc_alg_configuration.h`).

## Binary Stream Format
GraphZeppelin uses a binary stream format for efficient file parsing. The format of these files is as follows.
```
<num_nodes> <num_updates> <edge_update>  ...  <edge_update>
| 4 bytes  |   8 bytes   |   9 bytes   | ... |   9 bytes   |
```
num_nodes defines the number of nodes(vertices) in the graph. num_updates states the total number of edge updates (either insertions or deletions) in the stream.

Each edge_update has the following format:
```
<UpdateType> <src_node> <dst_node>
|  1 byte   | 4 bytes  | 4 bytes  |
```
Where UpdateType is 0 to indicate an insertion and 1 to indicate a deletion.

See our [StreamingUtilities](https://github.com/GraphStreamingProject/StreamingUtilities) repository for more details.

## GutteringSystems
To achieve high update throughput, GraphZeppelin buffers updates in what we call a GutteringSystem. Choosing the correct GutteringSystem is important for performance. If you expect storage to include on disk data-structures, choose the `GutterTree`. Otherwise, choose the `CacheTree`.

For more details see the [GutteringSystems](https://github.com/GraphStreamingProject/GutterTree) repository.

## Debugging
You can enable the symbol table and turn off compiler optimizations for debugging with tools like `gdb` or `valgrind` by performing the following steps
1. Re-initialize cmake by running `cmake -DCMAKE_BUILD_TYPE=Debug ..` in the build directory
2. Re-build by running `cmake --build .` in the build dir. Symbol table information is now included in the libraries and executables.

To switch back to the optimized version of the code without the symbol table redo the two above steps except specify `Release` as the CMAKE_BUILD_TYPE.
Other build types are available as well, but these should be the only two you need.

## Benchmarking
The `tools/benchmark` directory provides a number of benchmarks that allow for fine tuned performance testing of various parts of the system. These benchmarks are not built by default and require a Linux machine. Some optional benchmarks additionally require root access. More information can be found in the [benchmark documentation](/tools/benchmark/BENCH.md).

To build the benchmarks perform the following steps.
1. Initialize cmake with the BUILD_BENCH flag `cmake -DBUILD_BENCH:BOOL=ON ..` (recommended to use the Release build type)
2. Re-build by running `cmake --build .` in the build dir.

Turn off building the benchmarks by passing `-DBUILD_BENCH:BOOL=OFF` to cmake.

## Other Information
https://github.com/GraphStreamingProject/GraphZeppelin/wiki
