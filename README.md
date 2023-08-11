# GraphZeppelin: A C++ Library for Solving the Connected Components Problem on Large, Dense Graph Streams
This is the source code of GraphZeppelin: a compact, fast, and scalable graph processing system. Graph Zeppelin is described in detail in our paper published in [SIGMOD2022](https://dl.acm.org/doi/10.1145/3514221.3526146).

The full experiments for our SIGMOD paper can be found in our [Experiments Repository](https://github.com/GraphStreamingProject/ZeppelinExperiments).

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

### Basic Example
```
#include <graph.h>
#include <binary_graph_stream.h>

std::string file_name = "/path/to/binary/stream";

int main() {
  BinaryGraphStream stream(file_name, 1024*32);  // Create a stream object for parsing a stream 'file_name' with 32 KiB buffer
  node_id_t num_nodes   = stream.nodes();        // Extract the number of nodes from the stream 
  size_t    num_updates = stream.edges();        // Extract the number of edge updates from the stream
  Graph g{num_nodes};                            // Create a empty graph with 'num_nodes' nodes

  for (size_t e = 0; e < num_updates; e++)       // Loop through all the updates in the stream
    g.update(stream.get_edge());                 // Update the graph by applying the next edge update

  auto CC = g.connected_components();            // Extract the connected components in the graph defined by the stream
}
```
A more detailed example can be found in `tools/process_stream.cpp`.

### Binary Stream Format
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
The UpdateType is 0 to indicate an insertion of the associated edge and 1 to indicate a deletion.

### Other Stream Formats
Other file formats can be used by writing a simple file parser that passes graph `update()` the expected edge update format `GraphUpdate := std::pair<Edge, UpdateType>`. See our unit tests under `/test/graph_test.cpp` for examples of string based stream parsing.

If receiving edge updates over the network it is equally straightforward to define a stream format that will receive, parse, and provide those updates to the graph `update()` function.

## Configuration
GraphZeppelin has a number of parameters. These can be defined with the `GraphConfiguration` object. Key parameters include the number of graph workers and the guttering system to use for buffering updates.

See `include/graph_configuration.h` for more details.

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
