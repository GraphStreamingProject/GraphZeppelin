#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "types.h"
#include <binary_file_stream.h>

static constexpr size_t update_array_size = 10000;

int main(int argc, char** argv) {

  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_name = argv[1];
  std::string output_name = stream_name + "_shuffled";

  /*
   * Reading Input Edge Stream
   */
  
  BinaryFileStream stream(stream_name);

  std::cout << "Processing stream: " << stream_name << std::endl;

  GraphStreamUpdate update_array[update_array_size];

  std::vector<Edge> edges;
  size_t total_read_updates = 0;
  bool read_complete = false;

  std::cout << "Reading edges...\n";
  auto stream_read_start = std::chrono::steady_clock::now();
  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    total_read_updates += updates;
    if (total_read_updates % 100000000 == 0) {
      std::cout << "  Progress: " << total_read_updates << "\n";
    }

    for (size_t i = 0; i < updates; i++) {
      GraphUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);
      if (upd.type == BREAKPOINT) {
        read_complete = true;
        break;
      }
      else {
        edges.push_back(upd.edge);
      }
    }
  }
  auto stream_read_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> stream_read_time = stream_read_end - stream_read_start;

  std::cout << "Total number of edges read: " << edges.size() << "\n";
  std::cout << "Reading time (sec): " << stream_read_time.count() << "\n";

  /*
   * Shuffling Edges 
   */

  std::cout << "Shuffling edges...\n";

  // seed = 0
  std::default_random_engine e(0);

  auto stream_shuffle_start = std::chrono::steady_clock::now();
  std::shuffle(edges.begin(), edges.end(), e);
  auto stream_shuffle_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> stream_shuffle_time = stream_shuffle_end - stream_shuffle_start;
  std::cout << "Shuffling time (sec): " << stream_shuffle_time.count() << "\n";

  /*
   * Writing Edge Stream with Shuffled Edges
   */

  BinaryFileStream fout(output_name , false);
  fout.write_header(stream.vertices(), stream.edges());

  bool write_complete = false;

  // Re-Initialize update_array
  for (size_t i = 0; i < update_array_size; i++) {
		update_array[i].edge = {0, 0};
    update_array[i].type = INSERT;
	}
  
  size_t update_array_index = 0;
  size_t written_edges = 0;
  std::cout << "Wrting edges back to stream...\n";
  auto stream_write_start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < edges.size(); i++) {
    update_array[update_array_index].edge = edges[i];

    update_array_index++;
    written_edges++;
    if (update_array_index == update_array_size) {
      fout.write_updates(update_array, update_array_size);
      update_array_index = 0;
    }
  }

  // Write any remaining edges to stream
  if (update_array_index > 0) {
		fout.write_updates(update_array, update_array_index);
	}

  auto stream_write_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> stream_write_time = stream_write_end - stream_write_start;
  std::cout << "Writing time (sec): " << stream_write_time.count() << "\n";
  std::cout << "Total number of edges written to stream: " << written_edges << "\n";
}