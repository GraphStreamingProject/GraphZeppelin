#include <string.h>
#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <cmath>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

std::pair<double, double> calc_stats(std::vector<size_t> &data) {
  double avg = 0;
  for (auto d : data) {
    avg += d;
  }
  avg /= data.size();

  double dev = 0;
  for (auto d : data) {
    dev += pow(double(d) - avg, 2);
  }

  return {avg, sqrt(dev / data.size())};
}

// Main function that populates a graph and then tests how many SFs can be extracted from it
int main(int argc, char **argv) {
  if (argc < 3 || argc > 5) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_stream trials [seed]" << std::endl;
    std::cout << "\"graph_stream\" must be a BinaryFileStream." << std::endl;
    std::cout << "Optionally specify a seed, otherwise one is chosen randomly" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_file = argv[1];
  size_t trials = std::stol(argv[2]);
  size_t seed = get_seed();
  if (argc == 4) {
    seed = std::stol(argv[3]);
  }

  size_t num_threads = 24;
  size_t reader_threads = 4;
  std::vector<size_t> sfs_extracted;
  std::mt19937_64 seed_gen(seed);

  for (size_t trial = 0; trial < trials; trial++) {
    BinaryFileStream stream(stream_file);
    node_id_t num_vertices = stream.vertices();
    size_t num_updates  = stream.edges();
    std::cout << "Extracting Spanning Forests from: " << stream_file << std::endl;
    std::cout << "vertices    = " << num_vertices << std::endl;
    std::cout << "num_updates = " << num_updates << std::endl;
    std::cout << std::endl;

    auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
    auto cc_config = CCAlgConfiguration();
    CCSketchAlg cc_alg{num_vertices, seed_gen(), cc_config};
    GraphSketchDriver<CCSketchAlg> driver{&cc_alg, &stream, driver_config, reader_threads};

    std::cout << "Beginning stream ingestion ... "; fflush(stdout);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();
    std::cout << "Ingestion done!" << std::endl;
    bool extract = true;
    size_t s = 0;
    for (; extract; s++) {
      try {
        cc_alg.calc_spanning_forest();
      } catch (OutOfSamplesException &err) {
        std::cout << std::endl << "Got OutOfSamplesException on spanning forest " << s + 1 << std::endl;
        extract = false;
        break;
      } catch (...) {
        std::cout << std::endl << "Got unknown exception on spanning forest " << s + 1 << std::endl;
        extract = false;
        break;
      }
      std::cout << "sf: " << s + 1 << ", rounds: " << cc_alg.last_query_rounds 
                << "/" << cc_alg.max_rounds() << "               \r";
      fflush(stdout);


      SpanningForest forest = cc_alg.calc_spanning_forest();

      if (forest.get_edges().size() == 0) {
        std::cout << std::endl << "Exiting because of empty Spanning Forest " << s << std::endl;
        extract = false;
        continue;
      }

      const auto &sf_edges = forest.get_edges();

      // filter out all the found edges from the sketches
      // This is technically illegal behavior. Which is like the point of this test :)
      for (auto edge : sf_edges) {
        cc_alg.update({edge, DELETE});
      }
    }

    // add number of spanning forests extracted to vector
    sfs_extracted.push_back(s);
  }

  for (auto i : sfs_extracted) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  auto stats = calc_stats(sfs_extracted);
  std::cout << "avg = " << stats.first << " std dev = " << stats.second << std::endl;

  std::ofstream output_file("sfs_extracted.txt");
  for (auto i : sfs_extracted) {
    output_file << i << ", ";
  }
  output_file << std::endl;
  output_file << "avg, " << stats.first << std::endl;
  output_file << "std dev, " << stats.second << std::endl;
}
