#include <string.h>
#include <graph_sketch_driver.h>
#include <cc_sketch_alg.h>
#include <binary_file_stream.h>
#include <cmath>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

std::pair<double, double> calc_stats(const std::vector<size_t> &data) {
  double avg = 0;
  size_t total = 0;
  for (size_t i = 0; i < data.size(); i++) {
    avg += i * data[i];
    total += data[i];
  }
  avg /= total;

  double dev = 0;
  for (size_t i = 0; i < data.size(); i++) {
    dev += data[i] * pow(double(i) - avg, 2);
  }

  return {avg, sqrt(dev / total)};
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

  size_t num_threads = 40;
  size_t reader_threads = 8;
  std::vector<size_t> rounds_required;
  std::mt19937_64 seed_gen(seed);

  node_id_t num_vertices;
  size_t num_updates;
  {
    BinaryFileStream stream(stream_file);
    num_vertices = stream.vertices();
    num_updates  = stream.edges();
  }
  size_t vertex_power = ceil(log2(num_vertices));
  size_t errors = 0;
  size_t empty = 0;
  std::chrono::duration<double> ingest_time(0);
  std::chrono::duration<double> query_time(0);
  std::chrono::duration<double> filter_time(0);

  for (size_t trial = 0; trial < trials; trial++) {
    BinaryFileStream stream(stream_file);
    std::cout << "Extracting " << vertex_power << " Spanning Forests from: " << stream_file << std::endl;
    std::cout << "vertices    = " << num_vertices << std::endl;
    std::cout << "num_updates = " << num_updates << std::endl;
    std::cout << std::endl;

    auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
    auto cc_config = CCAlgConfiguration().sketches_factor(1.6);
    CCSketchAlg cc_alg{num_vertices, seed_gen(), cc_config};
    GraphSketchDriver<CCSketchAlg> driver{&cc_alg, &stream, driver_config, reader_threads};

    rounds_required.resize(cc_alg.max_rounds());

    std::cout << "Beginning stream ingestion ... "; fflush(stdout);
    auto start = std::chrono::steady_clock::now();
    driver.process_stream_until(END_OF_STREAM);
    std::cout << "Stream processed!" << std::endl;
    ingest_time += std::chrono::steady_clock::now() - start;
    size_t max_rounds_used = 0;

    // figure out how many rounds are required to extract log V spanning forests
    for (size_t s = 0; s < vertex_power; s++) {
      auto start = std::chrono::steady_clock::now();
      try {
        driver.prep_query();
        SpanningForest forest = cc_alg.calc_spanning_forest();
        std::cout << "sf: " << s + 1 << ", rounds: " << cc_alg.last_query_rounds
                  << "/" << cc_alg.max_rounds() << "               \r";
        fflush(stdout);
        max_rounds_used = std::max(max_rounds_used, cc_alg.last_query_rounds);

        query_time += std::chrono::steady_clock::now() - start;

        if (forest.get_edges().size() == 0) {
          std::cout << std::endl << "Exiting because of empty Spanning Forest " << s << std::endl;
          empty += 1;
          break;
        }
        start = std::chrono::steady_clock::now();
        const auto &sf_edges = forest.get_edges();

        // filter out all the found edges from the sketches
        // This is technically illegal behavior. Which is like the point of this test :)
        for (auto edge : sf_edges) {
          cc_alg.update({edge, DELETE});
        }
        filter_time += std::chrono::steady_clock::now() - start;
      } catch (OutOfSamplesException &err) {
        std::cout << std::endl << "Got OutOfSamplesException on spanning forest " << s + 1 << std::endl;
        errors += 1;
        break;
      } catch (...) {
        std::cout << std::endl << "Got unknown exception on spanning forest " << s + 1 << std::endl;
        errors += 1;
        break;
      }
    }

    // add number of rounds to get log V spanning forests to vector
    rounds_required[max_rounds_used] += 1;
  }

  std::cout << std::endl;
  std::cout << "ERRORS = " << errors << std::endl;
  std::cout << "EMPTY = " << empty << std::endl;

  for (size_t i = 0; i < rounds_required.size(); i++) {
    std::cout << i << ", " << rounds_required[i] << std::endl;
  }

  auto stats = calc_stats(rounds_required);
  std::cout << "avg = " << stats.first << " std dev = " << stats.second << std::endl;
  std::cout << "ingest: " << ingest_time.count() << std::endl;
  std::cout << "query:  " << query_time.count() << std::endl;
  std::cout << "filter: " << filter_time.count() << std::endl;

  std::ofstream output_file("rounds_required.txt");
  output_file << "ERRORS = " << errors << std::endl;
  output_file << "EMPTY = " << empty << std::endl;
  for (size_t i = 0; i < rounds_required.size(); i++) {
    output_file << i << ", " << rounds_required[i] << std::endl;
  }
  output_file << "avg, " << stats.first << std::endl;
  output_file << "std dev, " << stats.second << std::endl;
  output_file << "ingest: " << ingest_time.count() << std::endl;
  output_file << "query:  " << query_time.count() << std::endl;
  output_file << "filter: " << filter_time.count() << std::endl;
}
