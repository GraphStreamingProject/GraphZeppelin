#include <iostream>
#include "graph_sketch_driver.h"
#include "cc_sketch_alg.h"
#include "ascii_file_stream.h"
#include "graph_gen.h"
#include "file_graph_verifier.h"

static GraphConfiguration config;

static inline int do_run() {
    AsciiFileStream stream{"./sample.txt"};
    node_id_t n = stream.vertices();
    CCSketchAlg cc_alg{n};
    cc_alg.set_verifier(std::make_unique<FileGraphVerifier>(n, "./cumul_sample.txt"));
    GraphSketchDriver<CCSketchAlg> driver(&cc_alg, &stream, config);
    driver.process_stream_until(END_OF_STREAM);
    driver.prep_query();
    try {
        cc_alg.connected_components();
    } catch (std::exception const &err) {
        return 1;
    }
    return 0;
}

int small_graph_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({1024,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
        failures += do_run();
    }
    return failures;
}

int medium_graph_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({2048,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
        failures += do_run();
    }
    return failures;
}

int main() {
    int runs = 100;
    int num_trails = 500;
    std::vector<int> trial_list;
    std::ofstream out;

    // run both with GutterTree and StandAloneGutters
    for(int i = 0; i < 2; i++) { 
        bool use_tree = (bool) i;

        // setup configuration file per buffering
        config.gutter_sys(use_tree ? GUTTERTREE : STANDALONE);
        config.worker_threads(4);
        std::string prefix = use_tree? "tree" : "gutters";
        std::string test_name;

        /************* small graph test *************/
        test_name = prefix + "_" + "small_graph_test";
        fprintf(stderr, "%s\n", test_name.c_str());
        out.open("./" + test_name);
        for(int i = 0; i < num_trails; i++) {
            if (i % 50 == 0) fprintf(stderr, "trial %i\n", i);
            int trial_result = small_graph_test(runs);
            trial_list.push_back(trial_result);
        }
        // output the results of these trials
        for (unsigned i = 0; i < trial_list.size(); i++) {
            out << trial_list[i] << " " << runs << "\n";
        }
        trial_list.clear();
        out.close();

        /************* medium graph test ************/
        test_name = prefix + "_" + "medium_graph_test";
        fprintf(stderr, "%s\n", test_name.c_str());
        out.open("./" + test_name);
        for(int i = 0; i < num_trails; i++) {
            if (i % 50 == 0) fprintf(stderr, "trial %i\n", i);
            int trial_result = medium_graph_test(runs);
            trial_list.push_back(trial_result);
        }
        // output the results of these trials
        for (unsigned i = 0; i < trial_list.size(); i++) {
            out << trial_list[i] << " " << runs << "\n";
        }
        trial_list.clear();
        out.close();
    }
}
