#include <fstream>
#include <iostream>
#include "../../include/graph.h"
#include "../util/graph_gen.h"
#include "../util/graph_verifier.h"
#include "../util/write_configuration.h"

static inline int do_run() {
    ifstream in{"./sample.txt"};
    node_t n, m;
    in >> n >> m;
    Graph g{n};
    int type, a, b;
    while (m--) {
      in >> type >> a >> b;
      if (type == INSERT) {
        g.update({{a, b}, INSERT});
      } else g.update({{a, b}, DELETE});
    }
    g.set_cumul_in("./cumul_sample.txt");
    try {
        g.connected_components();
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
        write_configuration(use_tree, 4);
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
