#include <fstream>
#include <iostream>
#include "../../include/graph.h"
#include "../util/graph_gen.h"
#include "../util/file_graph_verifier.h"

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
    g.set_verifier(std::make_unique<FileGraphVerifier>("./cumul_sample.txt"));
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

int medium_with_iso_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({2048,0.002,0.5,0,"./sample.txt","./cumul_sample.txt"});
        std::fstream graph_file{"./sample.txt"};
        std::fstream cumul_file{"./cumul_sample.txt"};
        graph_file.write("2070", 4); // increase the node size to create iso nodes
        cumul_file.write("2070", 4); // do the same for the cumulative file
        failures += do_run();
    }
    return failures;
}

int main() {
    int runs = 100;
    int num_trails = 500;
    std::vector<int> trial_list;
    std::ofstream out;
    
    /************* small graph test *************/
    fprintf(stderr, "small graph test\n");
    out.open("./small_graph_test");
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
    fprintf(stderr, "medium graph test\n");
    out.open("./medium_graph_test");
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

    /************** medium iso test *************/
    // Turned off for now because I'm not convinced it provides useful information - e
    // fprintf(stderr, "medium iso graph test\n");
    // out.open("./medium_with_iso_test");
    // for(int i = 0; i < num_trails; i++) {
    //     if (i % 50 == 0) fprintf(stderr, "trial %i\n", i);
    //     int trial_result = medium_with_iso_test(runs);
    //     trial_list.push_back(trial_result);
    // }
    // // output the results of these trials
    // for (unsigned i = 0; i < trial_list.size(); i++) {
    //     out << trial_list[i] << " " << runs << "\n";
    // }
    // trial_list.clear();
    // out.close();
}
