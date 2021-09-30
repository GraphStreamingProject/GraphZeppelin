#include <fstream>
#include <iostream>
#include "../include/graph.h"
#include "../test/util/graph_gen.h"
#include "../test/util/graph_verifier.h"

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

int tiny_graph_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({1024,0.001,0.5,0,"./sample.txt","./cumul_sample.txt"});
        failures += do_run();
    }
    return failures;
}

int small_graph_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({1024,0.03,0.5,0,"./sample.txt","./cumul_sample.txt"});
        failures += do_run();
    }
    return failures;
}

int medium_graph_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({1024,0.20,0.5,0,"./sample.txt","./cumul_sample.txt"});
        failures += do_run();
    }
    return failures;
}

int large_with_iso_test(int runs) {
    int failures = 0;
    for (int i = 0; i < runs; i++) {
        generate_stream({1024,0.5,0.5,0,"./sample.txt","./cumul_sample.txt"});
        std::fstream graph_file{"./sample.txt"};
        std::fstream cumul_file{"./cumul_sample.txt"};
        graph_file.write("1074", 4); // increase the node size to create iso nodes
        cumul_file.write("1074", 4); // do the same for the cumulative file
        failures += do_run();
    }
    return failures;
}

int main() {
    int runs = 10;
    int num_trails = 2;
    std::vector<int> trial_list;
    std::ofstream out;
    
    /************* tiny graph test **************/
    out.open("./tiny_graph_test");
    for(int i = 0; i < num_trails; i++) {
        int trial_result = tiny_graph_test(runs);
        trial_list.push_back(trial_result);
    }
    // output the results of these trials
    for (unsigned i = 0; i < trial_list.size(); i++) {
        out << trial_list[i] << " " << runs << "\n";
    }
    trial_list.clear();
    out.close();

    /************* small graph test *************/
    out.open("./small_graph_test");
    for(int i = 0; i < num_trails; i++) {
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
    out.open("./medium_graph_test");
    for(int i = 0; i < num_trails; i++) {
        int trial_result = medium_graph_test(runs);
        trial_list.push_back(trial_result);
    }
    // output the results of these trials
    for (unsigned i = 0; i < trial_list.size(); i++) {
        out << trial_list[i] << " " << runs << "\n";
    }
    trial_list.clear();
    out.close();

    /************** large iso test **************/
    out.open("./large_with_iso_test");
    for(int i = 0; i < num_trails; i++) {
        int trial_result = large_with_iso_test(runs);
        trial_list.push_back(trial_result);
    }
    // output the results of these trials
    for (unsigned i = 0; i < trial_list.size(); i++) {
        out << trial_list[i] << " " << runs << "\n";
    }
    trial_list.clear();
    out.close();
}