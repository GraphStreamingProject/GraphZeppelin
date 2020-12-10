#ifndef TEST_GRAPH_GEN_H
#define TEST_GRAPH_GEN_H

/**
 * Generates a 1024-node graph with approximately 30,000 edge insert/deletes.
 * Writes stream output to sample.txt
 * Writes cumulative output to cum_sample.txt
 */
void generate_stream();

#endif //TEST_GRAPH_GEN_H
