#include "TokuInterface.h"
#include <stdio.h>

int main() {

	printf("Creating db ...\n");
	TokuInterface db = TokuInterface();

	printf("===== Beginning Insertions  =====\n");
	for (int i = 2; i <= 10; i++) {
		std::pair<uint64_t, uint64_t>edge(1, i);
		db.putEdge(edge, 1); // edge update INSERT
	}

	for (int i = 5; i <= 8; i++) {
		std::pair<uint64_t, uint64_t>edge(1, i);
		db.putEdge(edge, -1); // edge update DELETE
	}
	
	// insert and delete very quickly to see if we can handle it
	std::pair<uint64_t, uint64_t>edge(1, 5);
	db.putEdge(edge, 1);
	db.putEdge(edge, -1);
	db.putEdge(edge, 1);
	db.putEdge(edge, -1);
	db.putEdge(edge, 1);
	db.putEdge(edge, -1);
	db.putEdge(edge, 1);
	db.putEdge(edge, -1);
	db.putEdge(edge, 1);

	printf("=== Query response for node 1 ===\n");
	std::vector<std::pair<uint64_t, int8_t>> *edges = db.getEdges(1);
	for (auto edge : *edges) {
		printf("%lu : %d\n", edge.first, edge.second);
	}

	printf("=== Checking That All Deleted ===\n");
	edges = db.getEdges(1);
	for (auto edge : *edges) {
		printf("ERROR: edge %lu : %d is present\n", edge.first, edge.second);
	}

	delete edges;

    return 0;
}