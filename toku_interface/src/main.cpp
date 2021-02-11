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

	// flush everything that hasn't met threshold
	db.flush();

    return 0;
}
