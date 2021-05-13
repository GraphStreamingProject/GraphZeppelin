#pragma once
#include <istream>
#include "graph.h"

/**
* Let $G$ be the graph represented by the stream, and define 
* $F_i$ for $1 <= i <= k$ as a spanning forest of 
* $G - \cup_{j = 1}^{i-1} F_j$, where subtraction between graphs
* excludes nodes.
*
* @param k The number of edge-disjoint spanning forests to take
* the union of.
* 
* @return U $ = \cup_{i = 1}^{k} F_i$. A useful property of U is
* that G is k-edge-connected iff U is k-edge-connected. If U is 
* k-edge-connected, then it is called a k-skeleton of G.
*/
vector<vector<Node>> k_skeleton (istream * updates_stream, int k);
