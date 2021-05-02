/*
* Our implementation is_bipartite of the reduction must accept the
* update stream as input out of necessity of creating a sketch of the
* transformed graph during ingestion. This is not the case with the
* two functions we will be comparing it against. The functions below
* provides two wrappers so all three functions can be evaluated based
* on stream ingestion.
*/

#pragma once
#include "../../include/graph.h"
#include <boost/graph/bipartite.hpp>
#include <boost/graph/adjacency_list.hpp>

vector<set<Node>> ingest_con_comp (istream * udpates_stream);
bool ingest_boost_bip (istream * updates_stream);
