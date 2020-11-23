#pragma once
#include <cstdlib>
#include <exception>
#include <set>
#include "supernode.h"

typedef unsigned long long int Node;
typedef std::pair<Node, Node> Edge;

enum UpdateType {
  INSERT,
  DELETE,
};

typedef pair<Edge, UpdateType> GraphUpdate;

class Graph{
  const unsigned long long int num_nodes;
  bool UPDATE_LOCKED = false;
  // a set containing one "representative" from each supernode
  set<Node>* representatives;
  Supernode** supernodes;

  // DSU representation of supernode relationship
  Node* parent;
  Node get_parent(Node node);
public:
  explicit Graph(unsigned long long int num_nodes);
  ~Graph();
  void update(GraphUpdate upd);

  /**
   * Main algorithm utilizing Boruvka and L_0 sampling.
   * @return a vector of the connected components in the graph.
   */
  vector<set<Node>> connected_components();
};

class UpdateLockedException : public exception {
  virtual const char* what() const throw() {
    return "The graph cannot be updated: Connected components algorithm has "
           "already started";
  }
};