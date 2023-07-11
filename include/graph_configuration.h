#pragma once

#include <guttering_configuration.h>

// forward declaration
class Graph;

// TODO: Replace this with an enum defined by GutterTree repo
enum GutterSystem {
  GUTTERTREE,
  STANDALONE,
  CACHETREE
};

// Graph parameters
class GraphConfiguration {
private:
  // which guttering system to use for buffering updates
  GutterSystem _gutter_sys = STANDALONE;

  // Where to place on-disk datastructures
  std::string _disk_dir = ".";

  // Backup supernodes in memory or on disk when performing queries
  bool _backup_in_mem = true;

  // The number of graph workers
  size_t _num_groups = 1;

  // How many OMP threads each graph worker uses
  size_t _group_size = 1;

  // Option to create more sketches than for standard connected components
  // Ex factor of 1, double the sketches
  //    factor of 0.5, 1.5 times the sketches
  //    factor of 0, normal quantity of sketches
  double _adtl_skts_factor = 0;

  // Configuration for the guttering system
  GutteringConfiguration _gutter_conf;

  friend class Graph;

public:
  GraphConfiguration() {};

  // setters
  GraphConfiguration& gutter_sys(GutterSystem gutter_sys);

  GraphConfiguration& disk_dir(std::string disk_dir);

  GraphConfiguration& backup_in_mem(bool backup_in_mem);

  GraphConfiguration& num_groups(size_t num_groups);

  GraphConfiguration& group_size(size_t group_size);

  GraphConfiguration& adtl_skts_factor(double factor);

  GutteringConfiguration& gutter_conf();

  friend std::ostream& operator<< (std::ostream &out, const GraphConfiguration &conf);

  // no use of equal operator
  GraphConfiguration& operator=(const GraphConfiguration &) = delete;

  // moving and copying allowed
  GraphConfiguration(const GraphConfiguration &oth) = default;
  GraphConfiguration (GraphConfiguration &&) = default;

};
