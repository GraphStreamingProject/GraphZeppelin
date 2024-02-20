#pragma once

// Graph parameters
class MCAlgConfiguration {
private:
  // Where to place on-disk datastructures
  std::string _disk_dir = ".";

  // Option to create more sketches than for standard connected components
  // Ex factor of 1.5, 1.5 times the sketches
  //    factor of 1, normal quantity of sketches
  double _sketches_factor = 1;

  // Size of update batches as relative to the size of a Supernode
  double _batch_factor = 1;

  friend class MCSketchAlg;

public:
  MCAlgConfiguration() {};

  // setters
  MCAlgConfiguration& disk_dir(std::string disk_dir);
  MCAlgConfiguration& sketches_factor(double factor);
  MCAlgConfiguration& batch_factor(double factor);

  // getters
  std::string get_disk_dir() { return _disk_dir; }
  double get_sketch_factor() { return _sketches_factor; }
  double get_batch_factor() { return _batch_factor; }

  friend std::ostream& operator<< (std::ostream &out, const MCAlgConfiguration &conf);

  // moving and copying allowed
  MCAlgConfiguration(const MCAlgConfiguration &oth) = default;
  MCAlgConfiguration (MCAlgConfiguration &&) = default;
};
