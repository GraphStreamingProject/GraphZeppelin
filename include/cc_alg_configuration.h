#pragma once

// Graph parameters
class CCAlgConfiguration {
private:
  // Where to place on-disk datastructures
  std::string _disk_dir = ".";

  // Option to create more sketches than for standard connected components
  // Ex factor of 1.5, 1.5 times the sketches
  //    factor of 1, normal quantity of sketches
  double _sketches_factor = 1;

  // Size of update batches as relative to the size of a Supernode
  double _batch_factor = 1;

  friend class CCSketchAlg;
  friend class MCSketchAlg;

public:
  CCAlgConfiguration() {};

  // setters
  CCAlgConfiguration& disk_dir(std::string disk_dir);
  CCAlgConfiguration& sketches_factor(double factor);
  CCAlgConfiguration& batch_factor(double factor);

  // getters
  std::string get_disk_dir() { return _disk_dir; }
  double get_sketches_factor() { return _sketches_factor; }
  double get_batch_factor() { return _batch_factor; }

  friend std::ostream& operator<< (std::ostream &out, const CCAlgConfiguration &conf);

  // moving and copying allowed
  CCAlgConfiguration(const CCAlgConfiguration &oth) = default;
  CCAlgConfiguration (CCAlgConfiguration &&) = default;
};
