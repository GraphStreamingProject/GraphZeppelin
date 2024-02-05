#pragma once

#include <guttering_configuration.h>

enum GutterSystem {
  GUTTERTREE,
  STANDALONE,
  CACHETREE
};

// Paramaters for the sketching algorithm driver
class DriverConfiguration {
private:
  // which guttering system to use for buffering updates
  GutterSystem _gutter_sys = STANDALONE;

  // Where to place on-disk datastructures
  std::string _disk_dir = ".";

  // The number of worker threads
  size_t _num_worker_threads = 1;

  // Configuration for the guttering system
  GutteringConfiguration _gutter_conf;

public:
  DriverConfiguration() {};

  // setters
  DriverConfiguration& gutter_sys(GutterSystem gutter_sys);
  DriverConfiguration& disk_dir(std::string disk_dir);
  DriverConfiguration& worker_threads(size_t num_groups);
  GutteringConfiguration& gutter_conf();

  // getters
  GutterSystem get_gutter_sys() { return _gutter_sys; }
  std::string get_disk_dir() { return _disk_dir; }
  size_t get_worker_threads() { return _num_worker_threads; }

  friend std::ostream& operator<< (std::ostream &out, const DriverConfiguration &conf);

  // no use of equal operator
  DriverConfiguration& operator=(const DriverConfiguration &) = delete;

  // moving and copying allowed
  DriverConfiguration(const DriverConfiguration &oth) = default;
  DriverConfiguration (DriverConfiguration &&) = default;
};
