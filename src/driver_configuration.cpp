#include <iostream>

#include "driver_configuration.h"

DriverConfiguration& DriverConfiguration::gutter_sys(GutterSystem gutter_sys) {
  _gutter_sys = gutter_sys;
  return *this;
}

DriverConfiguration& DriverConfiguration::disk_dir(std::string disk_dir) {
  _disk_dir = disk_dir;
  return *this;
}

DriverConfiguration& DriverConfiguration::worker_threads(size_t num_worker_threads) {
  _num_worker_threads = num_worker_threads;
  if (_num_worker_threads < 1) {
    std::cout << "num_worker_threads="<< _num_worker_threads << " is out of bounds. [1, infty)"
              << "Defaulting to 1." << std::endl;
    _num_worker_threads = 1;
  }
  return *this;
}

GutteringConfiguration& DriverConfiguration::gutter_conf() {
  return _gutter_conf;
}

std::ostream& operator<< (std::ostream &out, const DriverConfiguration &conf) {
    out << "GraphSketchDriver Configuration:" << std::endl;
    std::string gutter_system = "StandAloneGutters";
    if (conf._gutter_sys == GUTTERTREE)
      gutter_system = "GutterTree";
    else if (conf._gutter_sys == CACHETREE)
      gutter_system = "CacheTree";
    out << " Guttering system      = " << gutter_system << std::endl;
    out << " Worker thread count   = " << conf._num_worker_threads << std::endl;
    out << " On disk data location = " << conf._disk_dir;
    return out;
  }
