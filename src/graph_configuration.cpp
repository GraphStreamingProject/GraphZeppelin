#include <iostream>

#include "../include/graph_configuration.h"

GraphConfiguration& GraphConfiguration::gutter_sys(GutterSystem gutter_sys) {
  _gutter_sys = gutter_sys;
  return *this;
}

GraphConfiguration& GraphConfiguration::disk_dir(std::string disk_dir) {
  _disk_dir = disk_dir;
  return *this;
}

GraphConfiguration& GraphConfiguration::backup_in_mem(bool backup_in_mem) {
  _backup_in_mem = backup_in_mem;
  return *this;
}

GraphConfiguration& GraphConfiguration::num_graph_workers(size_t num_graph_workers) {
  _num_graph_workers = num_graph_workers;
  if (_num_graph_workers < 1) {
    std::cout << "num_graph_workers="<< _num_graph_workers << " is out of bounds. [1, infty)"
              << "Defaulting to 1." << std::endl;
    _num_graph_workers = 1;
  }
  return *this;
}

GraphConfiguration& GraphConfiguration::sketches_factor(double factor) {
  _sketches_factor = factor;
  if (_sketches_factor <= 0) {
    std::cout << "sketches_factor=" << _sketches_factor << " is out of bounds. (0, infty)"
              << "Defaulting to 1." << std::endl;
    _sketches_factor = 1;
  }
  if (_sketches_factor != 1) {
    std::cerr << "WARNING: Your graph configuration specifies using a factor " << _sketches_factor 
              << " of the normal quantity of sketches." << std::endl;
    std::cerr << "         Is this intentional? If not, set sketches_factor to one!" << std::endl;
  }
  return *this;
}

GraphConfiguration& GraphConfiguration::batch_factor(double factor) {
  _batch_factor = factor;
  if (_batch_factor <= 0) {
    std::cout << "batch factor=" << _batch_factor << " is out of bounds. (0, infty)"
              << "Defaulting to 1." << std::endl;
    _batch_factor = 1;
  }
  return *this;
}

GutteringConfiguration& GraphConfiguration::gutter_conf() {
  return _gutter_conf;
}

std::ostream& operator<< (std::ostream &out, const GraphConfiguration &conf) {
    out << "GraphStreaming Configuration:" << std::endl;
    std::string gutter_system = "StandAloneGutters";
    if (conf._gutter_sys == GUTTERTREE)
      gutter_system = "GutterTree";
    else if (conf._gutter_sys == CACHETREE)
      gutter_system = "CacheTree";
#ifdef L0_SAMPLING
    out << " Sketching algorithm   = CubeSketch" << std::endl;
#else
    out << " Sketching algorithm   = CameoSketch" << std::endl;
#endif
    out << " Guttering system      = " << gutter_system << std::endl;
    out << " Num sketches factor   = " << conf._sketches_factor << std::endl;
    out << " Batch size factor     = " << conf._batch_factor << std::endl;
    out << " Graph worker count    = " << conf._num_graph_workers << std::endl;
    out << " On disk data location = " << conf._disk_dir << std::endl;
    out << " Backup sketch to RAM  = " << (conf._backup_in_mem? "ON" : "OFF") << std::endl;
    out << conf._gutter_conf;
    return out;
  }
