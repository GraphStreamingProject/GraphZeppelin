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

GraphConfiguration& GraphConfiguration::num_groups(size_t num_groups) {
  _num_groups = num_groups;
  if (_num_groups < 1) {
    std::cout << "num_groups="<< _num_groups << " is out of bounds. [1, infty)"
              << "Defaulting to 1." << std::endl;
    _num_groups = 1;
  }
  return *this;
}

GraphConfiguration& GraphConfiguration::group_size(size_t group_size) {
  _group_size = group_size;
  if (_group_size < 1) {
    std::cout << "group_size="<< _group_size << " is out of bounds. [1, infty)"
              << "Defaulting to 1." << std::endl;
    _group_size = 1;
  }
  return *this;
}

GraphConfiguration& GraphConfiguration::sketches_factor(double factor) {
  _sketches_factor = factor;
  if (_sketches_factor < 0) {
    std::cout << "adtl_skts_factor=" << _sketches_factor << " is out of bounds. (0, infty)"
              << "Defaulting to 1." << std::endl;
    _sketches_factor = 1;
  }
  if (_sketches_factor != 1) {
    std::cerr << "WARNING: Your graph configuration specifies using a factor " << _sketches_factor 
              << " of the normal quantity of sketches." << std::endl;
    std::cerr << "         Is this intentional? If not, set adtl_skts_factor to one!" << std::endl;
  }
  return *this;
}

GraphConfiguration& GraphConfiguration::batch_factor(double factor) {
  _batch_factor = factor;
  if (_batch_factor < 0) {
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
    out << " Guttering system      = " << gutter_system << std::endl;
    out << " Number of groups      = " << conf._num_groups << std::endl;
    out << " Size of groups        = " << conf._group_size << std::endl;
    out << " On disk data location = " << conf._disk_dir << std::endl;
    out << " Backup sketch to RAM  = " << (conf._backup_in_mem? "ON" : "OFF") << std::endl;
    out << conf._gutter_conf;
    return out;
  }