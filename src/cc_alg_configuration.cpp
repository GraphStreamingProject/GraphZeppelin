#include <iostream>

#include "cc_alg_configuration.h"

CCAlgConfiguration& CCAlgConfiguration::disk_dir(std::string disk_dir) {
  _disk_dir = disk_dir;
  return *this;
}

CCAlgConfiguration& CCAlgConfiguration::sketches_factor(double factor) {
  _sketches_factor = factor;
  if (_sketches_factor <= 0) {
    std::cout << "sketches_factor=" << _sketches_factor << " is out of bounds. (0, infty)"
              << "Defaulting to 1." << std::endl;
    _sketches_factor = 1;
  }
  return *this;
}

CCAlgConfiguration& CCAlgConfiguration::batch_factor(double factor) {
  _batch_factor = factor;
  if (_batch_factor <= 0) {
    std::cout << "batch factor=" << _batch_factor << " is out of bounds. (0, infty)"
              << "Defaulting to 1." << std::endl;
    _batch_factor = 1;
  }
  return *this;
}

std::ostream& operator<< (std::ostream &out, const CCAlgConfiguration &conf) {
    out << "Connected Components Algorithm Configuration:" << std::endl;
#ifdef L0_SAMPLING
    out << " Sketching algorithm   = CubeSketch" << std::endl;
#else
    out << " Sketching algorithm   = CameoSketch" << std::endl;
#endif
#ifdef NO_EAGER_DSU
    out << " Using Eager DSU       = False" << std::endl;
#else
    out << " Using Eager DSU       = True" << std::endl;
#endif
    out << " Num sketches factor   = " << conf._sketches_factor << std::endl;
    out << " Batch size factor     = " << conf._batch_factor << std::endl;
    out << " On disk data location = " << conf._disk_dir;
    return out;
  }
