#include <iostream>
#include <guttering_configuration.h>
static constexpr char config_loc[] = "streaming.conf";

// TODO: Replace this with an enum defined by GutterTree repo
enum GutterSystem {
  GUTTERTREE,
  STANDALONE,
  CACHETREE
};

// Graph parameters
class GraphConfiguration {
public:
  // which guttering system to use for buffering updates
  GutterSystem gutter_sys = STANDALONE;

  // Where to place on-disk datastructures
  std::string disk_dir = ".";

  // Backup supernodes in memory or on disk when performing queries
  bool backup_in_mem = true;

  // The number of graph workers
  size_t num_groups = 1;

  // How many OMP threads each graph worker uses
  size_t group_size = 1;

  // Configuration for the guttering system
  const GutteringConfiguration gutter_conf;

  GraphConfiguration(GutterSystem gut_sys, std::string dir, bool mem_back, size_t num_group, 
   size_t group_size, const GutteringConfiguration &gutter_conf) : 
   gutter_sys(gut_sys), disk_dir(dir), backup_in_mem(mem_back), num_groups(num_group), 
   group_size(group_size), gutter_conf(gutter_conf) {};

  GraphConfiguration() {
    std::ifstream config_file(config_loc);
    std::string line;
    if (config_file.is_open()) {
      while(getline(config_file, line)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if(line.substr(0, line.find('=')) == "buffering_system") {
          std::string buf_str = line.substr(line.find('=') + 1);
          if (buf_str == "tree") gutter_sys = GUTTERTREE;
          else if (buf_str == "cachetree") gutter_sys = CACHETREE;
          else if (buf_str == "standalone") gutter_sys = STANDALONE;
          else {
            std::cout << "WARNING: string " << buf_str << " is not a valid option for " 
                  "buffering. Defaulting to StandAloneGutters." << std::endl;
          }
        }
        if(line.substr(0, line.find('=')) == "disk_dir") {
          disk_dir = line.substr(line.find('=') + 1) + "/";
        }
        if(line.substr(0, line.find('=')) == "backup_in_mem") {
          std::string flag = line.substr(line.find('=') + 1);
          if (flag == "ON")
            backup_in_mem = true;
          else if (flag == "OFF")
            backup_in_mem = false;
          else
            std::cout << "WARNING: string " << flag << " is not a valid option for backup_in_mem"
                   "Defaulting to ON." << std::endl;
        }
        if(line.substr(0, line.find('=')) == "num_groups") {
          num_groups = std::stoi(line.substr(line.find('=') + 1));
          if (num_groups < 1) { 
            std::cout << "num_groups="<< num_groups << " is out of bounds. "
                          << "Defaulting to 1." << std::endl;
            num_groups = 1; 
          }
        }
        if(line.substr(0, line.find('=')) == "group_size") {
          group_size = std::stoi(line.substr(line.find('=') + 1));
          if (group_size < 1) { 
            std::cout << "group_size="<< group_size << " is out of bounds. "
                      << "Defaulting to 1." << std::endl;
            group_size = 1; 
          }
        }
      }
    } else {
      std::cout << "WARNING: Could not open configuration file! Using default values." << std::endl;
    }
  }

  void print() {
    std::cout << "GraphStreaming Configuration:" << std::endl;
    std::string gutter_system = "StandAloneGutters";
    if (gutter_sys == GUTTERTREE)
      gutter_system = "GutterTree";
    else if (gutter_sys == CACHETREE)
      gutter_system = "CacheTree";
    std::cout << " Guttering system      = " << gutter_system << std::endl;
    std::cout << " Number of groups      = " << num_groups << std::endl;
    std::cout << " Size of groups        = " << group_size << std::endl;
    std::cout << " On disk data location = " << disk_dir << std::endl;
    std::cout << " Backup sketch to RAM  = " << (backup_in_mem? "ON" : "OFF") << std::endl;
  }

  // no use of equal operator
  GraphConfiguration &operator=(const GraphConfiguration &) = delete;

  // moving and copying allowed
  GraphConfiguration(const GraphConfiguration &oth) = default;
  GraphConfiguration (GraphConfiguration &&) = default;

};
