#include <fstream>

static void write_configuration(bool use_tree, int groups=1, int g_size=1) {
	// read previous configuration to get GutterTree prefix
	// as this is system dependent and shouldn't be set by our code
	std::ifstream in("streaming.conf");
	std::string path_prefix;
	if (in.is_open()) {
		std::string line;
		while(getline(in, line)) {
			if (line[0] == '#' || line[0] == '\n') continue;
			if (line.substr(0, line.find('=')) == "path_prefix") {
				path_prefix = line.substr(line.find('=') + 1);
				break;
			}
		}
	}
	in.close();

	std::ofstream out("streaming.conf");
	out << "buffering_system=" << (use_tree? "tree" : "standalone") << std::endl;
	out << "path_prefix=" << path_prefix << std::endl;
	out << "num_groups=" << groups << std::endl;
	out << "group_size=" << g_size << std::endl;
	out.close();
}
