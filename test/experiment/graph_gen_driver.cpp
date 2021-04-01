#include <iostream>
#include "../util/graph_gen.h"

int main() {
  int n; double p, r = 0.1; std::string s,t; char c = 0; bool cum = false;
  std::cout << "n: "; std::cin >> n;
  std::cout << "p: "; std::cin >> p;
  std::cout << "r: "; std::cin >> r;
  std::cout << "cum (y/n): "; std::cin >> c;
  if (c == 'y' || c == 'Y') cum = true;
  char default_name[100];
  sprintf(default_name,"%d_%g_%g",n,p,r);
  std::cout << "Out file (" << default_name << ".stream): ";
  std::getline(std::cin,s);
  std::getline(std::cin,s);
  if (s.empty()) {
    s = default_name;
    s += ".stream";
  }
  if (cum) {
    std::cout << "Cum out (" << default_name << ".cum): ";
    std::getline(std::cin,t);
    if (t.empty()) {
      t = default_name;
      t += ".cum";
    }
  }

  auto start = time(NULL);
  generate_stream({n,p,r,0,s,t});
  std::cout << "Completed in " << time(NULL)-start << " seconds" << std::endl;
}