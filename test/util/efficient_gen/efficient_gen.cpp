#include <iostream>
#include "efficient_gen.h"

int main() {
  int n; double p, r = 0.1; std::string s,t; char c = 0; bool cum = false;
  std::cout << "n: "; std::cin >> n;
  std::cout << "p: "; std::cin >> p;
  std::cout << "r: "; std::cin >> r;
  std::cout << "cum (y/n): "; std::cin >> c;
  if (c == 'y' || c == 'Y') cum = true;
  std::cout << "Out file: "; std::cin >> s;
  if (cum) { std::cout << "Cum out: "; std::cin >> t; }

  auto start = time(NULL);
  write_edges(n, p, "./TEMP_F");
  insert_delete(r,"./TEMP_F", s);
  if (cum) write_cum(s,t);
  std::cout << "Completed in " << time(NULL)-start << " seconds" << std::endl;
}
