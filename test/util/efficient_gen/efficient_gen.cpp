#include <iostream>
#include "efficient_gen.h"

int main() {
  int n; double p, r = 0.1; std::string s,t; char c = 0; bool cumul = false;
  std::cout << "n: "; std::cin >> n;
  std::cout << "p: "; std::cin >> p;
  std::cout << "r: "; std::cin >> r;
  std::cout << "cumul (y/n): "; std::cin >> c;
  if (c == 'y' || c == 'Y') cumul = true;
  std::cout << "Out file: "; std::cin >> s;
  if (cumul) { std::cout << "Cumul out: "; std::cin >> t; }

  auto start = time(nullptr);
  write_edges(n, p, "./TEMP_F");
  insert_delete(r,"./TEMP_F", s);
  if (cumul) write_cumul(s,t);
  std::cout << "Completed in " << time(nullptr)-start << " seconds" << std::endl;
}
