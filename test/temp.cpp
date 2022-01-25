#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const int notification_frequency = 1e6;
const unsigned max_ccs = 4096; // 2^12
const int num_cc_combinations = 13;
unsigned long long total_edges = 4294967296;

int main() {
  vector<unsigned> cc_mod;
  unsigned long long num_rounds = total_edges / (notification_frequency * num_cc_combinations);
  int cnt = 0;
  for (unsigned i = 0; i <= total_edges / notification_frequency; ++i) {
    unsigned num_ccs = max(max_ccs >> cnt, 1u);
    for (unsigned j = 0; j < num_rounds; ++j, ++i) {
      cc_mod.push_back(num_ccs);
    }
    ++cnt;
  }
  sort(cc_mod.begin(), cc_mod.end(), greater<unsigned>());
  cout << cc_mod.size()  << endl;
  cout << cc_mod[0] << " "
    << cc_mod[380] << " " << cc_mod[720] << " "
    << cc_mod[cc_mod.size() - 1] << endl;
}