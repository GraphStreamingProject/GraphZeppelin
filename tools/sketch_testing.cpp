#include <iostream>
#include <random>
#include <set>
#include <cassert>

#include "sketch.h"
#include "cc_alg_configuration.h"

/*

  The purpose of this file is to test the probability that a sketch column returns a nonzero
  That is, for a number of nonzeroes z, how what's the probability of success?  

  We model this as a binomial process for the sake of confidence intervals / stddev.
  
  Originally, this code inserted z random elements into a sketch then queried it.

  As a first speed optimization (that didn't appear to change outcome) (xxHash works well) 
  We replaced random insertion with sequential inserted.

  As a second speed optimization, we queried the sketch after every update. 
  That is, instead of O(z^2) insertions per z data points, we perform O(z) insertions per z data points.
  This sacrifices independence. Whether or not the z-1th sketch is good is a fantastic predictor for the zth sketch being good.
  But, for a given z, the results are still independent.

  For parity with the main code, column seeds are sequential.

  The output of this is intended to be parsed into summary stats by sum_sketch_testing.py
*/


std::random_device dev;

std::mt19937_64 rng(dev());
using rand_type = std::mt19937_64::result_type;

    
rand_type gen(rand_type n)
{
    std::uniform_int_distribution<rand_type> dist(0,n-1); 
    return dist(rng);
}

rand_type seed = gen(1ll << 62);

rand_type gen_seed()
{
    //std::uniform_int_distribution<rand_type> dist(0,1ll << 63);
    //return dist(rng);
    return seed++;
}


enum ResultType {
    R_GOOD=0,
    R_BAD=1,
    R_HASHFAIL=2
};

ResultType test_z(rand_type n, rand_type z)
{
    assert(z >= 1);
    assert(z <= n*n);
    Sketch sketch(n, gen_seed(), 1, 1);

    // Generate z edges and track them
    /*std::unordered_set<rand_type> edges;
    while (edges.size() < z)
    {
        edges.insert(gen(n*n));
    }

    for (const auto& r : edges)
    {
        sketch.update(r);
    }
    */
    for (rand_type i = 0; i < z; i++)
        sketch.update(i);
    // Sample the sketches
    SketchSample query_ret = sketch.sample();
    SampleResult ret_code = query_ret.result;

    assert(ret_code != ZERO);

    if (ret_code == GOOD)
    {
        //if (edges.find(res) == edges.end())
        //    return R_HASHFAIL;
        return R_GOOD;
    }   
    return R_BAD;
}

std::pair<double, double> fit_to_binomial(rand_type ngood, rand_type ntrials)
{
    double p = ngood / (1.0 * ntrials);
    double variance = ntrials * p * (1-p);
    double stddev = sqrt(variance);
    return std::pair<double, double>(p, stddev/ntrials);
}

std::pair<double, double> test_nz_pair(rand_type n, rand_type z)
{
    int ntrials = 500;
    int results[3] = {0,0,0};
    for (int i = 0; i < ntrials; i++)
        results[test_z(n, z)]++;
    //std::cout << "GOOD: " << results[0] << std::endl;
    //std::cout << "BAD: " << results[1] << std::endl;
    //std::cout << "HASHFAIL: " << results[2] << std::endl;
    int ngood = results[0];
    // Fit to binomial
    return fit_to_binomial(ngood, ntrials);
}

void test_n_one(rand_type n, rand_type* good, rand_type max_z)
{
  Sketch sketch(n*n, gen_seed(), 1, 1);
  for (rand_type i = 0; i < max_z; i++)
  {
    sketch.update(i);
    // Sample the sketches
    SketchSample query_ret = sketch.sample();
    SampleResult ret_code = query_ret.result;
    //assert(ret_code != ZERO);
    if (ret_code == GOOD)
      good[i]++;
    sketch.reset_sample_state();
  }
}

void test_n(rand_type n)
{
  int ntrials = 500;
  rand_type max_z = 1+(n*n)/4;
  // Default init to 0?
  rand_type* good = new rand_type[max_z];
  for (int i = 0; i < ntrials; i++)
    test_n_one(n, good, max_z);

  double worst_3sigma = 1;
  rand_type worst_i = 0;
  for (rand_type i = 0; i < max_z; i++)
  { 
    auto pair = fit_to_binomial(good[i], ntrials);
    double ans = pair.first;
    double stddev = pair.second;
    std::cout << i << ": " << ans << " +- " << stddev << std::endl;
    if (ans - 3 * stddev < worst_3sigma)
    {
      worst_i = i;
      worst_3sigma = ans-3*stddev;
    }
  }
  auto pair = fit_to_binomial(good[worst_i], ntrials);
  double ans = pair.first;
  double stddev = pair.second;
  std::cout << "WORST" << std::endl;
  std::cout << worst_i << ": " << ans << " +- " << stddev << std::endl;

  delete[] good;  
}

int main()
{
  std::cout << CCAlgConfiguration() << std::endl;
  rand_type n = 1 << 13;
  std::cout << "TESTING: " << n << " TO " << (n*n)/4 << std::endl;
  test_n(n);
}
