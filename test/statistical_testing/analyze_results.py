
import numpy as np
import argparse
from scipy.stats import ttest_ind

def check_error(test_name, test_result_file, expected_result_file):
    print('::::: ', test_name, ' :::::', sep='')
    test_file = open(test_result_file)
    test_result = np.loadtxt(test_file)

    test_file = open(expected_result_file)
    test_expect = np.loadtxt(test_file)

    result_t = test_result.transpose()
    test_failures = result_t[0,:]
    test_runs     = result_t[1,:]

    total_expect_failures = test_expect[0]
    total_expect_runs     = test_expect[1]  

    assert (test_runs == 10).all(), "Each bin must be of size 10"

    # First step:  Verify that there is not a dependency between tests and upon the graph
    if (test_failures > 4).any():
        return False, "Dependency between tests or upon input graph found"

    # Second step: Verify that the number of test failures does not deviate from the expectation
    total_test_failures = np.sum(test_failures)
    total_test_runs     = np.sum(test_runs)

    assert total_test_runs == total_expect_runs, "The number of runs must be the same"
    pr = total_expect_failures / total_expect_runs
    z_test_deviation = np.ceil(1.96 * np.sqrt(pr * (1-pr)/ total_expect_runs) * total_expect_runs)
    print("Number of test failures:", total_test_failures, "{0}%".format(total_test_failures/total_test_runs))
    print("Total number of failures is allowed to deviate by at most", z_test_deviation)
    print("Deviation is", total_test_failures - total_expect_failures)
    if total_test_failures - z_test_deviation > total_expect_failures:
        return True, "Test error is statistically greater than expectation {0}/{1}".format(int(total_test_failures), int(total_test_runs))

    if total_test_failures + z_test_deviation < total_expect_failures:
        return True, "Test error is statistically less than expectation {0}/{1}".format(int(total_test_failures), int(total_test_runs))

    return False, "No statistical deviation detected {0}/{1}".format(int(total_test_failures), int(total_test_runs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Statistical testing on graph tests.')
    parser.add_argument('small', metavar="small output", type=str,
            help='the file which contains the results from the small graph test')
    parser.add_argument('medium', metavar="medium output", type=str,
            help='the file which contains the results from the medium graph test')
    parser.add_argument('iso', metavar="medium iso output", type=str,
            help='the file which contains the results from the medium+iso graph test')

    parser.add_argument('small_exp', metavar="small expect", type=str,
            help="the file which contains the results from a correct branch for small graph")
    parser.add_argument('medium_exp', metavar="medium expect", type=str,
            help="the file which contains the results from a correct branch for medium graph")
    parser.add_argument('iso_exp', metavar="medium iso expect", type=str,
            help="the file which contains the results from a correct branch for medium+iso graph")
    args = parser.parse_args()

    stat_result = check_error("small_test", args.small, args.small_exp, 0.1)
    print(stat_result[0])
    print(stat_result[1])
    
    stat_result = check_error("medium_test", args.medium, args.medium_exp, 0.1)
    print(stat_result[0])
    print(stat_result[1])
    
    stat_result = check_error("medium_iso_test", args.iso, args.iso_exp, 0.1)
    print(stat_result[0])
    print(stat_result[1])
