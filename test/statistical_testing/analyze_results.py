
import numpy as np
import argparse
from scipy.stats import ttest_ind

def check_greater_error(test_name, test_result_file, expected_result_file, confidence):
    print('::::: ', test_name, ' :::::', sep='')
    test_file = open(test_result_file)
    test_result = np.loadtxt(test_file)

    test_file = open(expected_result_file)
    test_expect = np.loadtxt(test_file)

    r_t = test_result.transpose()
    e_t = test_expect.transpose()

    # get the failure values
    test_failure   = r_t[0,:]
    expect_failure = e_t[0,:]
    # get the number of tests run
    test_runs   = r_t[1,:]
    expect_runs = e_t[1,:]

    assert test_result.shape == test_expect.shape, "Must run the same number of trials"
    assert (test_runs == expect_runs).any(), "Samples must have the same number of runs per trial"

    # Null Hypothesis:        There is no difference between the failure rates of these two test runs
    # Alternative Hypothesis: The failure rate of the current test is GREATER than that of what we expect
    # Our confidence level:   If the p-value is less than than this value, we have found good evidence that test_failure > expect_failure
    t_val, p_val = ttest_ind(test_failure, expect_failure, equal_var=True, alternative='greater')
    
    if p_val <= confidence:
        print('Deviation Found')
        print('Result', test_failure)
        print('Expect', expect_failure)
        print('t-value:', t_val, 'p-value:', p_val)
        return True
    print('No Deviation Found')
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Statistical testing on graph tests.')
    parser.add_argument('tiny', metavar="tiny_graph_output", type=str,
            help='the file which contains the results from the tiny graph test')
    parser.add_argument('small', metavar="small_graph_output", type=str,
            help='the file which contains the results from the small graph test')
    parser.add_argument('medium', metavar="medium_graph_output", type=str,
            help='the file which contains the results from the medium graph test')
    parser.add_argument('large', metavar="large_graph_output", type=str,
            help='the file which contains the results from the large graph test')
    
    parser.add_argument('tiny_exp', metavar="tiny_graph_expected", type=str,
            help="the file which contains the results from a correct branch for tiny graph")
    parser.add_argument('small_exp', metavar="small_graph_expected", type=str,
            help="the file which contains the results from a correct branch for small graph")
    parser.add_argument('medium_exp', metavar="medium_graph_expected", type=str,
            help="the file which contains the results from a correct branch for medium graph")
    parser.add_argument('large_exp', metavar="large_graph_expected", type=str,
            help="the file which contains the results from a correct branch for large graph")
    args = parser.parse_args()

    if check_greater_error("tiny_test", args.tiny, args.tiny_exp, 0.1):
        print("Found Error")
    if check_greater_error("small_test", args.small, args.small_exp, 0.1):
        print("Found Error")
    if check_greater_error("medium_test", args.medium, args.medium_exp, 0.1):
        print("Found Error")
    if check_greater_error("large_test", args.large, args.large_exp, 0.1):
        print("Found Error")
