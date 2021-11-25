import numpy as np
import pandas as pd
from magicce import entropy, find_error_probability

def test():
    # define inverse function of entropy function H = p*log(p)
    # find_error_probability is an approximation for small p
    # alternatively, calculate inverse numerically
    inventropyfunc = find_error_probability

    # import data as a numpy array
    data_file = './data/gene_expression.tab'
    data_df = pd.read_csv(data_file, sep='\t', index_col=0)
    all_data = data_df.values

    # define groups that are compared with each other
    # the chosen indices correspond to 2mo and 16mo brain samples
    groups = [[0,1,2,3,4],[5,6,7,8,9]]

    # find number of different conditions (same cell type/day in groups)
    # prior probabilities that a sample belongs to each group
    # Here, we are comparing one time point against another, so we take [1,1]
    # If we were comparing one time point against 3 others, we would use [1,3]
    nc = [1,1]

    H_result = entropy(all_data, groups, ncond=nc, k=10, N=1E3)
    p_result = inventropyfunc(H_result)

    df = pd.DataFrame({'H(c|x)':H_result, 'p_e':p_result}, index=data_df.index)
    df = df.sort_values(by='p_e')

    # save sorted list of promoter clusters most predictive for different groups
    writefile = './results/testresults.txt'
    df.to_csv(writefile, sep='\t', float_format='%.4e')


if __name__ == "__main__":
    test()
