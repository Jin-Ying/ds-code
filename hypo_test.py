from scipy.stats import ttest_1samp

example_distribution = [159, 280, 101,212,224,379,179,264,222,362,168,250,149,260,485,170]
expected_mean = 225
example_distribution_2 = [310,320,310,300,290]
t_stat, p_val = ttest_1samp(example_distribution_2, expected_mean,)
print(t_stat)
print(p_val)