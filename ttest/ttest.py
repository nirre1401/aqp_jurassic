
from scipy.stats import ttest_ind
'''dataset name LR # Epochs Jurrasic Hunch SS 10 SS 20 SS 30 Jurrasic Hunch SS 10 SS 20 SS 30
1 Bank 0.03 50 0.083 0.043 0.540 0.541 0.501 1.422 1.672 NA NA NA
2 Car 0.03 50 0.045 0.523 0.076 0.060 0.001 0.481 0.512 NA NA NA
3 Elections 0.06 60 0.069 0.224 0.219 0.322 0.270 1.738 1.812 NA NA NA
4 Employees 0.03 50 0.106 0.021 0.589 0.270 0.338 0.191 0.101 NA NA NA
5Healthcare Stroke0.03 50 0.172 0.031 1.284 0.934 0.645 0.440 0.512 NA NA NA
6Hotel Reservation0.03 60 0.175 0.050 0.229 0.179 0.172 0.657 0.612 NA NA NA
7 Pizza 0.03 50 0.399 0.405 0.413 0.415 0.413 0.043 0.071 NA NA NA
8 Toy 0.06 50 0.091 0.082 0.460 0.462 0.461 1.312 1.112 NA NA NA
9 Walmart 0.06 20 0.075 0.078 0.028 0.017 0.016 1.702 1.601 NA NA NA
'''

Jurrasic_nrmse = [0.083,0.0045,0.0069,0.0106,0.0172,0.0175,0.0399,0.0091, 0.0075]
hunch_nrmse = [0.043,0.523,0.224,0.106,0.021,0.031,0.050,0.405,0.082, 0.078]

Jurrasic_valloss = [0.422,0.481,1.738,0.191,0.440,0.657,0.043,0.312, .1702]
hunch_valloss = [1.672  ,0.512,1.812,1.812,0.512,0.612  ,0.071,1.112,1.601]
print(ttest_ind(Jurrasic_valloss, hunch_valloss))