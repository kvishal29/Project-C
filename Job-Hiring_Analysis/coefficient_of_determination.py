import math
import csv
from statistics import mean
import numpy as np

################################## Functions used ################################
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig,ys_line):
	return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
##################################################################################

for kkk in range(1,57):
	############################ Making feature matrix ###########################
	# csv file name
	filename = "prosody_modified.csv"

	# initializing the titles and rows list
	feature = []
	new = []

	with open(filename,'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile)
		# extracting field names through first row
		fields = csvreader.__next__()
		# num_of_col= len(fields)
		for row in csvreader:
			feature.append(row[kkk])
	for j in range(0,len(feature)):
		feature = [float(i) for i in feature]

	############################ Making response matrix #########################
	# csv file name
	filename = "turker_scores.csv"

	# initializing the titles and rows list
	response = []

	with open(filename,'r') as csvfile:
		# creating a csv reader object
		csvreader = csv.reader(csvfile)
		# extracting field names through first row
		fields = csvreader.__next__()
		# num_of_col= len(fields)
		for row in csvreader:
				response.append(row[7])
	response = [float(i) for i in response]

	#############################################################################

	xs = np.array(feature, dtype=np.float64)
	ys = np.array(response, dtype=np.float64)
    
	m, b = best_fit_slope_and_intercept(xs,ys)
	regression_line = [(m*x)+b for x in xs]
	r_squared = coefficient_of_determination(ys,regression_line)

	print("{},".format(r_squared), end='')
