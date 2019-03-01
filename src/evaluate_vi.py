import sys
if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")
import os 

import numpy as np
import scipy as sp

from collections import Counter
from itertools import count
from collections import defaultdict as ddict

from sklearn.metrics import mutual_info_score as mi

import argparse

def vi(true_labels, predicted_labels):
	def convert_to_1d_labels(labels):
		ecount = count()
		enames = ddict(ecount.__next__)
		return [enames[i] for i in labels]

	true_labels = convert_to_1d_labels(true_labels)
	predicted_labels = convert_to_1d_labels(predicted_labels)

	true_probs = np.array(list(Counter(true_labels).values()))
	predicted_probs = np.array(list(Counter(predicted_labels).values()))

	true_probs = true_probs / true_probs.sum()
	predicted_probs = predicted_probs / predicted_probs.sum()

	return sp.stats.entropy(true_probs) + sp.stats.entropy(predicted_probs) - 2 * mi(true_labels, predicted_labels)

def parse_args():
	pass

def main():
	true_labels = [(1,0), (1,0), (1,1), (1,1), (1,0)]
	predicted_labels = [(1,1), (1,1), (1,1), (1,1), (1,0)]

	print (vi(true_labels, predicted_labels))

if __name__ == "__main__":
	main()