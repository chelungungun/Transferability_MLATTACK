import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold


def load_data(dataset_name, folds, rng):
	X = np.loadtxt("multi_label_dataset/txtdata/" + dataset_name + 'x.txt' , delimiter=',')
	Y = np.loadtxt("multi_label_dataset/txtdata/" + dataset_name + 'y.txt' , delimiter=',')
	Y = Y.astype(int)
	Y[Y < 0] = 0
	unlabeled = np.where(Y.sum(-1) < 0)[0]
	print("Features shape = {}".format(X.shape))
	print("Label shape = {}".format(Y.shape))
	cv = []
	kf = KFold(n_splits=folds, shuffle=True, random_state=rng)
	for train, test in kf.split(X):
		cv.append(test.tolist())
	try:
		return (unlabeled, cv, csr_matrix(X), csr_matrix(Y))
	except:
		return (unlabeled, cv, X, Y)
