from __future__ import division
import numpy as np
import pickle
import glob
import sys
import math
import os
import scipy.spatial.distance as dist
import argparse

def gen_init_data(datapath,s):

	##### set random seed #####
	np.random.seed(seed=s)


	##### set file path #####
	sep = os.sep
	GB_list =  np.sort(glob.glob(datapath+sep+"*"))


	##### number of files #####
	K = np.shape(GB_list)[0]


	##### prepare variable for each file ####
	gbID = np.zeros(K) # file ID
	N = np.zeros(K) # number of candidate


	##### set variable #####
	for k in range(K):

		##### open pickle file #####
		with open(GB_list[k],"rb") as f:
			gb = pickle.load(f)
		
		##### load file ID and number of candidate #####
		gbID[k] = int(((((GB_list[k]).split(sep)[-1]).split(".")[0]).split("_"))[0])
		Y = -gb["Ene"]
		not_NaN = (Y==Y)
		N[k] = int(np.shape(gb["input_des"][not_NaN,:])[0])

	##### number of grain boundary #####
	Khat = np.shape(np.unique(gbID))[0]
	gbID = gbID.astype("int64")
	
	out = -np.ones(K)

	for kh in range(Khat):
		t = np.random.choice(np.array(range(K))[gbID == kh])
		x = np.random.choice(np.array(range(int(N[t]))))
		out[t] = int(x)

	return out.astype("int64")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="[Make GB-init.pickle]")
	parser.add_argument("targetDIR",help = "target directory",type=str)
	args = parser.parse_args()

	print(gen_init_data(datapath=args.targetDIR,s=100))














