from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import pickle
import glob
import sys
import seaborn as sns
import math
import os
import scipy.spatial.distance as dist
import argparse

def gen_init_data(datapath,n=100,testK=5,outDIR="./"):

	GB_list =  np.sort(glob.glob(datapath+"/*"))
	out_filename = (datapath.split("/"))[-1]

	if out_filename == "":
		out_filename = (datapath.split("/"))[-2]

	output = {"all_init":[],"dir_name":out_filename}

	K = np.shape(GB_list)[0]

	gbID = np.zeros(K)
	N = np.zeros(K)

	np.random.seed(0)

	##### set object functions #####
	for k in range(K):

		##### open pickle file #####
		with open(GB_list[k],"rb") as f:
			gb = pickle.load(f)
		
		gbID[k] = int(((((GB_list[k]).split("/")[-1]).split(".")[0]).split("_"))[0])
		Y = -gb["Ene"]
		N[k] = int(np.shape(gb["input_des"][Y==Y,:])[0])

	Khat = np.shape(np.unique(gbID))[0]

	tmpID = np.zeros(K)

	for i,ID in enumerate(np.unique(gbID)):
		tmpID[gbID == ID] = i

	gbID = tmpID.astype("int64")

	gbID = np.zeros(K)
	N = np.zeros(K)

	##### set object functions #####
	for k in range(K):

		##### open pickle file #####
		with open(GB_list[k],"rb") as f:
			gb = pickle.load(f)
		
		gbID[k] = int(((((GB_list[k]).split("/")[-1]).split(".")[0]).split("_"))[0])
		Y = -gb["Ene"]
		N[k] = int(np.shape(gb["input_des"][Y==Y,:])[0])

	Khat = np.shape(np.unique(gbID))[0]

	tmpID = np.zeros(K)

	for i,ID in enumerate(np.unique(gbID)):
		tmpID[gbID == ID] = i

	gbID = tmpID.astype("int64")
	
	out = -np.ones((n,K))

	for k in range(Khat):

		for i in range(n):
			t = np.random.choice(np.array(range(K))[gbID == k])
			x = np.random.choice(np.array(range(int(N[t]))))
			out[i,t] = int(x)

	output["all_init"] = out
	
	with open(outDIR+"GB_init.pickle","wb") as f:
		pickle.dump(output,f)

	print("#############################")
	print("Init-data for "+out_filename)
	print(output)
	print("#############################")

	return True


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="[Make GB-init.pickle]")
	parser.add_argument("targetDIR",help = "target directory",type=str)
	args = parser.parse_args()

	gen_init_data(datapath=args.targetDIR,n=20,testK=2)













