from __future__ import division
from matplotlib import pyplot as plt
from make_init import gen_init_data
import os
import glob
import copy
import random
import pickle
import argparse
import numpy as np
import myBOLib as bol

def SB_GBopt(exp_name, itr, init_No, cost, logscale):

	kf = bol.kernel_funcs
	af = bol.acquisition_funcs

	##### output regret log #####
	sep = os.sep
	if cost:
		outDIR = "."+sep+"log"+sep+exp_name+sep+"CSB_log"
	else:
		outDIR = "."+sep+"log"+sep+exp_name+sep+"SB_log"

	if not os.path.exists("."+sep+"log"+sep+exp_name):
	    os.mkdir("."+sep+"log"+sep+exp_name)

	if not os.path.exists(outDIR):
	    os.mkdir(outDIR)


	##### make kernel instance #####
	input_kernel = kf.RBF(1,49700.0) # median
	task_kernel01 = kf.RBF(1,50.472668979473895) # median 
	task_kernel02 = kf.RBF(1,1225.0) # median

	input_kernel.name = "input_kernel:RBF"
	task_kernel01.name = "task_kernel01:RBF"
	task_kernel02.name = "task_kernel02:RBF"

	##### load GB datas #####
	GB_list =  np.sort(glob.glob("."+sep+"Data"+sep+"gbdata"+sep+"*"))
	print("[Load GB files from ]")
	print("   ."+sep+"Data"+sep+"gbdata"+sep+"*")

	K = np.shape(GB_list)[0]
	init_data = gen_init_data(datapath="."+sep+"Data"+sep+"gbdata"+sep,s=init_No)
	print("[ID of initial training points]")
	print(init_data)

	##### get gbIDs #####
	gbID = np.zeros(K)
	for k in range(K):
		gbID[k] = int(((((GB_list[k]).split("/")[-1]).split(".")[0]).split("_"))[0])

	Khat = np.shape(np.unique(gbID))[0]
	gbID = gbID.astype("int64")

	tmpID = np.zeros(K)
	for i,ID in enumerate(np.unique(gbID)):
		tmpID[gbID == ID] = i
	gbID = tmpID.astype("int64")

	task_dis_tmp = []

	##### generate multi-multi-GPR lol #####
	mtgps = []
	for kh in range(Khat):

		new_mtgpr  = bol.MTGPR.DBLZs_MTGPRegression(input_kernel=input_kernel,task_kernel01=task_kernel01,task_kernel02=task_kernel02,input_dim=3,task_dim01=100,task_dim02=1) # GPR instance 
		mtgps.append(new_mtgpr)

	gps = []
	for k in range(K):

		##### open pickle file #####
		with open(GB_list[k],"rb") as f:
			gb = pickle.load(f)

		name = ((GB_list[k]).split(sep)[-1])
		angle = 90-abs(90-int(((((GB_list[k]).split(sep)[-1]).split(".")[0]).split("_"))[1]))
		
		X = gb["input_des"]
		Y = gb["Ene"]

		not_NaN = (Y==Y)
		X = X[not_NaN,:]
		Y = Y[not_NaN]

		if logscale:
			OBS = -np.log(Y)
		else: 
			OBS = -Y

		init = init_data[k]
		if init < 0:
			init = []


		mtgps[gbID[k]].add_objFunc(name=name, allX=X, allY=OBS, trainID=init, task_descriptor01=(gb["task_des"]), task_descriptor02=angle, cost=gb["cost"])
		mtgps[gbID[k]].gps[-1].org_Y = copy.copy(OBS)
		mtgps[gbID[k]].gps[-1].ForRegret = copy.copy(-Y)


		task_dis_tmp.append((gb["task_des"]))
		gps.append(mtgps[gbID[k]].gps[-1])

	##### make cost list #####
	cost_list = np.zeros(Khat)
	for k in range(K):
		cost_list[gbID[k]] += gps[k].cost
	for kh in range(Khat):
		cost_list[kh] = cost_list[kh]/np.shape(np.where(np.array(gbID) == kh))[1]

	##### each true max #####
	true_max = -np.inf*np.ones(Khat)
	for k in range(K):
		if true_max[gbID[k]] < max(gps[k].ForRegret):
			true_max[gbID[k]] = max(gps[k].ForRegret)

	##### MT Gaussian Process Optimization #####
	T = itr

	##### prepare log #####
	r_t = np.inf*np.ones((Khat,1))
	N_t = np.zeros((Khat,1))
	selected_p = []
	cum_cost = [0]
	regret_plot = []

	##### Bayesian optimization from here #####
	print("[Single-task Bayesian optimization T="+str(T)+"]")

	for t in (range(T)):
		print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

		##### simple regret #####
		if t == 0:
			for k in range(K):
				#cum_cost[t] += (gps[k].cost)*np.shape(np.atleast_1d(gps[k].trainID))[0]

				if np.shape(np.atleast_1d(gps[k].trainID))[0] == 0:
					pass
				else:
					last_tri = int(np.atleast_1d(gps[k].trainID)[-1])
					r_t[gbID[k],t] = true_max[gbID[k]] - gps[k].ForRegret[last_tri]
					N_t[gbID[k],t] += np.shape(np.atleast_1d(gps[k].trainID))[0]
		else:
			r_t = np.c_[r_t,np.atleast_2d(r_t[:,-1]).T]
			N_t = np.c_[N_t,np.atleast_2d(N_t[:,-1]).T]

			r_t[gbID[next_task],-1] = true_max[gbID[next_task]] - gps[next_task].ForRegret[next_ID]
			N_t[gbID[next_task],-1] = N_t[gbID[next_task],t-1] + 1

			cum_cost.append(cum_cost[t-1] + gps[next_task].cost)

		log_set = {"regret":r_t,"cum_cost":cum_cost,"train_num":N_t,"cost_list":cost_list,"selected_p":selected_p}

		if t%2 == 0:
			with open(outDIR+sep+"log_exp"+str("%03d"%init_No)+".pickle","wb") as f:
				pickle.dump(log_set,f)
		else:
			with open(outDIR+sep+"log_exp_tmp_"+str("%03d"%init_No)+".pickle","wb") as f:
				pickle.dump(log_set,f)

		print("t="+str("%03d"%(t+1)))
		print(" total cost: "+str(cum_cost[-1]))
		print(" regret: "+str(16021*np.average(np.min(r_t,axis=1)))+" [mJ/m^2]")
		regret_plot.append(16021*np.average(np.min(r_t,axis=1)))
		
		"""
		if t > 0:
			plt.figure()
			if cost:
				plt.title("CSB")
			else:
				plt.title("SB")
			plt.grid(True)
			plt.xlabel("total cost")
			plt.ylabel("GB energy gap from the minimum "+r"$[{\rm mJ}/{\rm m}^2]$")
			plt.ylim([0,1.1*np.max(regret_plot)])
			#plt.xlim([0,100000])
			plt.plot(cum_cost,regret_plot,color="#2E64FE")
			plt.pause(0.01)
			plt.close()
		"""

		##### MTGP Regression#####
		for kh in range(Khat):
			mtgps[kh].predict()

		##### calcurate acquisition #####
		each_maxID = -np.ones(K)
		each_maxAcq = -np.ones(K)
		current_max = -np.inf*np.ones(Khat)

		##### share current_max among gbs having same gbID ##### 
		for k in range(K):
			
			tri_tmp = np.atleast_1d(gps[k].trainID)
			if np.shape(tri_tmp)[0] == 0:
				pass
			else:
				current_max[gbID[k]] = max(current_max[gbID[k]],max(gps[k].allY[(tri_tmp).astype("int64")]))

		for k in range(K):
			
			x = gps[k].allX[:,0]
			y = gps[k].allY
			trainID = (np.atleast_1d(gps[k].trainID)).astype("int64")

			mu = gps[k].pred_dist["mean"]

			sigma = (gps[k].pred_dist["var"])
			sigma[sigma < 1e-6] = 1e-6
			sigma = np.sqrt(sigma)

			acq = af.EI_func(mu,sigma,current_max[gbID[k]])
			acq[trainID] = -np.inf

			each_maxAcq[k]= max(acq)

			if cost:
				each_maxAcq[k]= max(acq)/gps[k].cost

			if np.shape(np.argmax(acq)) != ():
				each_maxID[k] = random.choice(np.argmax(acq))
			else:
				each_maxID[k] = np.argmax(acq)

		next_task = np.argmax(each_maxAcq)
		next_ID = int(each_maxID[next_task])
		selected_p.append([next_task,next_ID])

		print(" selected task: " + gps[next_task].name)
		print(" selected id: "+ str(next_ID))
		
		tmp = np.atleast_2d(gps[next_task].trainID)
		tmp = np.c_[tmp,np.atleast_2d([next_ID])]
		gps[next_task].trainID = tmp[0,:]

	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("finish!!")
	##### output regret log #####
	log_set = {"regret":r_t,"cum_cost":cum_cost,"train_num":N_t,"cost_list":cost_list,"selected_p":selected_p}

	with open(outDIR+sep+"log_exp"+str("%03d"%init_No)+".pickle","wb") as f:
		pickle.dump(log_set,f)

if __name__ == "__main__":

	SB_GBopt("test", 5, 1, True, True)





