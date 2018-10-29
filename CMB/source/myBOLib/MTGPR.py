"""
Definition of Multi-task Gaussian Process Regression

26/10/2018
Tomohiro Yonezu 
"""
import numpy as np
import scipy.linalg as splinalg
import copy
import kernel_funcs as kf
import mean_funcs as mf
import GPR
import sys 
from matplotlib import pyplot as plt
from numba import jit
from scipy.spatial import distance

class DBLZs_MTGPRegression(object):
	"""
	1. get Covariance matrix by using kernel class
	2. make predictibe distribution from 1
	"""

	def __init__(self, input_kernel=kf.RBF(), task_kernel01=kf.RBF(),task_kernel02=kf.RBF(), mean=mf.Const(), input_dim=-1, task_dim01=-1,task_dim02=-1):
		self.input_kernel = copy.copy(input_kernel)
		self.task_kernel01 = copy.copy(task_kernel01)
		self.task_kernel02 = copy.copy(task_kernel02)
		self.mean = copy.copy(mean)
		self.Z01_dim = task_dim01
		self.Z02_dim = task_dim02
		self.X_dim = input_dim
		self.task_num = 0

		self.CovM = []

		self.task_descriptors01 = [] 
		self.task_descriptors02 = [] 
		self.gps = []
		self.task_cost = []
		self.pred_dist = []
		self.N = []
		
		self.name = "Multi-task Gaussian Process"
		
	def add_objFunc(self, name=None, allX=None, allY=None, trainID=None, task_descriptor01=None, task_descriptor02=None, cost=1):
		##### add objective function #####
		self.task_num += 1

		##### make new GPR instance #####
		new_gpr = GPR.GPRegression(allX = allX, allY=allY, input_dim=self.X_dim)

		##### set trainID #####
		new_gpr.trainID = trainID

		if name is None:
			name = "func_"+str(self.task_num)
		new_gpr.name = name

		##### set cost #####
		new_gpr.cost = cost

		##### check input size #####
		n,d = np.shape(np.atleast_2d(allX))
		if d != self.X_dim:
			n,d = d,n
			allX = allX.T
		new_gpr.input_dim = self.X_dim


		##### check args #####
		if (task_descriptor01 is None) or (task_descriptor02 is None):
			print("Error [add_objFunc]: Task descriptor is necessary")
			return
		if (allX is None) or (allY is None):
			print("Error [add_objFunc]: test point or function value is empty")
			return 

		self.task_descriptors01.append(np.atleast_2d(task_descriptor01)) # add 1st task_descriptor
		self.task_descriptors02.append(np.atleast_2d(task_descriptor02)) # add 2nd task_descriptor
		self.gps.append(new_gpr)
		
		print("Added function below")
		self.print_FuncInfo()

		return True


	def print_FuncInfo(self, k=None):
		if k is None:
			k =self.task_num-1

		if k > self.task_num:
			print("Error!! There is not function No."+str(k)+" yet.")
			return False

		print("=====[function No."+str("%2d"%k)+"]===========")
		print("| - name       : "+(self.gps[k]).name)
		print("| - input size : "+str(self.gps[k].N))
		print("| - #training  : "+str(np.shape(np.atleast_1d(self.gps[k].trainID))[0]))
		print("| - cost       : "+str(self.gps[k].cost))
		print("================================")

		return True


	def predict(self,full=False, predictAT=None):
		##### calcrate predictive distribution at test point #####
		epsilon = -6

		##### set trainID #####
		all_trainID = np.atleast_2d(self.gps[0].trainID)
		sn = self.gps[0].N
		for k in range(1,self.task_num):
			all_trainID = np.c_[all_trainID,np.atleast_2d(self.gps[k].trainID)+sn]
			sn += self.gps[k].N

		all_trainID = (np.sort(np.array(all_trainID)[0,:])).astype("int64")

		##### prepare all Ys #####
		allY = np.copy(np.atleast_2d(self.gps[0].allY))
		for k in range(1,self.task_num):
			allY = np.c_[allY,np.atleast_2d(self.gps[k].allY)]

		allY = allY[0,:]
		n = np.shape(allY)[0]

		##### prepare all 1st Zs #####
		allZ01 = np.copy(np.repeat(np.atleast_2d(self.task_descriptors01[0]),self.gps[0].N,axis=0))
		for k in range(1,self.task_num):
			allZ01 = np.r_[allZ01,np.repeat(np.atleast_2d(self.task_descriptors01[k]),self.gps[k].N,axis=0)]

		##### prepare all 1st Zs #####
		allZ02 = np.copy(np.repeat(np.atleast_2d(self.task_descriptors02[0]),self.gps[0].N,axis=0))
		for k in range(1,self.task_num):
			allZ02 = np.r_[allZ02,np.repeat(np.atleast_2d(self.task_descriptors02[k]),self.gps[k].N,axis=0)]

		##### prepaer all Xs #####
		allX = np.copy(self.gps[0].allX)
		for k in range(1,self.task_num):
			allX = np.r_[allX,self.gps[k].allX]

		##### if there is no training points, return prior distribution #####
		if np.shape(np.atleast_1d(all_trainID))==0:
			print("there is No training point")
			return -1
		else:
			##### function value at training point #####
			trainY = allY[all_trainID] # set function value at training point #
			trainY_til = trainY - self.mean.getMean(trainID=all_trainID, const=np.mean(trainY)) # gap between prior mean #

		##### make Kernel matrix about training point #####
		tri_K = self.task_kernel01.getCovMat(self.Z01_dim, (allZ01[all_trainID,:]), allZ01,"train") \
					* self.task_kernel02.getCovMat(self.Z02_dim, (allZ02[all_trainID,:]), allZ02,"train") \
					* self.input_kernel.getCovMat(self.X_dim, (allX[all_trainID,:]), allX,"train")

		it = 0
		while(1):
			try:
				alpha = np.linalg.solve(tri_K, trainY_til)
				break
			except np.linalg.LinAlgError:
				it += 1
				tri_K += (10**(epsilon))*np.eye(np.shape(np.atleast_1d(all_trainID))[0])


		if predictAT is not None:

			if predictAT == "hallc":

				pred_ID = np.atleast_2d(self.gps[0].hal_trainID)
				sn = self.gps[0].N
				for k in range(1,self.task_num):
					pred_ID = np.c_[pred_ID,np.atleast_2d(self.gps[k].hal_trainID)+sn]
					sn += self.gps[k].N

				pred_ID = (np.sort(np.array(pred_ID)[0,:])).astype("int64")
				#print(pred_ID)

				test_K = self.task_kernel01.getCovMat(self.Z01_dim, (allZ01[all_trainID,:]), allZ01[pred_ID,:],"test") \
						* self.task_kernel02.getCovMat(self.Z02_dim, (allZ02[all_trainID,:]), allZ02[pred_ID,:],"test") \
						* self.input_kernel.getCovMat(self.X_dim, (allX[all_trainID,:]), allX[pred_ID,:],"test")

				#plt.imshow(test_K)
				#plt.show()

				k = self.task_kernel01.getCovMat(self.Z01_dim, (allZ01[all_trainID,:]), allZ01[pred_ID,:],"cross") \
						* self.task_kernel02.getCovMat(self.Z02_dim, (allZ02[all_trainID,:]), allZ02[pred_ID,:],"cross") \
						* self.input_kernel.getCovMat(self.X_dim, (allX[all_trainID,:]), allX[pred_ID,:],"cross")

				mean = (k.T).dot(alpha) + self.mean.getMean(allY[pred_ID], np.mean(trainY))
				cov = test_K - (k.T).dot(np.linalg.solve(tri_K, k))
				var = np.diag(cov)


		else:
			batchsize = 10
			ns = 0
			nact = n / batchsize
			mean = np.zeros(n)
			var = np.zeros(n)
			cov = np.nan
			
			while ns <= nact:
				act = np.array(range(ns * batchsize, np.minimum((ns + 1) * batchsize, n))).astype("int64")

				test_K = np.diag(self.task_kernel01.getCovMat(self.Z01_dim, (allZ01[all_trainID,:]), allZ01[act,:],"test") \
					* self.task_kernel02.getCovMat(self.Z02_dim, (allZ02[all_trainID,:]), allZ02[act,:],"test") \
					* self.input_kernel.getCovMat(self.X_dim, (allX[all_trainID,:]), allX[act,:],"test"))

				k = self.task_kernel01.getCovMat(self.Z01_dim, (allZ01[all_trainID,:]), allZ01[act,:],"cross") \
					* self.task_kernel02.getCovMat(self.Z02_dim, (allZ02[all_trainID,:]), allZ02[act,:],"cross") \
					* self.input_kernel.getCovMat(self.X_dim, (allX[all_trainID,:]), allX[act,:],"cross")
				
				mean[act] = (k.T).dot(alpha) + self.mean.getMean(trainID=act, const=np.mean(trainY))
				var[act] = test_K - np.diag((k.T).dot(np.linalg.solve(tri_K, k)))

				ns += 1

			var[var<epsilon] = epsilon

			sn = 0
			for k in range(self.task_num):
				each_var = var[sn:(sn+self.gps[k].N)]				
				each_mean = mean[sn:(sn+self.gps[k].N)]
				each_cov = np.nan
				
				each_dist = {"mean":each_mean,"var":each_var,"cov":each_cov} 
				self.gps[k].pred_dist = each_dist
				sn += self.gps[k].N

		self.pred_dist = {"mean":mean,"var":var,"cov":cov}

		##### return predictive distribution #####
		return (self.pred_dist)

	










