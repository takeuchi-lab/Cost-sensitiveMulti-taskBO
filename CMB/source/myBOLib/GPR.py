"""
Definition of Gaussian Process Regression

pyGPs
Copyright(c) by 
Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014

07/12/2017 ver 1.0.0
Tomohiro Yonezu 
"""
import numpy as np
import scipy.linalg as splinalg
import copy
import kernel_funcs as kf
import mean_funcs as mf
import sys
import matplotlib.pyplot as plt

class GPRegression(object):
	"""
	1. get Covariance matrix by using kernel class
	2. make predictibe distribution from 1
	"""

	def __init__(self, allX, allY, kernel=kf.RBF(), mean=mf.Const(), input_dim=-1):
		self.kernel = copy.copy(kernel)
		self.mean = copy.copy(mean)
		self.halY = []
		self.hal_trainID = []
		self.hal_pred = []

		if input_dim != np.shape(np.atleast_2d(allX))[1]:
			print("Something wrong with input matrix. please check it.")
			print("Input dim from data : "+str(np.shape(np.atleast_2d(allX))[1]))
			print("Input dim you set : "+str(input_dim))
			sys.exit()

		self.allX = np.atleast_2d(allX)
		self.allY = allY
		self.input_dim = input_dim
		self.trainID = []
		self.pred_dist = []
		self.name = "Gaussian Process"
		self.cost = []
		self.org_Y = []
		self.ForRegret = []
		
		self.N = np.shape(allX)[0]
		
		self.average = None
		self.std = None

	def obs2ranking(self, trainID=None):
		self.allY[trainY] = np.argsort(self.org_Y[trainID])



	def predict(self, trainID=None,Hallc=False):
		""" calcrate predictive distribution at test point """
		epsilon = -6

		""" set trainID """
		if trainID is None:
			trainID = self.trainID


		""" if there is no training points, return prior distribution """
		
		trainX = self.allX[trainID,:]
		#print(trainX)
		
		""" kernel for test & training & corss """
		K = self.kernel.getCovMat(self.input_dim, trainX, self.allX,"full")
		k = self.kernel.getCovMat(self.input_dim, trainX, self.allX, "cross")
		tri_K = self.kernel.getCovMat(self.input_dim, trainX, self.allX, "train")+(10**epsilon)*np.eye(np.shape(np.atleast_1d(trainID))[0])
		
		""" function value at training point """

		if Hallc:
			trainY = self.halY[trainID]
		else:
			trainY = self.allY[trainID] # set function value at training point #

		trainY_til = trainY - self.mean.getMean(trainY, np.mean(trainY)) # gap between prior mean #
		""" posterior mean & covariance  """
		it = 0
		while(True):
			try:
				mean = (k.T).dot(np.linalg.solve(tri_K, trainY_til)) + self.mean.getMean(self.allY, np.mean(trainY))
				cov = K - (k.T).dot(np.linalg.solve(tri_K,k))
				break
			except np.linalg.LinAlgError:
				it += 1
				tri_K += (10**(epsilon))*np.eye(np.shape(trainID)[0])

		self.pred_dist = {"mean":mean,"cov":cov}

		""" return predictive distribution """
		return (self.pred_dist)

	def normalize(self, to="allY"):

		if self.average is None:
			print("Please set mean.")
			return False

		if self.std is None:
			print("Please set standard deviation")
			return False
		
		if to=="allY":
			self.allY = np.copy((self.org_Y - self.average)/self.std)
		elif to=="halY":
			self.halY = np.copy((self.org_Y - self.average)/self.std)







