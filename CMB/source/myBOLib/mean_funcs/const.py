# -*- coding: utf-8 -*-
u"""定数関数の定義."""
from __future__ import division
import numpy as np
from .base import Mymean
import copy


class Const(Mymean):
	def __init__(self, const=0):
		self.hyp = [const]
		self.name = "Const"

	def getMean(self, trainID, const=0):
		size = np.shape(np.atleast_1d(trainID))[0]
		self.hyp = [const]
		return np.array([self.hyp[0]] * size)

	def getDerMean(self):
		pass
