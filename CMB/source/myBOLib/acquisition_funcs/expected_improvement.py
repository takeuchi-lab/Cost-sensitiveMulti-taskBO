# -*- coding: utf-8 -*-
u"""Expected Improvement."""
from __future__ import division
import numpy as np
from scipy.stats import norm


def EI_func(mu, sigma, current_max ,xi=0):
	u""" Expected Improvement !!MAXimization!! """
	I = mu - current_max
	z = (I - xi)/sigma
	ei = (I - xi) * norm.cdf(z) + sigma * norm.pdf(z)
	
	ei[ei!=ei] = 0
	ei[ei<0] = 0

	return ei
