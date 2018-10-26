#!/usr/bin/python
#coding:utf-8

from __future__ import division
import os
import sys
import argparse
import time
from source import MB_GBopt
from source import RAND_GBopt
from source import SB_GBopt

##### argment #####
parser = argparse.ArgumentParser(description="[Code of Bayesian optimization for GB structure]")
parser.add_argument("opt_type",help = "choose type of optimization from MB,SB,RAND",type=str)
parser.add_argument("exp_name",help = "Name of Experiment",type=str)
parser.add_argument("itr",help = "# of Iteration",type=int)
parser.add_argument("init_No",help = "position of initial training point. 0~99",type=int)
parser.add_argument("-c","--cost",help="cost-sensitive or not",action="store_true")
parser.add_argument("-l","--logscale",help="log-scale GB-Energy or not",action="store_true")
args = parser.parse_args()

start = time.time()

if not os.path.exists("log"):
    os.mkdir("log")

if args.opt_type == "MB":
	if args.cost:
		c = " -c"
	else:
		c = " "

	if args.logscale:
		l = " -l"
	else:
		l = " "

        MB_GBopt.MB_GBopt(args.exp_name, args.itr, args.init_No, c, l)

elif args.opt_type == "SB":
	if args.cost:
		c = " -c"
	else:
		c = " "

	if args.logscale:
		l = " -l"
	else:
		l = " "

        SB_GBopt.SB_GBopt(args.exp_name, args.itr, args.init_No, c, l)

elif args.opt_type == "RAND":
	if args.cost:
		print("This model doesnt have option \"-cost\" ")
		sys.exit()
	else:
		c = " "

	if args.logscale:
		l = " -l"
	else:
		l = " "

        RAND_GBopt.RAND_GBopt(args.exp_name, args.itr, args.init_No, c, l)

else:
	print("please choose Optimization type from \"MB\",\"SB\" and \"RAND\"")

print("Elapsed_time: "+str(round(time.time()-start,3))+"[sec]")



