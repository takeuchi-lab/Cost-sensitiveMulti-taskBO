#!/usr/bin/python
#coding:utf-8

from __future__ import division
import os
import sys
import argparse
import time

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

	os.system("python ./source/MB_GBopt.py"+c+l+" "+args.exp_name+" "+str(args.itr)+" "+str(args.init_No))

elif args.opt_type == "SB":
	if args.cost:
		c = " -c"
	else:
		c = " "

	if args.logscale:
		l = " -l"
	else:
		l = " "

	os.system("python ./source/SB_GBopt.py"+c+l+" "+args.exp_name+" "+str(args.itr)+" "+str(args.init_No))

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

	os.system("python ./source/RAND_GBopt.py"+c+l+" "+args.exp_name+" "+str(args.itr)+" "+str(args.init_No))

else:
	print("please choose Optimization type from \"MB\",\"SB\" and \"RAND\"")

print("Elapsed_time: "+str(round(time.time()-start,3))+"[sec]")



