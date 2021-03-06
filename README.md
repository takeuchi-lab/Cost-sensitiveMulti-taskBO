# Cost-sensitiveMulti-taskBO
Knowledge-Transfer-Based Cost-Effective Search for Interface Structures

<div align="center">
<img src="figs/fig.png" width="500px">
</div>

## Abstract

Determining the atomic configuration of an interface is one of the most important issues in materials science research. Although theoretical simulations are effective tools, an exhaustive search is computationally prohibitive due to the high degrees of freedom of the interface structure. In the interface structure search, multiple energy surfaces created by a variety of orientation angles need to be explored, and the necessary computational costs for different angles vary substantially owing to significant variations in the supercell sizes. To overcome this difficulty, we introduce two machine-learning concepts, transfer learning and cost-sensitive search, to our interface-structure search mehod called cost-sensitive multi-task Bayesian optimization (CMB).

The basic idea of transfer learning is to transfer knowledge among different (but related) tasks to improve the efficiency of machine-learning methods. A grain boundary (GB) structure search for a fixed angle is considered to be a ``task''. When a set of tasks are similar to each other, information accumulated for one specific task can be useful for other tasks. In our structure-search problem, a sampled GB model for an angle provides information for other angles because of the energy-surface similarity. Futher, CMB incorporates cost information into the sampling decision, meaning that we evaluate each candidate based on both the possibility of an energy improvement and the cost of sampling. By combining the cost-sensitive search with transfer learning, CMB accumulates information by sampling low cost surfaces in the initial stage of the search, and can identify the stable structures in high cost surfaces with a small number of sampling steps by using the transferred surface information.

The code currently available here is to reproduce our results on fcc-Al [110] tilt grain boundary. We are planing to update our code so that it can apply to other systems for practical use.

## Environmental Requirement
- Linux, MACOS
- Python version 2.7.15
- Please install required packages when the python "ImportError" occurs

## Usage
`python run_GBopt.py [-h] [-c] [-l] opt_type exp_name itr init_No`

### Option
- -h : display help for arguments
- -c : use cost-sensitive acquisition function
- -l : use log-scaled energy
- opt_type : choose type of optimization from MB, SB, or RAND (multi-task Bayes, single-task Bayes, random)
- exp_name : directory name under which log-files are saved
- itr : number of BO loop
- init_No : random seed (non-negative integer) for the initial point selection (1 sample per task)


### Example
`python run_GBopt.py -c -l MB Exp1 500 1`

## Lisence
GNU General Public License

## Data format

### 1. Directory

```
root/
    |- Data/
    |    |- gbdata/
    |           |- 00_020_foo.pickle
    |           |- 00_020_bar.pickle
    |           |-  ...
    |
    |- source/...
    |- log/
    |- run_GBopt.py
```

### 2. Input data 

#### Pickle files under "gbdata/" directory

This directory contains a set of input data files which have the following format of the name:

(TaskNumber)\_(RotationAngle)\_(FileName).pickle

- TaskNumber: identifier of each task. Files which have the same TaskNumber are regarded as one common task.  
- RotationAngle: GB rotation angle used as task-specific descriptor.  
- FileName: arbitrary file name.  

Each pickle file should contain a python dictionary variable with the following keys:

- "input_des": structure-specific descriptor  
             (n x p) numpy array, where n is the number of samples in the individual pickle file and p is the dimension of the structure-specific descriptor.

- "Ene": GB Energy  
       n dimensional numpy array

- "task_des": task-specific descriptor  
            q dimensional numpy array, where q is the dimension of the task-specific descriptor  
        
- "cost": observation cost  
        scalar float variable  


### 3. Output data

Under "log/" directory, run_GBopt.py creates a directory having a name specified by "exp_name" of the "run_GBopt.py" argument. Under the directory named by each method (e.g., CMB_log), the "log_exp[init_No].pickle" file contains dictionary variable with the following keys:

- "regret": simple regret[eV/A^2] in each iteration  
        (T x K) numpy array where T is the number of iterations and K is the number of tasks  

- "cum_cost": cumulative cost at each iteration  
        T dimensional numpy array  
    
- "train_num": the number of samples at each iteration  
        (T x K) numpy array  
    
- "cost_list": sampling cost of each task  
        K dimensional numpy array  
    
- "selected_p": sample ID selected by each iteration  
        (T x 2) numpy array [p_1,p_2,p_3,...,p_T], where p_t = [task ID selected at the t-th iteration, sample ID selected at the t-th iteration]
