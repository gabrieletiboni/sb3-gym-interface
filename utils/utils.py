import numpy as np
import random
import string
import socket
import os
import glob
import pdb
from datetime import datetime

import gym

def get_run_name(args):
	current_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
	return str(current_date)+"_"+str(args.env)+"_"+str(args.algo)+"_t"+str(args.timesteps)+"_seed"+str(args.seed)+"_"+socket.gethostname()

def get_random_string(n=5):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def set_seed(seed):
	if seed > 0:
		np.random.seed(seed)

def create_dir(path):
	try:
		os.mkdir(os.path.join(path))
	except OSError as error:
		# print('Dir esiste già:', path)
		pass

def create_dirs(path):
	try:
		os.makedirs(os.path.join(path))
	except OSError as error:
		pass