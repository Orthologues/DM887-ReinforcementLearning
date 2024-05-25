from BDQN import *
import logging, logging.handlers
from tqdm import tqdm
from os import system as sys_cmd

if __name__ == "__main__":
    command = 'mkdir data' # Creat a direcotry to store models and scores.
    sys_cmd(command)


    env = "ALE/Breakout-v5"

    Learner = BDQN_Learner(env)

    Learner.training()