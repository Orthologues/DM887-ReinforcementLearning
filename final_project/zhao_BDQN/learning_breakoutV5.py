from BDQN import *




def learn_breakoutV5():
    config = Config(env_name='ALE/Breakout-v5')
    agent = BDQN_agent(config)