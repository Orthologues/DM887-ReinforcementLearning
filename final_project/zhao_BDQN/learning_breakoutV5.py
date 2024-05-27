from BDQN import *

def learn_breakoutV5():
    total_rewards = 0
    list_total_t_steps = []
    list_total_non_warmup_steps = []
    list_total_gd_t_steps = []
    list_gd_t_step_at_eval = [] 
    list_mean_eval_reward = []

    config = Config(env_name='ALE/Breakout-v5')
    agent = BDQNAgent(config)
    
    for eps in range(1, config.num_training_episodes+1):
        agent.train_one_episode()
        total_rewards += agent.episodal_reward
        list_total_t_steps.append(agent.total_t_steps)
        list_total_non_warmup_steps.append(agent.total_t_steps - config.num_warmup_t_steps)
        list_total_gd_t_steps.append(agent.total_t_steps)

        if eps % config.num_training_episodes_per_eval == 0:
            agent.run_eval_mode() #TODO