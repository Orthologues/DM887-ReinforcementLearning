from BDQN import *

def learn_breakoutV5():

    config = Config(env_name='ALE/Breakout-v5')
    agent = BDQNAgent(config)
    
    for eps in range(config.num_training_episodes):
        agent.train_one_episode()
        agent.save_training_record(eps+1, 'Breakout-v5_train')
        if (eps+1) % config.num_training_episodes_per_eval == 0:
            print(f"Saving model at episode {eps+1}...")
            agent.save_model(eps+1, "/content/Breakout-v5")
            agent.run_eval_mode(eps+1, "Breakout-v5_eval")

def learn_pongV5():

    config = Config(env_name='ALE/Pong-v5')
    agent = BDQNAgent(config)
    
    for eps in range(config.num_training_episodes):
        agent.train_one_episode()
        agent.save_training_record(eps+1, 'Pong-v5_train')
        if (eps+1) % config.num_training_episodes_per_eval == 0:
            print(f"Saving model at episode {eps+1}...")
            agent.save_model(eps+1, "/content/Pong-v5")
            agent.run_eval_mode(eps+1, "Pong-v5_eval")

def learn_tennisV5():

    config = Config(env_name='ALE/Tennis-v5')
    agent = BDQNAgent(config)
    
    for eps in range(config.num_training_episodes):
        agent.train_one_episode()
        agent.save_training_record(eps+1, 'Tennis-v5_train')
        if (eps+1) % config.num_training_episodes_per_eval == 0:
            print(f"Saving model at episode {eps+1}...")
            agent.save_model(eps+1, "/content/Tennis-v5")
            agent.run_eval_mode(eps+1, "Tennis-v5_eval")


if __name__ == "__main__":
    learn_tennisV5()
    learn_pongV5()
    learn_breakoutV5()