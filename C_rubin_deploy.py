'''
Testing of the Rubinstein env
The reward received should match the training reward

Example here:
https://discuss.ray.io/t/policy-rollout-on-ray-tune-2-0/8671/2

here:
https://github.com/ray-project/ray/blob/master/rllib/examples/sb2rllib_rllib_example.py

At the end here:
https://github.com/lcipolina/Ray_tutorials/blob/main/MARL_RLlib_Tutorial.ipynb
Also:
https://github.com/ray-project/ray/blob/master/rllib/examples/sb2rllib_rllib_example.py
With custom environment:
https://docs.ray.io/en/latest/rllib/rllib-training.html
'''



import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from A_rubin_env import Env #Import environment definition
from ray.rllib.agents.ppo import PPOTrainer



# retrieve the checkpoint path and create the algo
#checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial())
#https://www.oreilly.com/library/view/learning-ray/9781098117214/ch04.html


# THIS DOESNT WORK #Only in version  2.2 (or master) for this to work
# https://docs.ray.io/en/latest/rllib/rllib-training.html
#from ray.rllib.algorithms.algorithm import Algorithm
#algo = Algorithm.from_checkpoint(checkpoint_path)


#NOTE: this has to come EXACTLY from the last trained policy!!!
checkpoint_path = '/Users/lucia/ray_results/env_rubin/PPO_RubinsteinEnv_db467_00000_0_2023-01-09_19-14-36/checkpoint_000001'

# Register environment
env_name = 'RubinsteinEnv'
tune.register_env(env_name, lambda env_ctx: Env())


# https://docs.ray.io/en/latest/serve/tutorials/rllib.html#serving-rllib-models


#Same config as the training one
N_CPUS = 4
learning_rate = 1e-3
config = PPOConfig()\
    .training(lr=learning_rate,num_sgd_iter=10, train_batch_size = 4000)\
    .framework("torch")\
    .rollouts(num_rollout_workers=1, observation_filter="MeanStdFilter")\
    .resources(num_gpus=0,num_cpus_per_worker=1)\
    .evaluation(evaluation_interval=100,evaluation_duration = 2, evaluation_duration_unit='episodes',
                evaluation_config= {"explore": False})\
    .environment(env = env_name, env_config={
                                     "num_workers": N_CPUS - 1,
                                     "disable_env_checking":True} #env_config: arguments passed to the Env + num_workers = # number of parallel workers
                 )

# Build the Algorithm instance using the config.
algo = config.build(env=env_name)
# Restore the algo's state from the checkpoint.
algo.restore(checkpoint_path)
print(f"Agent loaded from saved model at {checkpoint_path}")


test_agent = config.build(env=env_name)
# Restore the algo's state from the checkpoint.
test_agent.restore(checkpoint_path)

# ****************** EVALUATION *********************
print('ENTERING EVALUATION')
test_agent.evaluate() #If we want to run an evaluation


# ****************** INFERENCE *********************
'''
print('ENTERING INFERENCE')
# Instantiate and reset environment class
env = Env()
obs = env.reset()
episode_reward = 0
done           = False
while not done:
        # ONE WAY OF DOING THIS: like the single-action
        #a1 = test_agent.compute_action(obs[0]) #brings a single action
        #a2 = test_agent.compute_action(obs["agent2"], policy_id="policy2") #receives a single action
        #actions_dict = {"0": a1, "agent2": a2}
        #actions_dict = {"0": a1}
        ## Step environment forward one more step.
        #obs, reward, terminated, info = env.step(actions_dict)
        #episode_reward += reward
        # print('REWARD', reward)

        # ALTERNATIVE WAY - compute directly from Dicts
        actions_dict = test_agent.compute_actions(obs) #receives an array of OBS
        obs, reward, dones, _ = env.step(actions_dict) #Note: every return is a dict
'''