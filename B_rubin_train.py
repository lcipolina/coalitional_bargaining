'''Trains a custom env with Ray 2.0 (new way)
   New documentation (Sept 2022):
   Train: https://docs.ray.io/en/latest/rllib/rllib-env.html
   PPO Config: https://github.com/ray-project/ray/blob/2189552ada299a23c8dabf49ef10f4a9ecad68ca/rllib/algorithms/ppo/ppo.py#L47

   # buena guia aca (codigo viejo)
   #https://github.com/vermashresth/MARL-medium/blob/master/train_RLlib.py

   # Latest Ray requires pip install tensorflow-probability==0.17.0
'''

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray import air, tune
#from ray.air import session
#from ray.tune.logger import DEFAULT_LOGGERS
#from ray.air.integrations.wandb import WandbLoggerCallback)
#from ray.air.callbacks.wandb import WandbLoggerCallback



#Import environment definition
from A_rubin_env import Env


#*************************************
# Driver code for training
#*************************************
def setup_and_train():
    ''' Registers Env and sets the dict for PPO
    '''

    #**********************************************************
    # Define configuration with hyperparam and training details
    #**********************************************************
    env_name = 'RubinsteinEnv'
    tune.register_env(env_name, lambda env_ctx: Env()) #the register_env needs a callable/iterable

    #TODO: study the PPO configs very well (see OneNote)

    #TODO: https://docs.ray.io/en/latest/rllib/rllib-training.html?highlight=observation_filter#common-parameters
    # see there how to pick the best result
    # config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

    # 'MeanStdFilter' centers around the mean

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



    #***********************************
    # Initialize ray and run
    #***********************************
    if ray.is_initialized(): ray.shutdown()
    ray.init(include_dashboard=False, ignore_reinit_error=True,) #If True, prints the dashboard running on a local port
    #ray.init(local_mode=True) # it might run faster, used to debug


    #Run with Tune and Air (newest method) - note, no need to Loop over
    train_steps = 50000
    experiment_name = 'env_rubin'

    tuner = tune.Tuner("PPO", param_space=config.to_dict(), #to run with Tune
                              run_config=air.RunConfig(
                              name =  experiment_name,
                              stop={"timesteps_total": train_steps}, #if I delete this, it runs forever
                              #verbose = 2,
                              checkpoint_config=air.CheckpointConfig(
                              checkpoint_frequency=50, checkpoint_at_end=True
                                 )
                                  )
                                   )

    #TODO: run with a hyperparameter tuner, see good example here:
    # https://docs.ray.io/en/latest/tune/examples/pbt_ppo_example.html
    # https://docs.ray.io/en/latest/tune/examples/includes/pb2_ppo_example.html

    results = tuner.fit()

   
    info = results.get_best_result().metrics["info"]
    print("Info", info)


    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # Get the best checkpoint corresponding to the best result and save the checkpoint path
    best_checkpoint = best_result.checkpoint #returns a folder path, not a file.
    print(f"Trained model saved at {best_checkpoint}")



    ray.shutdown()
    return best_checkpoint




if __name__=='__main__':
    checkpoint_path = setup_and_train()
