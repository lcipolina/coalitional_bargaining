'''
Ray documentation:
https://github.com/ray-project/ray/blob/master/rllib/env/multi_agent_env.py
'''

'''
Environment for a Okada bargaining game
:inputs:
    the agent's actions are the bid
    the other agents actions are the accept/reject
:output: environment step (state, reward)

OKADA
Stochastic turn-based game
Alternating offers—the first player makes an offer in the first period,
if the second player rejects, the game moves to the second period in which the second player makes an offer,
if the first rejects, the game moves to the third period, and so forth.
'''


#Remember: conda activate marl

from typing import Dict
import random
import numpy as np
from gym.spaces import Box,Discrete #Box env helps Ray understand that we are not dealing with pixels
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv


#HYPERPARAMETERS
DF  = 1 #0.9999
TIME_COST = 0.0 # Extra penalization ( a negative reward) for not reaching agreement
#OBS: the "max steps" is controlled by RLLIB in the '"timesteps_total": train_steps'


class Env(MultiAgentEnv):
    '''
    Environment of an n-person bargaining game following Okada protocol
    '''

    metadata = {'render_modes': ['human']}

    def __init__(self, **config:Dict):
        super().__init__()
        '''
        Description:
            N-person bargaining game.
            Agents interchange proposing (coalition, payoff) and (accepting/rejecting).
            One agent acts as proposer and the others act as evaluators.
        inputs:
            "num_agents": number of players.
        State space observations:
            For the evaluators: they observe the proposed deal (coalition, payoff).
            For proposers: they observe the decision of the evaluators: accept/reject the proposal.
            These are passed in a dictionary to access its values separately.
        Actions:
            For proposers: their action is to propose (coalition, payoff).
            For the evaluators: their action is to (accept/reject).
            Note: The tricky part of the env is that when one is proposing there is one action space,
            and when one is evaluating, the action space is accept/reject.
        Reward:
            +1 (for everyone in the coalition) if a coalition forms
            0 if there is no agreement or if it takes too long.
            TODO: consider <0 reward if it takes too long to agree (this might help to converge quicker)
        Episode termination:
            If there is no agreement after N rounds.
        '''

        self.num_agents = config.get('num_agents', 3) #One agent learning to propose.
        #Requiredby RLLIB
        self._agent_ids = {i: j for i,j in enumerate([0]* self.num_agents)} #initialize to dummy values - keyword required by RLLIB
        self.dones_dict = {}
        self.proposer = 0
        self.t = 0

        # Action and observation spaces uniform for each agent - dimension of each vector is N (nbr of agents)
        # Observation space:
          # Proposer: ([dummy vector][dummy vector])
          # Responders: ([payoff vectors][responses from other players]) /1 accepted, 0 non accepted, -1 not responded
        # Action space:
          # Proposer: ([payoff vectors][[dummy vector]])
          # Responders: ([dummy vector][vector of 0's or 1's])

        nbr_obs_types    = 1 # When evaluating: 1= accept, 0 = reject// When proposing: Bid
        nbs_act_types    = 1 #just the proposal (for now)
        #TODO: think whether we need to add the time step or anything else - TO think: # [ Step num, Opponent Bid, Own price  ]
        #TODO: add the role type (proposer/evaluator)- important as network learns 2 roles


        #******* Policy network that learns to bid (based on the agen't responsed) - Like Velochy *********
        # Turn-based game:
        #   Proposer  - Observations are accept/reject || Actions are the proposals
        #   Evaluator - Observations are proposals || Actions: reject/accept

        self.action_space = Box(low=0, high=1, shape=(nbs_act_types,), dtype=np.float32) #Actions are the bids we learn
        self.observation_space = Discrete(2) #{0,1}-Observations are the responses we get from the Env

        self._reset() #method that instantiates all variables, sets seed and returns first obs

    #*******************************************************************************************************
    #Demised functions to implement on next state
    #*******************************************************************************************************
    #def get_proposal(self):
    #    ''' This is the action for the proposer
    #       Gets proposed amount from policy
    #       Normalizes it for convenient
    #    '''
        # This one should be implemented later, for now the proposal comes from the policy.

        #if self.step == 0.0:
        #   self.proposal = np.random.uniform(0,1,1)  #TODO: self.observation_space.sample() --> sample values from the correct space
        #else:
        #    self.proposal = 0 #TODO: take from policy (how??)
        #TODO Normalize proposals for convenience
        #TODO: Save a history of proposals - calculate mean and variance and return normalized value.
        #return self.proposal #TODO:normalized proposal scalar

        #def get_proposer_proposer_responses(self):
    #    '''Plays the bargaining game
    #       Gets proposer, proposals and responses
    #    '''
        #TODO: Implement on next step

        # Step1: Proposer is chosen at random
           #proposer  = self.get_proposer() #Returns an Idx - For now, proposer is always agent 0
        # Step2: The proposer takes action (makes proposal following a Policy)
           #proposal = self.get_proposal() #normalized bid - taken from Policy
        # Step3: the proposal is passed to the Obs Dict of the other agents for evaluation
           #responses_dict = self.get_responses(proposal) #responses of each agent
           # return proposer, proposal, responses_dict

    #@property
    #def get_obs_dict(self):
    #    '''Observation dictionary for each running agent
    #       The observation space is:
    #           Proposer:  Step num, Bid, role type (1= proposer/ 0= evaluator)
    #           Evaluator: Step num, Bid, role type  (1= proposer/ 0= evaluator)

    #        NOTE:This env just learns to accept - not to propose -
    #     '''
        #TODO: this one will be implemented on the next stage.
        #obs_dict = {}
         # Build Obs Dict per agent - the Obs of one agent is the Action of the previous agent - Need to play bargaining game.
        #proposer, proposal, response_boolean = self.get_proposer_proposer_responses() #gets proposer, proposal and responses
        #obs_dict = {id:proposal for id in  self._agent_ids if id != proposer} #observation for evaluators
        #obs_dict[proposer]  = response_boolean      #observations for proposers (dummy info)
        # return obs_dict #observations per agent


   #def get_responses(self,proposal):
   #     '''Implements Okada's way of responding.
   #        Here, because is the Rubinstein variation (just two players), we simplify and the responder agent is always the same.
   #        In theory, the response should come from a Policy, however, here we are simplifying and the response is harcoded.
   #        NOTE: This IS the environment! (whose internal dynamics we don't know)
   #     '''
   #     for agent_id in self._agent_ids:  #TODO: actually needs to loop only throught the available players, NOT all the set
   #         if agent_id != self.proposer: #discard proposer
   #            if proposal != DF/(1+DF):
   #               responses_dict = {agent_id:0} #0-reject, 1-accept #TODO: process everyone's responses into a boolean
   #            else: responses = 1
   #     response_boolean = responses
   #     #TODO: if everyone agreed- pass a 1, else pass a 0
   #     return response_boolean #TODO: on the next step, responses should come from the learned Policy (how??)
    #*******************************************************************************************************


    def seed(self, seed=None):
        '''returns a seed for the env'''
        if seed is None:
            _, seed = seeding.np_random()
        return [seed]


    def set_proposer(self):
        '''Sets proposer'''
        random.choice(list(range(self.num_agents)))
        return None


    def _reset(self):
        '''Initializes all variables to zero and selects a proposer
           Used on the init and reset methods
        '''
        self.set_proposer() #sets a proposer randomly
        self.seed()
        self.t            = 0.0
        self.dones_dict   = {agent: False for agent in  self._agent_ids}
        self.dones_dict["__all__"] = False
        self.info         = {}
        self.reward_dict  = {agent: 0 for agent in  self._agent_ids}
        return None


    def reset(self):
        """Resets the state of the environment and returns an initial observation for the next episode.
           Selects player's roles (proposer, evaluators) and returns observations according to the role.
           Randomizes the board to train with different configurations each time.

           In our case, the Obs are the responses - {1-yes, 0-no}
           First observation are all zeros - to avoid passing a 'yes' and having an expurious reward

           TODO LATER: Note that RLLIB will use these observations to pass to the policy, which doesn't know who is playing
           (if the proposer or the responder), so we need to add an idx to retrieve from the right policy.
           The idx will indicate whose policy we need (i.e. next agent)
        """
        # Reset all variables and selects proposing agent
        self._reset()

        # Get first observation dict
        obs  = 0   #need to return something valid for the policy to start going. Best to start with a disagreement, to learn better.
        obs_dict = {agent_id: obs for agent_id in  self._agent_ids}
        return obs_dict #TODO Later:(step,role, observation) ## Observation dict per agent



    def calculate_rewards(self, action_dict):
        '''
        Helper function for the game's env.
        Returns rewards for each agent.
        '''
        pass


    def step(self, action_dict):
        """Run one timestep of the environment's dynamics.
           Accepts an action (response) and returns a tuple (observation, reward, done, info).
        :Arguments:
         action: proposal (i.e., the bid)
        :Returns: (next observation, rewards, dones, infos) after having taken the input actions.
         observation (tuple): the response
         reward (float) : Amount of reward returned after previous action.
         done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
         info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # Example of good structure: https://github.com/lcipolina/Ray/blob/main/MARL_RLlib_Tutorial.ipynb
        # Another example of a 2-person turn-based game: https://pettingzoo.farama.org/environments/classic/leduc_holdem/
         #TODO: -Question on Reward calculation -  my understanding is that RLLIB just needs the reward on the episode. Then they do the discounted and the accum rew

        #TODO next step: obs, rew, is_done, info,dones_dict = [], [], [], {},{}

        # selecciona al azar quien propone 1 solo por vez
        #Accion:(1,1,0) (10,90,0)--> OKADA // RUBINSTEIN
        #Respuesta: (1,0,1) --> siguiente ronda ==> 0


        #Hacks for learning
        accept_value = round(DF/(1+DF),2)
        tolerance = 0.50 #otherwise it doesn't learn --> need to think about this trick (could be other trick). Or we can do curriculum learning with this

        #process the action (bid)
        if list(action_dict.values()):  #on the first iteration, the policy comes back empty (RLLIB)
           proposal = round(list(action_dict.values())[0][0],2) # OBS: This gives an error when run standalone, but NOT when run with the RLLIB trainer
        else:
           proposal = 0

        print('proposal:',proposal)
        print('accept_value - minus:',accept_value-tolerance)
        print('accept_value - plus:',accept_value+tolerance)

        print('time', self.t)

        if (accept_value-tolerance)<= proposal <= (accept_value+tolerance): #agreement has a tolerance level
             rew = 1
             obs, rew, is_done, info = rew, rew, True, {} #TODO: obs = np.array([1,1,1]) #OBS: everyone (in the coalition) gets the same reward
             print('AGREEMENT REACHED!!')

        else: #game continues
            rew = 0
            obs, rew, is_done, info = rew, rew, False, {} #TODO: obs = np.array([1,1,1])
            self.t+=1
            print('game continues')

        print('obs (response):', obs)

        obs_dict   = {agent_id: obs for agent_id in self._agent_ids} #this is single agent, there is a single learning agent observing the response
        dones_dict = {agent_id: is_done for agent_id in self._agent_ids}
        dones_dict["__all__"] = True if is_done else False          #TODO: this only works for one agent
        rew_dict   = {agent_id: rew for agent_id in self._agent_ids} #TODO later: rewards should be only assigned to members of the coalition
        return obs_dict, rew_dict, dones_dict, info

        '''
        #TODO: If we start swapping agents, we need to pass the observations for **next agents** playing the t+1. Like this:
        # Observations and rewards per agent
        if is_done:
            obs_dict = {a: self._agent_observation(a) for a in self.agents}
            rewards = {a: self.rewards_to_send[a] for a in self.agents}
        else:
            next_agent = self.state["turn"]
            obs = {next_agent: self._agent_observation(next_agent)}
            rewards = {next_agent: self.rewards_to_send[next_agent]}
        dones = {"__all__": done}
        infos = {}

               #UPDATE DONES DICT
        # check for max length or agreement - if everyone agrees (observations = 1) - terminate the game (dones = True) and reward = 0


        if sum(self.dones_dict.values()) == self.num_agents:
              self.dones_dict["__all__"] = True
        else:
            self.dones_dict["__all__"] = False

        #If game continues, pass next observation - this is where we code the environment dynamics
        self.get_obs(action_dict) #sets  self.obs_dict # dict of lists [step, proposal, role]

        #TODO: OBS: Only those  agents' names that require actions in the next call to `step()` should
        # be present in the returned observation dict (here: all, as we always step
        #From here: https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical
        '''


    def render(self):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """

        #TODO implement
        pass

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        #to implement
        #pass



if __name__ == '__main__':

    env = Env()

    #print(env.reset())
    env.step({0:1})
    #print(env.reset())
    #print(env.step({0:0.5}))
    #print(env.observation_space.sample())
    #print(env.observation_space.sample())
