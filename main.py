# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from pathlib import Path
from sac import SAC
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load environment
climate_zone = 5
buildings = ["Building_1"]

params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':buildings,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (0, 8760*1-1),#8760*1-1
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False }

env = CityLearn(**params)

# Contains the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

params_agent = {'building_ids':buildings,
                 'buildings_states_actions':'buildings_state_action_space.json', 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}

obs_dim = 30
act_dim = env.action_space.shape[0]
sac = SAC(obs_dim,act_dim,id=24)
sac.load_weights()
sac.set_normal(-74970.45, 224360.34)
#对观察值进行处理
def obs_init(x):
    #x[:,:1]=12.
    month = tf.keras.layers.Embedding(13,2,input_length = 1)(x[:,:1])
    month = tf.keras.layers.Flatten()(month)
    day = tf.keras.layers.Embedding(9,2,input_length = 1)(x[:,1:2])
    day = tf.keras.layers.Flatten()(day)
    hour = tf.cos((x[:,2:3]-12)/12*(3.1415926/2))
    x = x[:,3:]
    x = tf.keras.layers.Concatenate()([month,day,hour,x])
    return x



def run_episode(training=False):
    done = False
    obs = env.reset()
    print(obs)
    obs = obs_init(obs)
    total_reward = 0
    while not done:
        i = 0
        act = []

        for build in env.buildings:
            build_obs = np.array(obs[i]).astype('float32')

            build_act = sac.select_action(build_obs)
            act.append(build_act)
            i += 1

        next_obs,reward,done,_ = env.step(act)
        #print(reward)
        next_obs = obs_init(next_obs)
        sac.append_raplay(obs[0],act[0],reward[0],next_obs[0],tf.cast(done,tf.float32))

        if training :
            sac.train_step()
        total_reward =total_reward+ reward[0]
        obs = next_obs
        #break
    return total_reward




for epoch in range(20000):
    reward  = run_episode(training=True)
    #print(sac.sample_replay(30))
    sac.save_weights()
    print(reward)


#X = np.array([j[2] for j in sac.replay_buffer])
