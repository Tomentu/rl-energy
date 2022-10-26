# Run this again after editing submodules so Colab uses the updated versions
from citylearn import CityLearn
from pathlib import Path
from sac import SAC
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from common.preprocessing import *
import json


# Load environment

#对观察值进行处理
def obs_init(x,encoder):
    return np.hstack(encoder*x)

climate_zone = 5
buildings = ["Building_1"]




params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':buildings,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (0,2000),#8760*1-1
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False }



env = CityLearn(**params)

# Contains the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()
observation_spaces = {uid : o_space for uid, o_space in zip(buildings, observations_spaces)}
params_agent = {'building_ids':buildings,
                 'buildings_states_actions':'buildings_state_action_space.json', 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}


obs_dim = 34


act_dim = env.action_space.shape[0]
sac = SAC(obs_dim,act_dim,act_scale=0.5,id=24)
#sac.load_weights()

def get_encoder():
    with open('buildings_state_action_space.json') as json_file:
        buildings_states_actions = json.load(json_file)

    encoder = {}
    for uid in buildings:
        state_n = 0
        encoder[uid] = []
        for s_name, s in buildings_states_actions[uid]['states'].items():
            if not s:
                pass
                #encoder[uid].append(0)
            elif s_name in ["month", "hour"]:
                encoder[uid].append(periodic_normalization(observation_spaces[uid].high[state_n]))
                state_n += 1
            elif s_name == "day":
                encoder[uid].append(onehot_encoding([1, 2, 3, 4, 5, 6, 7, 8]))
                state_n += 1
            elif s_name == "daylight_savings_status":
                encoder[uid].append(onehot_encoding([0, 1]))
                state_n += 1
            elif s_name == "net_electricity_consumption":
                encoder[uid].append(remove_feature())
                state_n += 1
            else:
                encoder[uid].append(
                normalize(observation_spaces[uid].low[state_n], observation_spaces[uid].high[state_n]))
                state_n += 1
        return encoder
encoder = get_encoder()
def init():
    env.reset()
    done = False
    obs_list = []
    reward = []
    while not done:
        act = env.action_space.sample()
        #print(act)
        obs,reward,done,_ = env.step([act])
        obs = obs_init(obs,encoder["Building_1"])
        for b,o,r in zip(buildings,obs,reward):
            obs_list.append(o)
    obs_list = np.array(obs_list).astype('float32')
    print(len(obs_list))
    return np.mean(obs_list,axis=0),np.std(obs_list,axis=0)


obs_mean,obs_std = init()

sac.set_normal(-74970.45, 224360.34)

def run_episode(training=False):
    done = False
    obs = env.reset()
    #print(obs)
    obs = np.array(obs).astype('float32')
    total_reward = 0
    k=0
    while not done:
        i = 0
        act = []
        if k%500==0:
            print("\r", end="")
            print(k,end="")

        k+=1

        for build_obs,build in zip(obs,env.buildings):
            build_obs = obs_init(build_obs,encoder[build])
            build_obs = np.array(build_obs).astype('float32')
            build_act = sac.select_action(build_obs)
            act.append(build_act)
            i += 1
        act = np.array(act)
        next_obs,reward,done,_ = env.step(act)
        n_obs = []
        for o,next_o,r,d in zip(obs,next_obs,reward,buildings):
            #print(reward)
            o  = obs_init(o, encoder["Building_1"])
            next_o = obs_init(next_o, encoder["Building_1"])
            sac.append_raplay(o,act[0],reward[0],next_o,tf.cast(done,tf.float32))
            #break
        if training :
            sac.train_step()
        total_reward =total_reward+ reward[0]
        obs = next_obs
        #break
    return total_reward




for epoch in range(30):
    print("epoch{}".format(epoch))
    reward  = run_episode(training=True)
    #print(sac.sample_replay(30))
    sac.save_weights()
    print("")
    print("epoch{},reward{}".format(epoch,reward))

run_episode(False)
print(sac.sample_replay(3)[3])

#X = np.array([j[2] for j in sac.replay_buffer])
