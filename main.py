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
    obs = np.array(encoder*x)
    return np.hstack(obs)

climate_zone = 5
buildings = ["Building_1"]




params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':buildings,
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (0,8760*1-1),#8760*1-1
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
sac.load_weights()

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
    reward_list = []
    while not done:
        act = env.action_space.sample()
        #print(act)
        obs,reward,done,_ = env.step([act])

        for b,o,r in zip(buildings,obs,reward):
            o = obs_init(o, encoder["Building_1"])


            obs_list.append(o)
            reward_list.append(r)

    obs_list = np.array(obs_list).astype('float32')
    reward_list = np.array((reward_list)).astype('float32')
    print(obs_list.shape)
    reward_mean = np.mean(reward_list,axis=0)+1e-5
    reward_std = np.std(reward_list,axis=0)+1e-5

    obs_mean = np.mean(obs_list,axis=0)
    obs_std = np.std(obs_list,axis=0)+1e-5
    return reward_mean,reward_std,obs_mean,obs_std


r_normal_mean,r_normal_std,normal_mean,normal_std = init()
print(r_normal_mean,r_normal_std)
#exit()
sac.set_normal(r_normal_mean,r_normal_std)


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




for epoch in range(1):
    print("epoch{}".format(epoch))
    reward  = run_episode(training=False)
    #print(sac.sample_replay(30))
    #sac.save_weights()
    print("")
    print("epoch{},reward{}".format(epoch,reward))

    sim_period = (8000,8500)
    # Plotting electricity consumption breakdown
    interval = range(sim_period[0], sim_period[1])
    plt.figure(figsize=(30,8))
    #plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
    plt.plot(env.net_electric_consumption_no_storage[interval])
    plt.plot(env.net_electric_consumption[interval], '--')
    plt.xlabel('time (hours)', fontsize=24)
    plt.ylabel('kW', fontsize=24)
    plt.xticks(fontsize= 24)
    plt.yticks(fontsize= 24)
    plt.legend([ 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RL for storage(kW)'], fontsize=4)
    plt.show()
#run_episode(False)
#print(sac.sample_replay(3)[2])

#X = np.array([j[2] for j in sac.replay_buffer])
