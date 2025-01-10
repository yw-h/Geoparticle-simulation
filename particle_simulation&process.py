from prtsim3 import prt_sim
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
import concurrent.futures
import os

L0 = 6
Kecs = [500, 1000, 1500, 1800, 2100, 2600, 3400, 4200, 5200]
# Kecs = [50, 100, 300]
Es = np.arange(0.1, 1, 0.1)
Lambdas = np.arange(0, 1.8, 0.1)
# Lambdas = [-1]
longi0s = np.arange(0, 360, 20)

def process(combination):
    Kec,E, Lambda, longi0 = combination
    print(Kec,np.round(E,2), Lambda, longi0)
    input_parameters = {
        "xmu": -1, # -1 for electron, 1 for proton
        "kp": 0, # 
        "raddist0": L0, # initial L
        "longi0": longi0, # intital phi
        'pa':90, # pitch angle
        "KEc0": Kec, # initial kinetic energy /KeV
        "timedir": 1,
        "Tout": 60,
        "Dmin": 0,
        "pulse_flag": 0, 
        "tmax": 10800, # maximum simulation time /s
    }         

    background_parameters = {
        "phi0" : np.pi/2,
        "E0" : E, # mV/m
        "omega":0,
        "guass_flag":0,
        "random_phi_flag":0,
        "Lambda":Lambda
    }
    p1 = prt_sim(input_parameters, {}, background_parameters)
    sol = p1.prt_sim()    
    t = sol.t
    y = sol.y
    if not os.path.exists("./result/{}/".format(Kec)):
        os.makedirs("./result/{}/".format(Kec))
    np.savez("./result/{:.0f}/delay1000_{:.1f}_{:.1f}_{:.0f}.npz".format(Kec,E, Lambda, longi0), t=sol.t, y=sol.y)


combinations = list(itertools.product(Kecs, Es, Lambdas, longi0s))

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process, combination) for combination in combinations]

combinations = list(itertools.product(Kecs, Es, Lambdas))

def process2(combination):
    square_delta_Ls = list()
    ts = list()
    for longi0 in longi0s:
        data = np.load("./result/delay1000_{:.0f}/{:.1f}_{:.1f}_{:.0f}.npz".format(combination[0],combination[1],combination[2],longi0))
        t = data['t']
        y = data['y'].T
        L = np.linalg.norm(y, axis=1)
        square_delta_L = list(np.square(L-6))
        square_delta_Ls.append(square_delta_L)
        ts.append(t)
    max_length = len(max(square_delta_Ls, key=len))
    padded_square_delta_Ls = [lst + [lst[-1]]  * (max_length - len(lst)) for lst in square_delta_Ls]
    mean_SDL = np.mean(padded_square_delta_Ls, axis=0)
    t = max(ts, key=len)
    if not os.path.exists("./result_processed/{}/".format(combination[0])):
        os.makedirs("./result_processed/{}/".format(combination[0]))
    np.savez("./result_processed/{:.0f}/delay1000_{:.1f}_{:.1f}.npz".format(combination[0],combination[1],combination[2]), t=t, mean_SDL=mean_SDL)

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process2, combination) for combination in combinations]