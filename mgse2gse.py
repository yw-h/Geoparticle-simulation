import pyspedas
from pytplot import get_data, store_data, get_coords, set_coords, tplot, split_vec, tsmooth
import numpy as np
from pytplot import tcrossp

def mgse2gse(var_name, ax_var, reversal=False, newname=None):
    var_data = get_data(var_name)
    times = var_data.times
    ax_data = get_data(ax_var)
    output_data = np.zeros_like(var_data.y)
    for i in range(len(times)):
        sgse = ax_data.y[i, :]
        zgse = np.array([0, 0, 1])
        xmgse = sgse
        sgsexzgse = np.cross(sgse, zgse)
        absgsexzgse = np.linalg.norm(sgsexzgse)
        ymgse = - sgsexzgse/absgsexzgse
        zmgse = np.cross(xmgse, ymgse)

        vx = var_data.y[i, 0]
        vy = var_data.y[i, 1]
        vz = var_data.y[i, 2]

        if not reversal:
            output_data[i, 0] = xmgse[0] * vx + ymgse[0] * vy + zmgse[0] * vz
            output_data[i, 1] = xmgse[1] * vx + ymgse[1] * vy + zmgse[1] * vz
            output_data[i, 2] = xmgse[2] * vx + ymgse[2] * vy + zmgse[2] * vz                  
        else:
            output_data[i, 0] = xmgse[0] * vx + xmgse[1] * vy + xmgse[2] * vz
            output_data[i, 1] = ymgse[0] * vx + ymgse[1] * vy + ymgse[2] * vz
            output_data[i, 2] = zmgse[0] * vx + zmgse[1] * vy + zmgse[2] * vz                 

    if newname is None:
        newname = var_name + '2gse'
    store_data(newname, data={'x':times, 'y':output_data}) 
    return newname  
            