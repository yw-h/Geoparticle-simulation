from sim_pulse_mag import prt_sim
import numpy as np
import multiprocessing
from threading import Thread
import os
from scipy import constants as C
from scipy.interpolate import interp1d
import pandas as pd
import random
import time
import logging
# from scipy.optimize import curve_fit
# from scipy.stats import kappa3
# from scipy import integrate

# 日志配置
def setup_logger(log_file='./pulse_simulation.log'):
    logger = logging.getLogger('SimulationLogger')
    logger.setLevel(logging.DEBUG)

    # 文件日志处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()



standard_kec = 1800
standard_L = 5.0
standard_file = r"./std_file.csv"
df_std = pd.read_csv(standard_file)

t_std = df_std["time(sec)"].to_numpy()

def process(KEc, L, PHI):
    try:
        logger.info("Simulating: E={} , L={}, PHI={}".format(KEc, L, PHI))
        input_parameters = {
            "xmu": -1, # -1 for electron, 1 for proton
            "IOPT": 1, # = kp + 1
            "raddist0": L, # initial L
            "longi0": PHI, # intital phi
            'pa':90, # pitch angle
            "KEc0": KEc, # initial kinetic energy /KeV
            "timedir": 1,
            "Tout": 60,
            "Dmin": 0,
            "pulse_flag": 1, 
            "tmax": 30, # maximum simulation time /min
        }         
        pulse_parameters = {
            "phi0":0,
            "E0": 2, # 5mV/m
            "c0":1,
            "c1":1,
            "c2":0,
            "c3":1,
            "p":1,
            "va":50000,
            "ri":0.6371E8 * 0.8, # m ri/vpulse determine the arrival time of pusle
            "ti":100,
            "di":0.25e7, # width of the pusle
            "rd":2.06, # pulse rebound at rd/2 Re
            "vpulse": 0.5e5, # m/s
            "duration" : 450 # s 
        }
        p1 = prt_sim(input_parameters, pulse_parameters, {}) 
        p1_r, p1_t, p1_wtot = p1.prt_sim() 

        df = pd.DataFrame(columns=["time(sec)", "r", "wtot"])
        df["time(sec)"] = p1_t
        df["r"] = p1_r 
        df["wtot"] = p1_wtot

        df_sp = df
        df_sp["x"] = df_sp["r"].apply(lambda x:x[0])
        df_sp["y"] = df_sp["r"].apply(lambda x:x[1])
        df_sp["z"] = df_sp["r"].apply(lambda x:x[2])
        df_sp = df_sp.drop('r', axis=1)
        x_sp = df_sp["x"].to_numpy()
        y_sp = df_sp["y"].to_numpy()
        z_sp = df_sp["z"].to_numpy()
        wtot_sp = df_sp["wtot"].to_numpy()
        t_sp = df_sp["time(sec)"].to_numpy()
        interp_x_sp = interp1d(t_sp, x_sp, fill_value="extrapolate")(t_std) 
        interp_y_sp = interp1d(t_sp, y_sp, fill_value="extrapolate")(t_std)
        interp_z_sp = interp1d(t_sp, z_sp, fill_value="extrapolate")(t_std)
        interp_wtot_sp = interp1d(t_sp, wtot_sp, fill_value="extrapolate")(t_std)

        particle_result = np.array([
            t_std, # 时间序列
            interp_x_sp,      # x 坐标
            interp_y_sp,      # y 坐标
            interp_z_sp,      # z 坐标
            interp_wtot_sp,   # 总能量（J）  
        ]) 

        return particle_result
    except Exception as e:
        logger.error(f"Process failed for KEc={KEc}, L={L}, PHI={PHI}. Error: {e}")

def subprocess(random_id, stop_event, shared_counter, counter_lock, target_count, result_queue):
    while not stop_event.is_set():
        KEc = random.random() * (5800 - 300) + 300
        L = random.random() * (7.5 - 5.5) + 5.5
        PHI = random.random() * 360
        particle_result = process(KEc, L ,PHI)  

        if particle_result is not None:
            result_queue.append(particle_result)  

        with counter_lock:
            shared_counter.value += 1
            logger.info(f"Process {random_id}: Total calls={shared_counter.value}")
            if shared_counter.value >= target_count:
                stop_event.set()

        time.sleep(0.1) 
                    

if __name__ == '__main__':
    def save_batch_to_npy(batch, batch_id):
        try:
            batch_array = np.stack(batch)  # 将粒子结果数组组合成一个大数组
            save_path = f"./results_batch/batch_{batch_id}.npy"
            if not os.path.exists('./results_batch'):
                os.makedirs('./results_batch')
            np.save(save_path, batch_array)
            logger.info(f"Saved batch {batch_id} to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save batch {batch_id}. Error: {e}")

    def monitor_and_save(result_queue, stop_event, batch_size=10):
        batch_id = 0
        current_batch = []
        while not stop_event.is_set() or len(result_queue) > 0:
            # 持续从队列中取出结果
            while len(result_queue) > 0:
                current_batch.append(result_queue.pop(0))
                # 如果达到批量大小，保存并清空当前批次
                if len(current_batch) >= batch_size:
                    batch_id += 1
                    save_batch_to_npy(current_batch, batch_id)
                    current_batch = []
            time.sleep(0.1)  # 避免主进程占用过多 CPU

        # 保存剩余的结果
        if current_batch:
            batch_id += 1
            save_batch_to_npy(current_batch, batch_id)
    
    with multiprocessing.Manager() as manager:

        stop_event = manager.Event()
        shared_counter  = manager.Value('i', 0)  # 共享计数器，初始值为0
        counter_lock = manager.Lock()  # 锁保护计数器
        target_count = 1000 # 模拟的粒子总数
        batch_size = 500 # batch大小
        result_queue = manager.list()  # 用于缓存子进程的结果

        pool = multiprocessing.Pool(processes=15)

        try:
            # 启动保存线程
            save_thread = Thread(target=monitor_and_save, args=(result_queue, stop_event, batch_size))
            save_thread.start()

            # 启动子进程任务
            for i in range(15):
                pool.apply_async(subprocess, args=(i, stop_event, shared_counter, counter_lock, target_count, result_queue))

            pool.close()
            pool.join()  # 等待所有子进程完成
            stop_event.set()  # 通知保存线程结束
            save_thread.join()  # 等待保存线程完成
        except KeyboardInterrupt:
            logger.warning("Received interrupt, stopping processes...")
            stop_event.set()
            pool.terminate()
            pool.join()
        except Exception as e:
            logger.error(f"Unexpected error in main process: {e}")