import numpy as np
import pandas as pd
import scipy as sp 
import os
import argparse

from scipy import constants as C
from scipy.optimize import curve_fit
from scipy.stats import kappa3
from scipy import integrate
import h5py
import glob
from math import ceil

from scipy.interpolate import interp2d, RectBivariateSpline
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.interpolate import interp2d, RectBivariateSpline
class FluxCalculator:
    def __init__(self, csv_file, interpolation_method='linear'):
        """
        初始化通量计算器
        
        Parameters:
        csv_file (str): CSV文件路径
        interpolation_method (str): 插值方法，可选 'linear', 'cubic', 'quintic', 'spline'
        """
        # 读取数据，跳过表头
        self.df = pd.read_csv(csv_file)
        
        # 获取L值（第一列到倒数第二列）
        self.L_values = self.df.columns[:-1].astype(float).values
        
        # 获取能量值（最后一列，去除第一行的'Kec'）
        self.energies = self.df.iloc[:, -1].astype(float).values
        
        # 获取通量数据（去除第一行和最后一列）
        self.flux_data = self.df.iloc[:, :-1].astype(float).values
        
        # 计算最大通量值用于归一化
        self.max_flux = np.max(self.flux_data)
        
        # 保存插值方法
        self.interpolation_method = interpolation_method

        self._create_interpolator()
        
    def _create_interpolator(self):
        """创建二维插值函数"""
        if self.interpolation_method == "linear":
            kx = 1
            ky = 1
        elif self.interpolation_method == "cubic" or 'spline':
            kx = 3
            ky = 3
        elif self.interpolation_method == "quintic":
            kx = 5
            ky = 5
        self.interpolator = RectBivariateSpline(
            self.energies, 
            self.L_values, 
            self.flux_data, 
            kx=kx, 
            ky=ky,
            s=0)

    def get_flux(self, L, energy):
        """
        获取指定L值和能量的通量
        
        Parameters:
        L (float): L值
        energy (float): 能量值(MeV)
        
        Returns:
        float: 插值计算的通量值
        """
        # max_L = np.max(self.L_values) 
        # L_clipped = min(L, max_L)
        # if (L_clipped < np.min(self.L_values) or  # 使用裁剪后的 L_clipped 进行范围检查
        #     energy < np.min(self.energies) or energy > np.max(self.energies)):
        #     return 0  # 能量超出范围返回 0
        
        return self.interpolator(energy, L)[0][0]
    
    def get_particle_weight(self, L, energy):
        """
        计算单个粒子的权重
        
        Parameters:
        L (float): L值
        energy (float): 能量值(MeV)
        
        Returns:
        float: 相对于最大通量的权重值 (0-1范围)
        """
        flux = self.get_flux(L, energy)
        # 使用最大通量归一化
        weight = max(0, flux) / self.max_flux
        return weight
        # return flux
    
    def get_weights(self, L_values, energies):
        """
        计算一组粒子的权重
        
        Parameters:
        L_values (array): L值数组
        energies (array): 能量值数组
        
        Returns:
        array: 权重数组 (0-1范围)
        """
        weights = np.array([self.get_particle_weight(L, E) 
                          for L, E in zip(L_values, energies)])
        return weights


def kec_label_batch(wtot_array):
    kec_array = (wtot_array / C.elementary_charge / 1e6 - 0.511) * 1000
    # 定义能道区间和对应的能道标签
    ranges = [
        (300, 700, 500),
        (700, 900, 800),
        (900, 1200, 1000),
        (1200, 1600, 1500),
        (1600, 2000, 1800),
        (2000, 2500, 2100),
        (2500, 3200, 2600),
        (3200, 4000, 3400),
        (4000, 5000, 4200),
        (5000, 6200, 5200),
    ]
    labels = np.zeros_like(kec_array, dtype=int)

    # 根据区间查找标签
    for lower, upper, label in ranges:
        mask = (kec_array > lower) & (kec_array < upper)
        labels[mask] = label
    return labels   

def func_kappa(x, a, loc, scale):
    return kappa3.pdf(x, a, loc, scale) # a: shape param, loc:location param, scale: scale param
def func_exp(x, a, b, c):
    return a * b ** (c*x)

def normal_distribution(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)
def double_normal_distribution(x, amplitude, mean, stddev):
    guass1 = amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)
    guass2 = amplitude * np.exp(-0.5 * ((x - mean - 2)/ stddev*2)**2)
    return guass1 + guass2


# def exp_param_dict():
#     Ls = np.arange(3, 7.5, 0.01)
#     Ls = np.round(Ls, 2)
#     exp_dict = {L:[] for L in Ls}
#     for L in Ls:
#         L_temp = 6.54 if L > 6.54 else L
#         fit_flux = observed_flux
#         xdata = fit_flux["Kec"]
#         ydata = fit_flux["{:.2f}".format(L_temp)]
        
#         ydata = ydata / integrate.trapz(ydata, xdata) 
#         popt, pcov = curve_fit(func_exp, xdata[14:], ydata[14:], p0=(1,1,-1))
#         exp_dict[L] = popt
#     return exp_dict

def get_flux_batch(batch_data, flux_csv_path="./observed_flux.csv"):
    calculator = FluxCalculator(flux_csv_path)
    L_shell_array = np.sqrt(batch_data[:, 1, :]**2 + batch_data[:, 2, :]**2, + batch_data[:, 3, :]**2)
    wtot_array = batch_data[:,-1,:]
    flux_array = np.zeros_like(wtot_array)
    wtot_init_array = batch_data[:, -1, 0]
    kec_init_array = (wtot_init_array / C.elementary_charge / 1e6 - 0.511) # MeV
    L_init = L_shell_array[:, 0]
    # mod001 = divmod(L_init, 0.01)
    # mask = (mod001[1] < 0.005)
    # L_init_near_array = np.zeros_like(L_init)
    # L_init_near_array[mask] = np.round(mod001[0][mask] * 0.01, 2)
    # L_init_near_array[~mask] = np.round((mod001[0]+1)[~mask] * 0.01, 2)
    # exp_params = [exp_dict[5] for L in L_init_near_array]
    # x = [float(i) for i in observed_flux.columns[:-1]] # 观测到的L值 
    # y =  observed_flux.loc[observed_flux["Kec"]==1.8,:].to_numpy().squeeze()[:-1] # 观测到的通量（1.8MeV）
    # # 使用正态分布曲线进行拟合
    # popt, pcov = curve_fit(normal_distribution, x, y)  # 通量和L值的关系近似为正态分布
    # y_fit = double_normal_distribution(L_init_near_array, *popt) # 在L处的1.8MeV电子的通量（双高斯分布拟合）

    # density1800 = np.array([func_exp(1.8, *exp) for exp in exp_params])
    # densitykec = np.array([func_exp(i[0], *i[1]) for i in zip(kec_init_array, exp_params)])
    # fitted_flux = (densitykec / density1800) * y_fit
    weights = calculator.get_weights(L_init, kec_init_array)
    return weights

# mageis_observed_flux = pd.read_csv("./mageis_fedu_mean.csv")
# rept_observed_flux = pd.read_csv("./rept_fedu_mean.csv")
# mageis_observed_flux = mageis_observed_flux.loc[mageis_observed_flux["Kec"]<1.6, :]
# observed_flux = pd.merge(mageis_observed_flux, rept_observed_flux, how='outer')
# exp_dict =  exp_param_dict() # 指数分布参数表
def process_and_save_timepoint_data(input_dir, output_dir, flux_csv_path="./observed_flux.csv",batch_size=10):
    batch_files_folder = os.path.join(input_dir, "results_batch") # 使用 input_dir
    batch_files = glob.glob(os.path.join(batch_files_folder, "*.npy"))
    kecs = [500, 800, 1000, 1500, 1800, 2100, 2600, 3400, 4200, 5200]

    # 获取时间步长
    sample_data = np.load(batch_files[0])
    time_steps = sample_data.shape[2]

    # 创建输出目录
    timepoint_output_dir = os.path.join(output_dir, "timepoint_data")
    for t in range(time_steps):
        os.makedirs(timepoint_output_dir, exist_ok=True)

    num_batches = ceil(len(batch_files) / batch_size)

    for batch_idx in range(num_batches):
        print(f"processing batch {batch_idx}/{num_batches}")
        # 分批处理保存文件
        start_idx = batch_idx * batch_size 
        end_idx = min((batch_idx + 1) * batch_size, len(batch_files))
        current_batch_files = batch_files[start_idx:end_idx]

        with h5py.File(os.path.join(timepoint_output_dir, f"particle_data_batch{batch_idx}.h5"), 'w') as f:
            for kec in kecs:
                kec_group = f.create_group(f"kec_{kec:.1f}")
                # 为每个时间点预创建数据集
                for t in range(time_steps):
                    kec_group.create_group(f"timestep_{t}")    

            # 处理每个批次文件
            for batch_idx, batch_file in enumerate(current_batch_files):
                print(f"Processing subbatch {batch_idx+1}/{len(current_batch_files)}")
                batch_data = np.load(batch_file)
                
                # 计算L_shell和通量
                L_shell_array = np.sqrt(batch_data[:, 1, :]**2 + batch_data[:, 2, :]**2 + batch_data[:, 3, :]**2)
                kec_label_array = kec_label_batch(batch_data[:, 4, :])
                flux_array = get_flux_batch(batch_data, flux_csv_path=flux_csv_path)
                flux_array = np.broadcast_to(flux_array[:, np.newaxis], np.shape(L_shell_array))

                batch_data_new = np.zeros((batch_data.shape[0], batch_data.shape[1]+3, batch_data.shape[2]))
                batch_data_new[:, :5, :] = batch_data
                batch_data_new[:, 5, :] = L_shell_array
                batch_data_new[:, 6, :] = kec_label_array
                batch_data_new[:, 7, :] = flux_array

                # 对每个时间点进行处理
                for t in range(time_steps):
                    # 提取当前时间点的数据
                    current_data = batch_data_new[:, :, t]
                    kec_labels = current_data[:, 6]

                    # 按能量分类保存数据
                    for kec in kecs:
                        mask = kec_labels == kec
                        if np.any(mask):
                            kec_data = current_data[mask]
                            group_path = f"kec_{kec:.1f}/timestep_{t}"
                            if "data" in f[group_path]:
                                existing_data = f[group_path]["data"][:]
                                combined_data = np.vstack([existing_data, kec_data])
                                del f[group_path]["data"]
                                f[group_path].create_dataset("data", data=combined_data)
                            else:
                                f[group_path].create_dataset("data", data=kec_data)

            # 保存统计信息
            stats = f.create_group("statistics")
            for kec in kecs:
                particle_counts = []
                for t in range(time_steps):
                    group_path = f"kec_{kec:.1f}/timestep_{t}"
                    if "data" in f[group_path]:
                        particle_counts.append(f[group_path]["data"].shape[0])
                    else:
                        particle_counts.append(0)
                stats.create_dataset(f"kec_{kec:.1f}_counts", data=particle_counts)
                print(f"KEC {kec:.1f}: Particles per timestep = {particle_counts}")


def process_and_save_initkec_data(input_dir, output_dir, flux_csv_path="./observed_flux.csv"):
    batch_files_folder = os.path.join(input_dir, "results_batch") # 使用 input_dir
    batch_files = glob.glob(os.path.join(batch_files_folder, "*.npy"))
    file_shapes = {file: np.load(file, mmap_mode='r').shape[0] for file in batch_files} # 获取每个batch的粒子数
    particle_nums = sum(file_shapes[file] for file in batch_files) # 总粒子数

    print(f"total particle nums: {particle_nums}")
    batch_data_sp =  np.load(batch_files[0])  # 获得形状

    kecs = [500, 800, 1000, 1500, 1800, 2100, 2600, 3400, 4200, 5200]
    # result_dict = {kec: np.empty((0, batch_data_sp.shape[1]+3, batch_data.shape[2])) for kec in kecs} 
    kec_indices = {kec: [] for kec in kecs}  # 用于存储每个 kec 的起始索引
    result_shape = (particle_nums, batch_data_sp.shape[1] + 3, batch_data_sp.shape[2])
    result_array = np.zeros(result_shape)  # 预分配结果数组
    # exp_dict =  exp_param_dict() # 指数分布参数表

    current_index = 0
    for batch_file in batch_files:
        
        batch_data =  np.load(batch_file)  # 时间序列 x 坐标 y 坐标 z 坐标 总能量（J） 
        # particle_nums = batch_data.shape[0]
        print(batch_file)
        # 计算 L_shell, kec_label, flux
        time_length = batch_data.shape[2]
        L_shell_array = np.sqrt(batch_data[:, 1, :]**2 + batch_data[:, 2, :]**2, + batch_data[:, 3, :]**2)
        kec_label_array = kec_label_batch(batch_data[:, 4, :])
        flux_array = get_flux_batch(batch_data, flux_csv_path=flux_csv_path)
        flux_array = np.broadcast_to(flux_array[:, np.newaxis], np.shape(L_shell_array)) # 扩展至同一形状
        # 扩展数据结构, 将计算结果添加到batch数组中
        batch_data_new = np.zeros((batch_data.shape[0], batch_data.shape[1]+3, batch_data.shape[2]))
        batch_data_new[:, :5, :] = batch_data
        batch_data_new[:, 5, :] = L_shell_array
        batch_data_new[:, 6, :] = kec_label_array
        batch_data_new[:, 7, :] = flux_array

        # 按 kec 分组写入结果数组
        init_kec_values = batch_data_new[:, -2, 0] 
        unique_kecs, indices = np.unique(init_kec_values, return_inverse=True) # 按照初始能量来分组
        for i, kec in enumerate(unique_kecs):
            if kec in kecs:
                mask = indices == i
                count = np.sum(mask)
                result_array[current_index:current_index+count] = batch_data_new[mask]
                kec_indices[kec].append((current_index, current_index + count))
                current_index += count
    init_kec_output_dir = os.path.join(output_dir, "init_kec_batch") 
    for kec in kecs:
        try:
            kec_data = np.concatenate(
                [result_array[start:end] for start, end in kec_indices[kec]], axis=0
            )
            os.makedirs(init_kec_output_dir, exist_ok=True) # 使用 init_kec_output_dir
            np.save(os.path.join(init_kec_output_dir, f"{kec:.1f}.npy"), kec_data) 
            print(f"kec={kec}, 数据形状: {kec_data.shape}")
        except Exception as e:
            print(f"kec={kec}, 数据形状: 0 {e}")

# mageis_observed_flux = pd.read_csv("./mageis_observed_flux.csv")
# rept_observed_flux = pd.read_csv("./rept_observed_flux.csv")
# mageis_observed_flux = mageis_observed_flux.loc[mageis_observed_flux["Kec"]<1.6, :]
# observed_flux = pd.merge(mageis_observed_flux, rept_observed_flux, how='outer')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw particle simulation results.")
    parser.add_argument("input_dir", help="Path to the input directory containing 'results_batch'") # 输入目录
    parser.add_argument("output_dir", help="Path to the output directory to save processed results") # 输出目录
    parser.add_argument("--flux_csv", default="./observed_flux.csv", help="Path to the observed flux CSV file") # 通量CSV文件路径
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing timepoint data") # 批次大小
    # parser.add_argument("--interpolation_method", default='cubic', choices=['linear', 'cubic', 'quintic', 'spline'], help="Interpolation method for flux calculation") # 插值方法
    parser.add_argument("--process_type", default="both", choices=["timepoint", "initkec", "both"], help="Type of processing to perform: timepoint, initkec, or both") # 处理类型

    args = parser.parse_args()

    if args.process_type in ["timepoint", "both"]:
        process_and_save_timepoint_data(args.input_dir, args.output_dir, args.flux_csv, args.batch_size)
    if args.process_type in ["initkec", "both"]:
        process_and_save_initkec_data(args.input_dir, args.output_dir, args.flux_csv)

    print("Preprocessing completed.")

    # process_and_save_timepoint_data(input_dir=r"I:\Geoparticle-simulation\result_compressedfield", output_dir=r"I:\Geoparticle-simulation\result_compressedfield")
    # process_and_save_initkec_data(input_dir=r"I:\Geoparticle-simulation\result_compressedfield", output_dir=r"I:\Geoparticle-simulation\result_compressedfield")