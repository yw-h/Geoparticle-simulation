粒子模拟类：prtsim3.py 使用了scipy自带的runge-kutta方法求解速度微分方程，代码更简洁
          sim_pulse_mag.py 功能同上，代码比较繁琐，但是修改性比较强
多线程模拟：multi_sim_randomstate.py 多线程模拟工作，文件内设置粒子能量位置，模拟粒子数，batch大小等
数据处理：raw_result_process.py 处理打包好的粒子数据，按照初始能量等分类。
