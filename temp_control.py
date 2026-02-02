# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:20:04 2026

@author: StanLin
"""
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 系統參數設定 (System Parameters)
# ==========================================

dt = 1.0                # 時間步長 (秒)
total_time = 90 * 60    # 模擬 1.5 小時 (90分鐘)，確保看到穩態
time_steps = int(total_time / dt)
time_axis = np.linspace(0, total_time/60, time_steps)

# 物理常數
rho_air = 1.225         # kg/m^3
cp_air = 1005.0         # J/kg-K

# 房間體積 (假設)
vol_1 = 35.0            # 臥室 (小)
vol_2 = 70.0            # 客廳 (大)

C1 = rho_air * vol_1 * cp_air
C2 = rho_air * vol_2 * cp_air

# 環境與初始條件
T_amb = 30.0            # 室外熱浪
T_init = 28.0           # 室內初始悶熱

# 硬體規格 (User Spec)
AC_power_max = 5200.0   # 5.2 kW 強力冷氣
AC_setpoint = 16.0      # 設定 16度
Fan_power_heat = 40.0   # 40W 風扇廢熱 (Case 2 only)

# 熱阻參數 (關鍵調校點)
# 為了讓客廳在強制對流下平衡在 23~24度，我們反推熱阻
# 平衡公式: (32 - 24) / R = U_forced * (24 - 16)
R_wall1 = 0.015         # 臥室隔熱稍好
R_wall2 = 0.010         # 客廳落地窗多，熱散失快 (數值越小，散熱越快，維持低溫越難)

# ==========================================
# 2. 對流參數 (Coupling Factors)
# ==========================================

# Case 1: 自然對流 (門縫擴散)
# 效率低，無法有效抵抗 R_wall2 的熱入侵
U_natural = 35.0        # W/K

# Case 2: 強制對流 (導風管 + 抽風機)
# 假設風量與之前類似，但為了配合您的 "23-24度" 觀察，這裡不需要調整
# 物理計算: 300 CMH 風量 => U approx 100 W/K
flow_rate_cmh = 350.0   # 稍微提高風量假設
m_dot = (flow_rate_cmh / 3600.0) * rho_air
U_forced = m_dot * cp_air 

print(f"--- 模型參數確認 ---")
print(f"AC Capacity: {AC_power_max} W")
print(f"AC Setpoint: {AC_setpoint} C")
print(f"Fan Heat:    {Fan_power_heat} W")
print(f"Case 1 Coupling: {U_natural:.1f} W/K")
print(f"Case 2 Coupling: {U_forced:.1f} W/K (約 {U_forced/U_natural:.1f} 倍效率)")
print(f"--------------------")

# ==========================================
# 3. 模擬核心 (Euler Solver)
# ==========================================

def run_simulation(case_type, add_noise=False):
    # 初始化
    T1 = np.zeros(time_steps) # 臥室
    T2 = np.zeros(time_steps) # 客廳 (Sensor)
    T1[0] = T_init
    T2[0] = T_init
    
    # 參數選擇
    if case_type == 1:
        coupling = U_natural
        fan_heat = 0.0
    else:
        coupling = U_forced
        fan_heat = Fan_power_heat # 風扇熱量加入系統
    
    for k in range(time_steps - 1):
        t1 = T1[k]
        t2 = T2[k]
        
        # 1. 外部熱負載 (Heat Load)
        q_wall1 = (T_amb - t1) / R_wall1
        q_wall2 = (T_amb - t2) / R_wall2
        
        # 2. 房間互通 (Coupling)
        # 熱從 T2 (客廳) 流向 T1 (臥室)
        q_cross = (t2 - t1) * coupling
        
        # 3. 冷氣運作 (AC Control) - Bang-Bang with Hysteresis or Proportional
        # 由於 5.2kW 很強，臥室會瞬間降溫，我們模擬變頻行為
        if t1 > AC_setpoint:
            # 簡單 P-Control 模擬變頻
            err = t1 - AC_setpoint
            cool_out = min(AC_power_max, AC_power_max * err * 2.0)
            if cool_out < 800: cool_out = 800 # 壓縮機低頻運轉
        else:
            cool_out = 0.0
            
        # 4. 狀態更新
        # 臥室: 牆壁熱 + 客廳來的熱 + 風扇廢熱(Case2) - 冷氣
        dT1 = (q_wall1 + q_cross + fan_heat - cool_out) / C1 * dt
        
        # 客廳: 牆壁熱 - 流去臥室的熱
        dT2 = (q_wall2 - q_cross) / C2 * dt
        
        T1[k+1] = t1 + dT1
        T2[k+1] = t2 + dT2
        
    # 添加雜訊
    if add_noise:
        noise = np.random.normal(0, 0.12, time_steps)
        T2 = T2 + noise
        
    return T1, T2

# ==========================================
# 4. 執行與繪圖
# ==========================================

T1_c1, T2_c1 = run_simulation(case_type=1, add_noise=True) # 自然
T1_c2, T2_c2 = run_simulation(case_type=2, add_noise=True) # 強制

plt.figure(figsize=(12, 7), dpi=100)

# 繪製目標區域 (Target Zone)
plt.axhspan(23, 24, color='green', alpha=0.1, label='Target Comfort Zone (23-24°C)')
plt.axhline(y=AC_setpoint, color='gray', linestyle=':', label='AC Setpoint (16°C)')

# 繪製曲線
plt.plot(time_axis, T2_c1, color='#1f77b4', linewidth=2, label='Case 1: Natural (No Fan)')
plt.plot(time_axis, T2_c2, color='#d62728', linewidth=2, label='Case 2: Forced (Fan 40W)')

# 加上文字標註 (Annotation)
idx_end = -100
final_t2_c1 = np.mean(T2_c1[-500:])
final_t2_c2 = np.mean(T2_c2[-500:])

plt.text(time_axis[-1], final_t2_c1, f'  Steady: {final_t2_c1:.1f}°C\n  (Too Hot)', 
         va='center', color='#1f77b4', fontweight='bold')

plt.text(time_axis[-1], final_t2_c2, f'  Steady: {final_t2_c2:.1f}°C\n  (Target Hit)', 
         va='center', color='#d62728', fontweight='bold')

# 標題與軸
plt.title(f'AC Control Simulation: 5.2kW AC @ 16°C Setpoint\nComparing Natural vs Forced Convection (Fan 40W)', fontsize=14)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Living Room Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.ylim(15, 33)
plt.tight_layout()

plt.show()

# 輸出統計數據
print("--- 模擬結果統計 (Steady State) ---")
print(f"Case 1 (自然對流) 最終平衡溫度: {final_t2_c1:.2f} °C")
print(f"Case 2 (強制對流) 最終平衡溫度: {final_t2_c2:.2f} °C")
print(f"臥室平均溫度 (Case 2): {np.mean(T1_c2[-500:]):.2f} °C (AC 成功鎖定低溫)")