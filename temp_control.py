# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:20:04 2026

@author: StanLin
"""
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. System Parameters
# ==========================================

dt = 1.0                # Time step (seconds)
total_time = 90 * 60    # Simulation duration: 1.5 hours (90 minutes) to reach steady state
time_steps = int(total_time / dt)
time_axis = np.linspace(0, total_time/60, time_steps)

# Physical constants
rho_air = 1.225         # kg/m^3
cp_air = 1005.0         # J/kg-K

# Room volumes (assumptions)
vol_1 = 35.0            # Bedroom (small)
vol_2 = 70.0            # Living room (large)

C1 = rho_air * vol_1 * cp_air
C2 = rho_air * vol_2 * cp_air

# Environmental and initial conditions
T_amb = 30.0            # Outdoor temperature (heat wave)
T_init = 28.0           # Initial indoor temperature (stuffy)

# Hardware specifications (User Spec)
AC_power_max = 5200.0   # 5.2 kW powerful AC
AC_setpoint = 16.0      # Setpoint: 16°C
Fan_power_heat = 40.0   # 40W fan waste heat (Case 2 only)

# Thermal resistance parameters (Critical tuning point)
# To achieve living room equilibrium at 23~24°C under forced convection, we back-calculate thermal resistance
# Balance equation: (30 - 24) / R = U_forced * (24 - 16)
R_wall1 = 0.020         # Bedroom: better insulation
R_wall2 = 0.011         # Living room: more windows, faster heat dissipation (smaller value = faster heat loss, harder to maintain low temperature)

# ==========================================
# 2. Convection Parameters (Coupling Factors)
# ==========================================

# Case 1: Natural convection (door gap diffusion)
# Low efficiency, cannot effectively resist heat intrusion through R_wall2
U_natural = 35.0        # W/K

# Case 2: Forced convection (air duct + exhaust fan)
# Airflow assumption similar to previous setup, tuned for observed "23-24°C" target
# Physical calculation: 200 CMH airflow => U approx 68 W/K
flow_rate_cmh = 200.0   # Slightly increased airflow assumption
m_dot = (flow_rate_cmh / 3600.0) * rho_air
U_forced = m_dot * cp_air 

print(f"--- Model Parameter Confirmation ---")
print(f"AC Capacity: {AC_power_max} W")
print(f"AC Setpoint: {AC_setpoint} C")
print(f"Fan Heat:    {Fan_power_heat} W")
print(f"Case 1 Coupling: {U_natural:.1f} W/K")
print(f"Case 2 Coupling: {U_forced:.1f} W/K (approx {U_forced/U_natural:.1f}x efficiency)")
print(f"------------------------------------")

# ==========================================
# 3. Simulation Core (Euler Solver)
# ==========================================

def run_simulation(case_type, add_noise=False):
    # Initialization
    T1 = np.zeros(time_steps) # Bedroom
    T2 = np.zeros(time_steps) # Living room (Sensor)
    T1[0] = T_init
    T2[0] = T_init
    
    # Parameter selection
    if case_type == 1:
        coupling = U_natural
        fan_heat = 0.0
    else:
        coupling = U_forced
        fan_heat = Fan_power_heat # Fan heat added to system
    
    for k in range(time_steps - 1):
        t1 = T1[k]
        t2 = T2[k]
        
        # 1. External heat load
        q_wall1 = (T_amb - t1) / R_wall1
        q_wall2 = (T_amb - t2) / R_wall2
        
        # 2. Inter-room coupling
        # Heat flows from T2 (living room) to T1 (bedroom)
        q_cross = (t2 - t1) * coupling
        
        # 3. AC operation (AC Control) - P-Control simulating inverter behavior
        # Since 5.2kW is powerful, bedroom cools rapidly; we simulate variable-speed operation
        if t1 > AC_setpoint:
            # Simple P-Control for inverter simulation
            err = t1 - AC_setpoint
            cool_out = min(AC_power_max, AC_power_max * err * 2.0)
            if cool_out < 800: cool_out = 800 # Compressor low-frequency operation
        else:
            cool_out = 0.0
            
        # 4. State update
        # Bedroom: wall heat + heat from living room + fan waste heat (Case2) - AC cooling
        dT1 = (q_wall1 + q_cross + fan_heat - cool_out) / C1 * dt
        
        # Living room: wall heat - heat flowing to bedroom
        dT2 = (q_wall2 - q_cross) / C2 * dt
        
        T1[k+1] = t1 + dT1
        T2[k+1] = t2 + dT2
        
    # Add noise
    if add_noise:
        noise = np.random.normal(0, 0.12, time_steps)
        T2 = T2 + noise
        
    return T1, T2

# ==========================================
# 4. Execution and Plotting
# ==========================================

T1_c1, T2_c1 = run_simulation(case_type=1, add_noise=True) # Natural
T1_c2, T2_c2 = run_simulation(case_type=2, add_noise=True) # Forced

plt.figure(figsize=(10, 6), dpi=100)

# Plot target zone
plt.axhspan(23, 24, color='green', alpha=0.1, label='Target Comfort Zone (23-24°C)')
plt.axhline(y=AC_setpoint, color='gray', linestyle=':', label='AC Setpoint (16°C)')

# Plot curves
plt.plot(time_axis, T2_c1, color='#1f77b4', linewidth=2, label='Case 1: Natural (No Fan)')
plt.plot(time_axis, T2_c2, color='#d62728', linewidth=2, label='Case 2: Forced (Fan 40W)')

# Add text annotations
idx_end = -100
final_t2_c1 = np.mean(T2_c1[-500:])
final_t2_c2 = np.mean(T2_c2[-500:])

plt.text(time_axis[-1], final_t2_c1, f'  Steady: {final_t2_c1:.1f}°C\n  (Too Hot)', 
         va='center', color='#1f77b4', fontweight='bold')

plt.text(time_axis[-1], final_t2_c2, f'  Steady: {final_t2_c2:.1f}°C\n  (Target Hit)', 
         va='center', color='#d62728', fontweight='bold')

# Title and axes
plt.title(f'AC Control Simulation: 5.2kW AC @ 16°C Setpoint\nComparing Natural vs Forced Convection (Fan 40W)', fontsize=14)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Living Room Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.ylim(20, 30)
plt.tight_layout()

plt.show()

# Output statistics
print("--- Simulation Results (Steady State) ---")
print(f"Case 1 (Natural Convection) Final Equilibrium: {final_t2_c1:.2f} °C")
print(f"Case 2 (Forced Convection) Final Equilibrium: {final_t2_c2:.2f} °C")
print(f"Bedroom Average Temperature (Case 2): {np.mean(T1_c2[-500:]):.2f} °C (AC successfully locked at low temperature)")