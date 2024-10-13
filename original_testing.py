import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ThermalSystem:
    def __init__(self, m1, m2, c1, c2, R, R1_leak, R2_leak, P_max, on_time, off_time):
        self.m1 = m1  # Mass of concrete floor (kg)
        self.m2 = m2  # Mass of air (kg)
        self.c1 = c1  # Specific heat capacity of concrete (J/kg·°C)
        self.c2 = c2  # Specific heat capacity of air (J/kg·°C)
        self.R = R    # Thermal resistance between floor and air (°C/W)
        self.R1_leak = R1_leak  # Leakage resistance of floor (°C/W)
        self.R2_leak = R2_leak  # Leakage resistance of air (°C/W)
        self.P_max = P_max  # Maximum power input to floor (W)
        self.on_time = on_time  # Duration of 'on' state (s)
        self.off_time = off_time  # Duration of 'off' state (s)

    def power_input(self, t):
        cycle_time = self.on_time + self.off_time
        t_in_cycle = t % cycle_time
        return self.P_max if t_in_cycle < self.on_time else 0

    def simulate(self, T1_initial, T2_initial, T_ambient, time_step, total_time):
        n_steps = int(total_time / time_step)
        time = np.arange(0, total_time, time_step)
        power = np.zeros(n_steps)

        y0 = [T1_initial, T2_initial, T_ambient]

        def system_ode(y, t, system):
            P = system.power_input(t)

            # Heat transfer between floor and air
            Q_transfer = (y[0] - y[1]) / self.R
            
            # Heat loss due to leakage
            Q1_leak = (y[0] - y[2]) / system.R1_leak
            Q2_leak = (y[1] - y[2]) / system.R2_leak
            
            # Temperature change for floor
            dT1dt = (P - Q_transfer - Q1_leak) / (self.m1 * self.c1)
            
            # Temperature change for air
            dT2dt = (Q_transfer - Q2_leak) / (self.m2 * self.c2)

            dydt = [dT1dt, dT2dt, 0]
            
            return dydt

        from scipy.integrate import odeint
        solution = odeint(system_ode, y0, time, args=(self,))
        T1 = solution[:,0]
        T2 = solution[:,1]

        return time, T1, T2, power

    def plot_results(self, time, T1, T2, power, T_ambient):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Temperature plot
        ax1.plot(time / 3600, T1, label='Concrete Floor')
        ax1.plot(time / 3600, T2, label='Room Air')
        ax1.axhline(y=T_ambient, color='r', linestyle='--', label='Ambient Temperature')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Evolution of Concrete Floor and Room Air')
        ax1.legend()
        ax1.grid(True)
        # ax1.set_ylim(T_ambient - 5, T_ambient + 20)  # Adjust y-axis limits

        # Power input plot
        ax2.plot(time / 3600, power, label='Power Input')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power Input to Concrete Floor Over Time')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # System parameters for a small room with concrete floor
    room_length = 3  # m
    room_width = 2   # m
    room_height = 2.5  # m
    concrete_thickness = 0.1  # m (increased thickness)
    
    concrete_density = 2400  # kg/m³
     
    
    m1 = room_length * room_width * concrete_thickness * concrete_density  # Mass of concrete floor (kg)
    m2 = room_length * room_width * room_height * air_density  # Mass of air in the room (kg)
    
    c1 = 880/2  # Specific heat capacity of concrete (J/kg·°C)
    c2 = 1005  # Specific heat capacity of air (J/kg·°C)
    
    # Thermal resistance between floor and air (estimated)
    R = 0.3  # °C/W
    
    # Leakage resistances (adjusted for more realistic heat loss)
    R1_leak = 0.05  # Leakage resistance of floor (°C/W)
    R2_leak = 0.2  # Leakage resistance of air (°C/W)
    
    P_max = 500  # Maximum power input to floor (W) 
    on_time = 3600*1  # Duration of 'on' state 
    off_time = 3600*1  # Duration of 'off' state 

    # Initial conditions and simulation parameters
    T_ambient = 18   # Ambient temperature outside the room (°C)
    T1_initial = T_ambient  # Initial temperature of concrete floor (°C)
    T2_initial = T_ambient  # Initial temperature of room air (°C)
    time_step = 1  # Time step for simulation (s)
    total_time = 1 * 24 * 3600  # Total simulation time (7 days in seconds)

    # Create and run simulation
    system = ThermalSystem(m1, m2, c1, c2, R, R1_leak, R2_leak, P_max, on_time, off_time)
    time, T1, T2, power = system.simulate(T1_initial, T2_initial, T_ambient, time_step, total_time)
    system.plot_results(time, T1, T2, power, T_ambient)