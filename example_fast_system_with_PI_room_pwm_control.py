from math import sin, pi
from src.thermal_sim.thermal_sim import ThermalMass, ThermalResistance, ThermalSystem, PowerSource, TemperatureSource, Solver

def clip(n, min, max): 
    sat = n
    if n < min: 
        sat =  min
    elif n > max: 
        sat =  max
    
    return (sat, n != sat)


def PI(e, integral, Kp, Ki, dt, feedforward):
    prev_integral = integral
    out_min = 0
    out_max = 1

    int_min = 0
    int_max = 1

    # Proportional term
    P = Kp * e

    # Update integral term
    integral += e * dt

    # Calculate the integral contribution and clamp is within bounds
    I = Ki * integral
    integral_saturated = False
    if I > int_max:
        I = int_max
        integral = int_max / Ki
        integral_saturated = True
    elif I < int_min:
        I = int_min
        integral = int_min / Ki
        integral_saturated = True

    # Calculate the output and check for saturation
    setpoint_ref_raw = P + I + feedforward
    setpoint_ref_sat, output_saturated = clip(setpoint_ref_raw, out_min, out_max)

    # Prevent integral windup
    if output_saturated and not integral_saturated:
        integral = prev_integral

    return setpoint_ref_sat, integral
    



def bangbang(T, T_ref, T_hyst, prev_u):
    T_max = T_ref + T_hyst
    T_min = T_ref - T_hyst

    if T > T_max:
        u = 0
    elif T < T_min:
        u = 1
    else:
        u = prev_u

    return u

def pwm(t: float, duty:float, period: float) -> float:
    return 1 if (t % period) < duty*period else 0

def get_temp_by_name(system: ThermalSystem, name: str):
    for mass in system.thermal_masses:
        if mass.__name__ == name:
            return mass.T
    return None



if __name__ == "__main__":
    # Simulate floor and air connected model with air and floor feedback (PI + Bangbang)

    T0 = 19
    GND = ThermalMass.get_ground(T=T0)

    floor_name = "Floor"
    air_name = "Air"
    
    def controller(t: float, prev_Q: float, system: ThermalSystem) -> float:
        if not hasattr(controller, "integral"):
            controller.integral = 0
            controller.t_prev = 0
        
        P = 2000
        u_prev = 1 if prev_Q > 0 else 0

        # TRM can only run at 1s update rate so make sure we do that
        dt_last_run = t - controller.t_prev
        if (dt_last_run) < 1:
            return u_prev * P

        T_air = get_temp_by_name(system, air_name)
        T_ref = 24
        if t > system.t_max/2:
            T_ref = 20
        

        e = T_ref - T_air
        Kp = 1/3
        Ti = 300*4 # Time constant in seconds
        Ki = Kp / Ti
        
        # Kp = 1/3
        # Ki = 0
        pwm_period = 5*60
        

        setpoint_duty, integral = PI(e, controller.integral, Kp, Ki, dt_last_run, 0)
        controller.integral = integral
        u = pwm(t, duty=setpoint_duty, period=pwm_period)
        
        controller.t_prev = t
        return u * P
    


    # Setup physcal system elements
    m_floor = ThermalMass(C=1000*4*5, T_initial=T0, name=floor_name)
    m_air = ThermalMass(C=1000*8*5, T_initial=T0, name=air_name)

    R_floor_gnd = ThermalResistance(R=0.01, name="R_f_gnd")
    R_air_gnd = ThermalResistance(R=0.1, name="R_a_gnd")
    R_floor_air = ThermalResistance(R=0.05, name="R_f_a")
    R_roof_gnd = ThermalResistance(R=0.01, name="R_r_gnd")
    R_air_roof = ThermalResistance(R=1, name="R_a_r")

    heat_source = PowerSource(controller, name="Heating cable")

    # Connect elements
    m_floor.connect(heat_source)
    R_floor_gnd.connect(m_floor, GND)

    # m_air.connect(sun_source)
    R_air_gnd.connect(m_air, GND)
    R_floor_air.connect(m_floor, m_air)

    # Simulate
    thermal_masses = [m_floor, m_air]
    system = ThermalSystem(thermal_masses)
    system.generate_diagram('img/multi_mass.png')
    dt = 1 # [s]
    total_time = 3600*2 # [s]
    system.simulate(dt, total_time, solver=Solver.RK4)