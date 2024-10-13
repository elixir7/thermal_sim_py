from thermal_sim import ThermalMass, ThermalResistance, ThermalSystem, PowerSource

def pwm(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 500
    period = 3600
    duty = 0.5
    return P if (t % period) < duty*period else 0

if __name__ == "__main__": 
    # Simulate a paralell connected thermal mass and leakage resistance with a power source connected.
    # A simple floor heating model with a constant power source (PWM controlled)

    # Create elements
    GND = ThermalMass.get_ground(T=0) # A thermal ground is always reqquired
    floor_name = "Floor (concrete)"
    m_floor = ThermalMass(C=1440 * 440, name=floor_name) # [J/°C]
    R_floor_gnd = ThermalResistance(R=0.01) # [°C/W]
    heat_source = PowerSource(pwm)

    # Connection is done by connecting thermal masses to the elements via connect()
    m_floor.connect(heat_source)
    m_floor.connect(R_floor_gnd, GND)

    # Simulate
    time_step = 60 # [s]
    total_time = 1 * 24 * 3600 # [s]
    system = ThermalSystem(m_floor)
    system.simulate(time_step, total_time)



