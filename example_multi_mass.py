from math import sin, pi
from src.thermal_sim.thermal_sim import ThermalMass, ThermalResistance, ThermalSystem, PowerSource, TemperatureSource

# Disturbance power source
def sun(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 20
    f = 1/(24*3600)
    return P * sin(2*pi*f*t)

# Floor heating power source
def pwm(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 1000
    period = 3600
    duty = 0.5
    return P if (t % period) < duty*period else 0


if __name__ == "__main__":
    # Simulate multiple connected thermal masses with disturbance
    # Idea: Cable heats floor, floor heats air, air heats roof
    # Disturbance: Sun heats air
    # Leakage: Floor, air and roof, leaks to ground
    T0 = 17
    GND = ThermalMass.get_ground(T=T0)


    # Create elements
    m_floor = ThermalMass(C=1440 * 440, T_initial=T0, name="Floor")
    m_air = ThermalMass(C=1440 * 200, T_initial=T0, name="Air")
    m_roof = ThermalMass(C=1440 * 100, T_initial=T0, name="Roof")

    R_floor_gnd = ThermalResistance(R=0.01, name="R_f_gnd")
    R_air_gnd = ThermalResistance(R=0.1, name="R_a_gnd")
    R_floor_air = ThermalResistance(R=0.05, name="R_f_a")
    R_roof_gnd = ThermalResistance(R=0.01, name="R_r_gnd")
    R_air_roof = ThermalResistance(R=1, name="R_a_r")

    heat_source = PowerSource(pwm, name="Heating cable")
    sun_source = PowerSource(sun, name="Sun")

    # Connect elements
    m_floor.connect(heat_source)
    R_floor_gnd.connect(m_floor, GND)

    m_air.connect(sun_source)
    R_air_gnd.connect(m_air, GND)
    R_floor_air.connect(m_floor, m_air)

    R_air_roof.connect(m_air, m_roof)

    # Simulate
    thermal_masses = [m_floor, m_air, m_roof]
    system = ThermalSystem(thermal_masses)
    system.generate_diagram('img/multi_mass.png')
    dt = 60 # [s]
    total_time = 4 * 24 * 3600 # [s]
    system.simulate(dt, total_time)