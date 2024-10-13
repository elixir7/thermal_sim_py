from thermal_sim import ThermalMass, ThermalResistance, ThermalSystem, PowerSource

if __name__ == "__main__":
    # Simulate a paralell connected thermal mass and leakage resistance controlled with a bang bang controller

    T0 = 17
    GND = ThermalMass.get_ground(T=T0) # non zero "ambient" temperature
    
    # Create elements
    floor_name = "Floor (concrete)"
    m_floor = ThermalMass(C=1440 * 440, T_initial=T0, name=floor_name)
    R_floor_gnd = ThermalResistance(R=0.01)

    def bangbang(t: float, prev_Q: float, system: ThermalSystem) -> float:
        # Ugly way to get the floor mass/temperature
        for mass in system.thermal_masses:
            if mass.name == floor_name:
                floor = mass

        # Bang Bang controller on the floor temperature
        P = 1000
        T_hyst = 2
        T_ref = 20
        T_max = T_ref + T_hyst
        T_min = T_ref - T_hyst
        
        prev_output = 1 if prev_Q > 0 else 0

        if prev_output and floor.T > T_max:
            output = 0
        elif not prev_output and floor.T < T_min:
            output = 1
        else:
            output = prev_output

        return P*output
    
    heat_source = PowerSource(bangbang)

    # Connect elements
    m_floor.connect(heat_source)
    m_floor.connect(R_floor_gnd, GND)

    # Simulate
    system = ThermalSystem(m_floor, GND)
    time_step = 60 # [s]
    total_time = 1 * 24 * 3600 # [s]
    # system.simulate(time_step, total_time, solver="rk4") # Use e.g a more accurate but slower solver

    system = ThermalSystem([m_floor, GND]) # Include ground in thermal masses to see leakage
    system.simulate(time_step, total_time)