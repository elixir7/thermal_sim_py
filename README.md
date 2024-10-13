# Thermal Sim
Create thermal system based on componennts, automatically calculates differetial equations based on connections and simulates using a variety of solvers.
Basic components:
- Thermal Resistor   
- Thermal Mass 
- Power Source
- Temperature source (infinite)


## Usage
### Simulate
Create a basic parallell thermal mass and leakage resistor with a PWM controlled powersource.
```python
from thermal_sim import ThermalSystem, Resistor, Capacitor, PowerSource, TemperatureSource

# Power source (PWM controlled)
def pwm(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 500
    period = 3600
    duty = 0.5
    return P if (t % period) < duty*period else 0

# Create components
GND = ThermalMass.get_ground(T=0) # A thermal ground is always reqquired
m_floor = ThermalMass(C=1440 * 440, name="Floor") # [J/°C] Concrete floor 
R_floor_gnd = ThermalResistance(R=0.01) # [°C/W]
heat_source = PowerSource(pwm)

# Connection is done by connecting thermal masses to the elements via connect()
m_floor.connect(heat_source)
m_floor.connect(R_floor_gnd, GND)


# Simulate
dt = 60 # [s]
total_time = 1 * 24 * 3600 # [s]
system = ThermalSystem(m_floor)
system.simulate(dt, total_time)
```

### Diagram
Create a diagram of the system
```python
system = ThermalSystem(thermal_masses)
system.draw_diagram()
```


## Examples
### PWM controlled RC circuit
<img src="img/example_PWM_RC.png" height="500px" />

## Bang-Bang controlled RC circuit
<img src="img/example_bangbang_RC.png" height="500px" />

### Multiple masses connected
<img src="img/example_multimass.png" height="500px" />
