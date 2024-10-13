from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ThermalElement(ABC):
    @abstractmethod
    def heat_flow(self, t, system):
        pass

class ThermalResistance(ThermalElement):
    def __init__(self, R):
        self.__name__ = "R"
        self.R = R
        self.TM = None
        self.element = None
        self.connected = False

    def heat_flow(self, t, system):
        assert self.connected
        return -(self.TM.T - self.element.T) / self.R

    def connect(self, TM, element):
        assert isinstance(TM, ThermalMass), "TM must be a Mass"
        assert isinstance(element, ThermalMass), "Element must be a Mass"
        self.TM = TM
        self.element = element
        self.connected = True
    
class PowerSource(ThermalElement):
    def __init__(self, heat_flow_func):
        self.__name__ = "PS"
        self.Q = heat_flow_func
        self.prev_Q = 0

    def heat_flow(self, t, system):
        self.prev_Q = self.Q(t, self.prev_Q, system)
        return self.prev_Q

class TemperatureSource(ThermalElement):
    def __init__(self, temp_func):
        self.__name__ = "TS"
        self.temp_func = temp_func
        self.TM = ThermalMass(C=1e12, T_initial=temp_func(0))
        self.R = ThermalResistance(1e-9)

    def heat_flow(self, t, system):
        return 0

class ThermalMass():
    i = 0
    ground_instance = None

    def __init__(self, C, T_initial=0, name=None):
        self.T = T_initial
        self.C = C
        self.connected_elements = []
        if name is None:
            name = f"Thermal mass {ThermalMass.i}"
            ThermalMass.i += 1
        self.name = name

    @classmethod
    def get_ground(cls, T=0):
        if cls.ground_instance is None:
            cls.ground_instance = cls(C=1e20, T_initial=T, name="GND")
        return cls.ground_instance

    def connect(self, element1, element2=None):
        if element2 is not None:
            if isinstance(element1, ThermalResistance) and isinstance(element2, ThermalMass):
                element1.connect(self, element2)
                self._connect(element1)
                element2._connect(element1)
            else:
                raise ValueError("Invalid connection")
            return
        
        if isinstance(element1, TemperatureSource):
            self.connect(element1.TM, element1.R)
            self._connect(element1.R)
            return
        
        self._connect(element1)

    def _connect(self, element):
        self.connected_elements.append(element)

class ThermalSystem:
    def __init__(self, thermal_masses, GND: ThermalMass = None):
        # Assume 0C ground if disregarded
        if GND is None:
            GND = ThermalMass.get_ground(0)

        if not isinstance(thermal_masses, list):
            thermal_masses = [thermal_masses]

        assert all(isinstance(mass, ThermalMass) for mass in thermal_masses), "All elements must be ThermalMass instances"
        assert isinstance(GND, ThermalMass), "GND must be a ThermalMass instance"
        self.thermal_masses = thermal_masses
        self.heat_flows = {}  # New dictionary to store heat flows

    def dydt(self, y, t): 
        temperatures = y if y.size > 1 else [y]
        
        # Update temperatures in system elements
        for mass, T in zip(self.thermal_masses, temperatures):
            mass.T = T
        
        dy_dt = np.zeros((len(self.thermal_masses),))

        for i, mass in enumerate(self.thermal_masses):
            Q = 0
            for element in mass.connected_elements:
                heat_flow = element.heat_flow(t, self)
                Q += heat_flow
                
                # Store heat flow for each element for plotting, TODO: extract this somewhere else since solvers can make multiple calls to dy/dt for each simulation time step
                if mass not in self.heat_flows:
                    self.heat_flows[mass] = {}
                if not isinstance(element, ThermalMass):
                    if element not in self.heat_flows[mass]:
                        self.heat_flows[mass][element] = []
                    self.heat_flows[mass][element].append((t, heat_flow))

            dTdt = Q / mass.C
            dy_dt[i] = dTdt
        return dy_dt
    
    def rk4_step(self, y, t, dt):
        k1 = self.dydt(y, t)
        k2 = self.dydt(y + 0.5*dt*k1, t + 0.5*dt)
        k3 = self.dydt(y + 0.5*dt*k2, t + 0.5*dt)
        k4 = self.dydt(y + dt*k3, t + dt)
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def euler_step(self, y, t, dt):
        return y + dt * self.dydt(y, t)

    def custom_ode_solver(self, y0, t, solver):
        y = np.array(y0)
        dt = t[1] - t[0]
        solution = np.zeros((len(t), len(y0)))

        for i, time in enumerate(t):
            solution[i] = y
            if solver == "rk4":
                y = self.rk4_step(y, time, dt)
            elif solver == "euler":
                y = self.euler_step(y, time, dt)
            else:
                raise ValueError("Solver not found")

        return solution

    def simulate(self, dt: float, duration: float, plot=True, solver=None):
        time = np.arange(0, duration, dt)

        y0 = [mass.T for mass in self.thermal_masses]
        
        if solver is None:
            solution = self.custom_ode_solver(y0, time, "euler") 
        elif solver == "odeint":
            solution = odeint(self.dydt, y0, time)
        else:
            solution = self.custom_ode_solver(y0, time, solver) 

        if plot:
            self.plot_results(time, solution)

        return time, solution

    def plot_results(self, time, solution):
        num_temperatures = min(solution.shape)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.suptitle('Simulation results')

        for i in range(num_temperatures):
            if self.thermal_masses[i] == ThermalMass.get_ground():
                continue
            ax1.plot(time / 3600, solution[:, i], label=self.thermal_masses[i].name)

        # Heat flow between elements
        for mass, elements in self.heat_flows.items():
            for element, heat_flow_data in elements.items():
                times, heat_flows = zip(*heat_flow_data)
                if isinstance(element, ThermalResistance):
                    connected_masses = [element.TM, element.element]
                    connected_masses = [m for m in connected_masses if m != mass]
                    if connected_masses:
                        label = f"{connected_masses[0].name} -> {element.__name__} -> {mass.name}"
                    else:
                        label = f"{element.__name__} -> {mass.name}"
                else:
                    label = f"{element.__name__} -> {mass.name}"
                
                ax2.plot(np.array(times) / 3600, heat_flows, label=label)

        ax1.set(ylabel="Temperature (°C)")
        ax2.set(xlabel="Time (hours)", ylabel="Heat Flow (W)")
        
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.legend()

        plt.show(block=True)




def pwm(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 1000
    period = 3600
    duty = 0.5
    return P if (t % period) < duty*period else 0



if __name__ == "__main__":
    T0 = 17
    GND = ThermalMass.get_ground(T=T0)

    
    # Create elements
    floor_name = "Floor"
    m_floor = ThermalMass(C=1440 * 440, T_initial=T0, name=floor_name)
    m_air = ThermalMass(C=1440 * 44, T_initial=20, name="Air")
    R_floor_gnd = ThermalResistance(R=0.01)
    R_floor_air = ThermalResistance(R=0.05)
    R_air_gnd = ThermalResistance(R=0.05)
    
    heat_source = PowerSource(pwm)

    heat_source2 = PowerSource(pwm)

    # Connect elements
    m_air.connect(R_air_gnd, GND)
    # m_air.connect(heat_source2)

    m_floor.connect(heat_source)
    m_floor.connect(R_floor_gnd, GND)
    
    m_floor.connect(R_floor_air, m_air)

    # Simulate
    thermal_masses = [m_floor, m_air]
    system = ThermalSystem(thermal_masses)
    time_step = 60 # [s]
    total_time = 1 * 24 * 3600 # [s]
    system.simulate(time_step, total_time, solver="euler")