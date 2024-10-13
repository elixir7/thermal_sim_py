from __future__ import annotations
from typing import List, Callable
from abc import ABC, abstractmethod
from math import sin, pi
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class ThermalElement(ABC):
    @abstractmethod
    def heat_flow(self, t: float, system: ThermalSystem):
        pass

class ThermalResistance(ThermalElement):
    def __init__(self, R:float, name:str=None):
        self.__name__ = "R"
        self.name = name
        self.R = R
        self.TM1 = None
        self.TM2 = None
        self.connected = False

    def heat_flow(self, t:float, system:ThermalSystem):
        assert self.connected
        return -(self.TM1.T - self.TM2.T) / self.R

    def connect(self, TM1:ThermalMass, TM2:ThermalMass):
        assert isinstance(TM1, ThermalMass), "TM1 must be a Mass"
        assert isinstance(TM2, ThermalMass), "TM2 must be a Mass"
        self.TM1 = TM1
        self.TM2 = TM2
        self.connected = True
    
class PowerSource(ThermalElement):
    def __init__(self, heat_flow_func: Callable[[float, float, ThermalSystem], float], name:str=None):
        self.__name__ = "PS" if name is None else name
        self.dQdt = heat_flow_func
        self.prev_dQdt = 0 #  Store previous heat flow for easier implementation of controllers

    def heat_flow(self, t, system):
        self.prev_dQdt = self.dQdt(t, self.prev_dQdt, system)
        return self.prev_dQdt

class TemperatureSource(ThermalElement):
    def __init__(self, temp_func, name:str=None):
        self.__name__ = "TS" if name is None else name
        self.temp_func = temp_func
        self.TM = ThermalMass(C=1e12, T_initial=temp_func(0))
        self.R = ThermalResistance(1e-9)

    def heat_flow(self, t, system):
        return 0
    
class ThermalMass():
    ground_instance = None

    def __init__(self, C, T_initial=0, name:str=None):
        self.T = T_initial
        self.C = C
        self.connected_elements = []
        self.__name__ = "TM" if name is None else name

    @classmethod
    def get_ground(cls, T:float=0):  
        if cls.ground_instance is None:
            cls.ground_instance = cls(C=1e20, T_initial=T, name="GND")
        return cls.ground_instance

    def connect(self, element1:ThermalElement|ThermalMass, element2:ThermalMass=None):
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
        if element not in self.connected_elements:
            self.connected_elements.append(element)

class Solver(Enum):
    RK4 = "rk4"
    EULER = "euler"
    ODEINT = "odeint"

class ThermalSystem:
    def __init__(self, thermal_masses:List[ThermalMass], GND:ThermalMass=None):
        # No ground specified, create a new ground instance
        if GND is None:
            GND = ThermalMass.get_ground()

        if not isinstance(thermal_masses, list):
            thermal_masses = [thermal_masses]

        assert all(isinstance(mass, ThermalMass) for mass in thermal_masses), "All elements in thermal_masses must be ThermalMass instances"
        assert isinstance(GND, ThermalMass), "GND must be a ThermalMass instance"
        self.thermal_masses = thermal_masses
        self.heat_flows = {}  # New dictionary to store heat flows between elements

    def dydt(self, y, t): 
        temperatures = y if y.size > 1 else [y]
        
        # Update temperatures in system elements
        for mass, T in zip(self.thermal_masses, temperatures):
            mass.T = T
        
        dy_dt = np.zeros((len(self.thermal_masses),))

        for i, mass in enumerate(self.thermal_masses):
            Q = 0
            for element in mass.connected_elements:
                if isinstance(element, ThermalResistance):
                    heat_flow = element.heat_flow(t, self)
                    if element.TM2 == mass:
                        heat_flow = -heat_flow
                else:
                    heat_flow = element.heat_flow(t, self)
                Q += heat_flow
                
                # Store heat flow for each element for plotting
                if mass not in self.heat_flows:
                    self.heat_flows[mass] = {}
                if element not in self.heat_flows[mass]:
                    self.heat_flows[mass][element] = []
                self.heat_flows[mass][element].append((t, heat_flow))

            dTdt = Q / mass.C
            dy_dt[i] = dTdt
        return dy_dt
    
    def rk4_step(self, y, t:float, dt:float):
        k1 = self.dydt(y, t)
        k2 = self.dydt(y + 0.5*dt*k1, t + 0.5*dt)
        k3 = self.dydt(y + 0.5*dt*k2, t + 0.5*dt)
        k4 = self.dydt(y + dt*k3, t + dt)
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def euler_step(self, y, t:float, dt:float):
        return y + dt * self.dydt(y, t)

    def custom_ode_solver(self, y0, t:float, solver:Solver):
        y = np.array(y0)
        dt = t[1] - t[0]
        solution = np.zeros((len(t), len(y0)))

        for i, time in enumerate(t):
            solution[i] = y
            if solver is Solver.RK4:
                y = self.rk4_step(y, time, dt)
            elif solver is Solver.EULER:
                y = self.euler_step(y, time, dt)
            else:
                raise ValueError("Solver not found")

        return solution

    def simulate(self, dt:float, duration:float, plot:bool=True, solver:Solver=Solver.EULER):
        time = np.arange(0, duration, dt)
        y0 = [mass.T for mass in self.thermal_masses]
        
        assert solver in Solver, "Invalid solver"
        if solver == Solver.ODEINT:
            solution = odeint(self.dydt, y0, time)
        else:
            solution = self.custom_ode_solver(y0, time, solver) 

        if plot:
            self.plot_results(time, solution)

        return time, solution

    def _get_heat_flow_label(self, element, mass):
        if isinstance(element, ThermalResistance):
            connected_masses = [element.TM1, element.TM2]
            connected_masses = [m for m in connected_masses if m != mass]
            if connected_masses:
                return f"{connected_masses[0].__name__} -> {element.__name__} -> {mass.__name__}"
            else:
                return f"{element.__name__} -> {mass.__name__}"
        else:
            return f"{element.__name__} -> {mass.__name__}"
        
    def plot_results(self, time, solution, plot_heat_flows:bool=True, plot_ground:bool=False):
        N_masses = min(solution.shape)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10), sharex=True)
        fig.suptitle('Simulation results')

        for i in range(N_masses):
            if self.thermal_masses[i] == ThermalMass.get_ground() and not plot_ground:
                continue
            ax1.plot(time / 3600, solution[:, i], label=self.thermal_masses[i].__name__)

        # Heat flow between elements
        if plot_heat_flows:
            for mass, elements in self.heat_flows.items():
                for element, heat_flow_data in elements.items():
                    times, heat_flows = zip(*heat_flow_data)
                    ax2.plot(np.array(times) / 3600, heat_flows, label=self._get_heat_flow_label(element, mass))

        ax1.set(xlabel="Time (hours)", ylabel="Temperature (°C)")
        ax2.set(xlabel="Time (hours)", ylabel="Heat Flow (W)")
        
        # Adjust axis
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.legend()

        plt.show(block=True)


def pwm(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 1000
    period = 3600
    duty = 0.5
    if t > 3600*12:
        duty = 0
    return P if (t % period) < duty*period else 0

def sun(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 100
    f = 1/(24*3600)
    return P * sin(2*pi*f*t)



if __name__ == "__main__":
    T0 = 17
    GND = ThermalMass.get_ground(T=T0)

    
    # Create elements
    floor_name = "Floor"
    m_floor = ThermalMass(C=1440 * 440, T_initial=T0, name=floor_name)
    m_air = ThermalMass(C=1440 * 44, T_initial=T0, name="Air")
    R_floor_gnd = ThermalResistance(R=0.01)
    R_floor_air = ThermalResistance(R=0.05)
    R_air_gnd = ThermalResistance(R=0.05)
    
    heat_source = PowerSource(pwm)

    sun_source = PowerSource(sun, name="Sun")

    # Connect elements
    m_air.connect(R_air_gnd, GND)
    m_air.connect(sun_source)
    m_floor.connect(heat_source)
    m_floor.connect(R_floor_gnd, GND)
    m_floor.connect(R_floor_air, m_air) # Connect floor to air


    # Simulate
    thermal_masses = [m_floor, m_air]
    system = ThermalSystem(thermal_masses)
    dt = 60 # [s]
    total_time = 1 * 24 * 3600 # [s]
    system.simulate(dt, total_time)