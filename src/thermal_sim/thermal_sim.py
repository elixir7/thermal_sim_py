from __future__ import annotations
from typing import List, Callable
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import matplotlib.pyplot as plt

class ThermalElement(ABC):
    @abstractmethod
    def heat_flow(self, t: float, system: ThermalSystem):
        pass

class ThermalResistance(ThermalElement):
    def __init__(self, R:float, name:str=None):
        self.__name__ = "R" if name is None else name
        self.name = self.__name__
        self.R = R
        self.TM1 = None
        self.TM2 = None
        self.connected = False

    def heat_flow(self, t:float, system:ThermalSystem):
        assert self.connected
        return -(self.TM1.T - self.TM2.T) / self.R

    def connect(self, TM1:ThermalMass, TM2:ThermalMass):
        assert isinstance(TM1, ThermalMass), "TM1 must be a ThermalMass"
        assert isinstance(TM2, ThermalMass), "TM2 must be a ThermalMass"
        self.TM1 = TM1
        self.TM2 = TM2
        self.connected = True
        TM1._connect(self)
        TM2._connect(self)

class PowerSource(ThermalElement):
    def __init__(self, heat_flow_func: Callable[[float, float, ThermalSystem], float], name:str=None):
        self.__name__ = "PS" if name is None else name
        self.dQdt = heat_flow_func
        self.prev_dQdt = 0 #  Store previous heat flow for easier implementation of controllers

    def heat_flow(self, t, system):
        self.prev_dQdt = self.dQdt(t, self.prev_dQdt, system)
        return self.prev_dQdt

class TemperatureSource(ThermalElement):
    def __init__(self, temp_func: Callable[[float], ThermalSystem], name:str=None):
        self.__name__ = "TS" if name is None else name
        self.temp_func = temp_func
        self.TM = ThermalMass(C=1e20)
        self.R = ThermalResistance(1e-9)

    def heat_flow(self, t:float, system: ThermalSystem):
        self.TM.T = self.temp_func(t, system)
        return 0

    def connect(self, TM:ThermalMass):
        self.TM.connect(self.R, TM)

class ThermalMass():
    ground_instance = None

    def __init__(self, C, T_initial=0, name:str=None):
        self.T = T_initial
        self.C = C
        self.connected_elements = []
        self.__name__ = "TM" if name is None else name

    def update(self, t:float):
        self.T = t

    @classmethod
    def get_ground(cls, T:float=0):
        if cls.ground_instance is None:
            cls.ground_instance = cls(C=1e20, T_initial=T, name="GND")
        return cls.ground_instance

    def connect(self, element1:ThermalElement|ThermalMass, element2:ThermalMass=None):
        if element2 is not None:
            if isinstance(element1, ThermalResistance) and isinstance(element2, ThermalMass):
                element1.connect(self, element2)
            else:
                raise ValueError("Invalid connection")
            return

        if isinstance(element1, TemperatureSource):
            TS = element1
            TS.connect(self)
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
        self.time = 0

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

    def step(self, dt:float, solver:Solver=Solver.EULER):
        y = [mass.T for mass in self.thermal_masses]
        t = [self.time, self.time+dt]

        assert solver in Solver, "Invalid solver"
        if solver == Solver.ODEINT:
            solution = odeint(self.dydt, y, t)
        else:
            solution = self.custom_ode_solver(y, t, solver)

        # Update the simulation parameters for the next step
        for i, mass in enumerate(self.thermal_masses):
            mass.update(solution[1, i])

        self.time += dt

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

    def generate_diagram(self, filename: str = 'thermal_system_diagram.png'):
        """
        Generate a basic node diagram of the thermal system and save it as an image.

        Args:
        filename (str): The name of the file to save the diagram (default: 'thermal_system_diagram.png').
        """
        import networkx as nx

        # Create a new graph
        G = nx.Graph()

        # Add nodes for thermal masses
        for mass in self.thermal_masses:
            G.add_node(mass.__name__, node_type='thermal_mass')

        # Add ground node
        ground = ThermalMass.get_ground()
        G.add_node(ground.__name__, node_type='ground')

        # Add edges for connections
        for mass in self.thermal_masses:
            for element in mass.connected_elements:
                if isinstance(element, ThermalResistance):
                    other_mass = element.TM1 if element.TM2 == mass else element.TM2
                    G.add_edge(mass.__name__, other_mass.__name__, element_type='resistance', label=element.__name__)
                elif isinstance(element, PowerSource):
                    source_name = f"{element.__name__}\n(Power)"
                    G.add_node(source_name, node_type='power_source')
                    G.add_edge(mass.__name__, source_name, element_type='power_source')
                elif isinstance(element, TemperatureSource):
                    source_name = f"{element.__name__}\n(Temp)"
                    G.add_node(source_name, node_type='temp_source')
                    G.add_edge(mass.__name__, source_name, element_type='temp_source')

        # Set up the plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.9, iterations=50)

        # Draw nodes
        thermal_masses = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'thermal_mass']
        ground_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'ground']
        source_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') in ['power_source', 'temp_source']]

        nx.draw_networkx_nodes(G, pos, nodelist=thermal_masses, node_color='lightblue', node_size=3000, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=ground_nodes, node_color='lightgreen', node_size=3000, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='lightyellow', node_size=3000, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(G, pos)

        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

        # Add edge labels for resistances
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Remove axis
        plt.axis('off')

        # Save the diagram
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Diagram saved as {filename}")

    def generate_thermal_schematic(self, filename: str = 'thermal_schematic.png'):
        """
        !!! Not working well, needs to be fixed !!!
        Generate a schematic diagram of the thermal system using Schemdraw with a grid system.

        Args:
        filename (str): The name of the file to save the schematic (default: 'thermal_schematic.png').
        """
        import schemdraw
        import schemdraw.elements as elm

        d = schemdraw.Drawing()

        # Dictionary to keep track of drawn elements
        drawn_elements = {}

        # Grid settings
        grid_step = 3
        current_x = 0
        current_y = 0

        def draw_thermal_mass(d, mass, x, y):
            if mass not in drawn_elements:
                label = mass.__name__
                if mass == ThermalMass.get_ground():
                    return  # We'll draw separate grounds for each connection
                else:
                    element = d.add(elm.Capacitor().down().label(label).at((x, y)))
                    d.add(elm.Ground().at(element.end))
                drawn_elements[mass] = element

        def draw_thermal_resistance(d, resistance, start_mass, end_mass):
            if resistance not in drawn_elements:
                start = drawn_elements[start_mass].start
                if end_mass == ThermalMass.get_ground():
                    end = (start[0], start[1] - grid_step)
                    d.add(elm.Resistor().down().label(resistance.__name__).at(start).to(end))
                    d.add(elm.Ground().at(end))
                else:
                    end = drawn_elements[end_mass].start
                    d.add(elm.Resistor().label(resistance.__name__).at(start).to(end))
                drawn_elements[resistance] = True

        def draw_power_source(d, source, mass):
            if source not in drawn_elements:
                mass_element = drawn_elements[mass]
                element = d.add(elm.SourceI().left().label(source.__name__)).at(mass_element.start)
                d.add(elm.Ground().at(element.end))
                drawn_elements[source] = True

        def draw_temperature_source(d, source, mass):
            if source not in drawn_elements:
                mass_element = drawn_elements[mass]
                d.add(elm.SourceV().up().label(source.__name__).at(mass_element.center).length(grid_step))
                drawn_elements[source] = True

        # Draw thermal masses
        for mass in self.thermal_masses:
            draw_thermal_mass(d, mass, current_x, current_y)
            current_x += grid_step

        # Draw connections
        for mass in self.thermal_masses:
            for element in mass.connected_elements:
                if isinstance(element, ThermalResistance):
                    other_mass = element.TM1 if element.TM2 == mass else element.TM2
                    draw_thermal_resistance(d, element, mass, other_mass)
                elif isinstance(element, PowerSource):
                    draw_power_source(d, element, mass)
                elif isinstance(element, TemperatureSource):
                    draw_temperature_source(d, element, mass)

        # Save the schematic
        d.save(filename)
        print(f"Schematic saved as {filename}")


def pwm(t: float, prev_Q: float, system: ThermalSystem) -> float:
    P = 1000
    period = 3600
    duty = 0.5
    if t > 3600*12:
        duty = 0
    return P if (t % period) < duty*period else 0

if __name__ == "__main__":
    T0 = 17
    GND = ThermalMass.get_ground(T=T0)


    # Create elements
    floor_name = "Floor"
    m_floor = ThermalMass(C=1440 * 440, T_initial=T0, name=floor_name)
    m_air = ThermalMass(C=1440 * 44, T_initial=T0, name="Air")
    R_floor_gnd = ThermalResistance(R=0.01)
    R_air_gnd = ThermalResistance(R=0.05)
    R_floor_air = ThermalResistance(R=0.05)

    heat_source = PowerSource(pwm, name="Heating cable")
    sun_source = PowerSource(sun, name="Sun")
    atmos = TemperatureSource(lambda t, system: 10)

    # Connect elements
    R_air_gnd.connect(m_air, GND)
    R_floor_air.connect(m_floor, m_air)
    m_floor.connect(heat_source)
    R_floor_gnd.connect(m_floor, GND)

    # Simulate
    thermal_masses = [m_floor, m_air]
    system = ThermalSystem(thermal_masses)
    system.generate_diagram('node_diagram.png')
    dt = 60 # [s]
    total_time = 2 * 24 * 3600 # [s]
    system.simulate(dt, total_time)