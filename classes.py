import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI
from typing import Dict, List, Tuple, Optional, Union


class Fluid:
    """
    Class to handle fluid properties using CoolProp.
    """

    def __init__(self, fluid_name: str,
                 fluid_pressure: float,
                 fluid_temperature: float):
        """
        Initialize a fluid object.

        Args:
            fluid_name: Name of the fluid as recognized by CoolProp
        """
        self.name = fluid_name
        self.p = fluid_pressure
        self.T = fluid_temperature
        self.is_compressible = self._check_if_compressible()

    def _check_if_compressible(self) -> bool:
        """
        Check if the fluid is compressible (gas) or incompressible (liquid).

        Returns:
            bool: True if fluid is compressible, False otherwise
        """
        try:
            # Get fluid phase at inlet conditions
            phase = CP.PhaseSI('P', self.p, 'T', self.T, self.name)
            if phase=='twophase':
                st.error('ISA standard is not applicable to two phase fluids.')
            return phase == "gas"
        except:
            st.warning(f"Could not determine phase of {self.name}. Assuming compressible.")
            return True

    def get_density(self) -> float:
        """
        Get fluid density at given conditions.

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K

        Returns:
            float: Density in kg/m³
        """
        return PropsSI('D', 'P', self.p, 'T', self.T, self.name)

    def get_viscosity(self) -> float:
        """
        Get fluid viscosity at given conditions.

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K

        Returns:
            float: Dynamic viscosity in Pa·s
        """
        return PropsSI('viscosity', 'P', self.p, 'T', self.T, self.name)

    def get_z_factor(self) -> float:
        """
        Get compressibility factor at given conditions.

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K

        Returns:
            float: Compressibility factor (dimensionless)
        """
        if not self.is_compressible:
            return 1.0

        try:
            return PropsSI('Z', 'P', self.p, 'T', self.T, self.name)
        except:
            st.warning(f"Could not calculate Z factor for {self.name}. Using Z=1.")
            return 1.0

    def get_molecular_weight(self) -> float:
        """
        Get molecular weight of the fluid.

        Returns:
            float: Molecular weight in kg/mol
        """
        try:
            return PropsSI('molar_mass', self.name)
        except:
            st.warning(f"Could not get molecular weight for {self.name}. Using default value of 0.029 kg/mol.")
            return 0.029  # Default value (close to air)

    def get_critical_pressure(self) -> float:
        """
        Get critical pressure of the fluid.

        Returns:
            float: Critical pressure in Pa
        """
        try:
            return PropsSI('pcrit', self.name)
        except:
            st.warning(f"Could not get critical pressure for {self.name}. Using default value.")
            return 22064000  # Default value (approximately 220 bar)

    def get_vapour_pressure(self) -> float:
        """
        Get vapour pressure of the fluid.

        Returns
            float: Vapour pressure in Pa
        """
        try:
            return PropsSI('P', 'Q', 1, 'T', self.T, self.name)
        except:
            st.warning(f"Could not get vapour pressure for {self.name}. Using default value.")
            return 0.0



class ValveProperties:
    def __init__(self):
        # Initialize the data dictionary with nested structure:
        # {valve_type: {trim_type: {flow_direction: {'Fl': value, 'xt': value, 'Fd': value}}}}
        self.valve_data = {
            'Globe, single port': {
                '3 V-port plug': {
                    'Open or close': {'Fl': 0.9, 'xt': 0.70, 'Fd': 0.48}
                },
                '4 V-port plug': {
                    'Open or close': {'Fl': 0.9, 'xt': 0.70, 'Fd': 0.41}
                },
                '6 V-port plug': {
                    'Open or close': {'Fl': 0.9, 'xt': 0.70, 'Fd': 0.30}
                },
                'Contoured plug (linear and equal percentage)': {
                    'Open': {'Fl': 0.9, 'xt': 0.72, 'Fd': 0.46},
                    'Close': {'Fl': 0.8, 'xt': 0.55, 'Fd': 1.70}
                },
                '60 equal diameter hole drilled cage': {
                    'Outward': {'Fl': 0.9, 'xt': 0.68, 'Fd': 0.13},
                    'Inward': {'Fl': 0.9, 'xt': 0.68, 'Fd': 0.13}
                },
                '120 equal diameter hole drilled cage': {
                    'Outward': {'Fl': 0.9, 'xt': 0.68, 'Fd': 0.09},
                    'Inward': {'Fl': 0.9, 'xt': 0.68, 'Fd': 0.09}
                },
                'Characterized cage, 4-port': {
                    'Outward': {'Fl': 0.9, 'xt': 0.75, 'Fd': 0.41},
                    'Inward': {'Fl': 0.85, 'xt': 0.70, 'Fd': 0.41}
                }
            },
            'Globe, double port': {
                'Ported plug': {
                    'Inlet between seats': {'Fl': 0.9, 'xt': 0.75, 'Fd': 0.28}
                },
                'Contoured plug': {
                    'Either direction': {'Fl': 0.85, 'xt': 0.70, 'Fd': 0.32}
                }
            },
            'Globe, angle': {
                'Contoured plug (linear and equal percentage)': {
                    'Open': {'Fl': 0.9, 'xt': 0.72, 'Fd': 0.46},
                    'Close': {'Fl': 0.8, 'xt': 0.65, 'Fd': 1.00}
                },
                'Characterized cage, 4-port': {
                    'Outward': {'Fl': 0.9, 'xt': 0.65, 'Fd': 0.41},
                    'Inward': {'Fl': 0.85, 'xt': 0.60, 'Fd': 0.41}
                }
            },
            'Globe, small flow trim': {
                'Venturi': {
                    'Close': {'Fl': 0.5, 'xt': 0.20, 'Fd': 1.00}
                },
                'V-notch': {
                    'Open': {'Fl': 0.98, 'xt': 0.84, 'Fd': 0.70}
                },
                'Flat seat (short travel)': {
                    'Close': {'Fl': 0.85, 'xt': 0.70, 'Fd': 0.30}
                },
                'Tapered needle': {
                    'Open': {'Fl': 0.95, 'xt': 0.84, 'Fd': None}  # Special case with formula
                }
            },
            'Rotary': {
                'Eccentric spherical plug': {
                    'Open': {'Fl': 0.85, 'xt': 0.60, 'Fd': 0.42},
                    'Close': {'Fl': 0.68, 'xt': 0.40, 'Fd': 0.42}
                },
                'Eccentric conical plug': {
                    'Open': {'Fl': 0.77, 'xt': 0.54, 'Fd': 0.44},
                    'Close': {'Fl': 0.79, 'xt': 0.55, 'Fd': 0.44}
                }
            },
            'Butterfly (centered shaft)': {
                'Swing-through (70°)': {
                    'Either': {'Fl': 0.62, 'xt': 0.35, 'Fd': 0.57}
                },
                'Swing-through (80°)': {
                    'Either': {'Fl': 0.70, 'xt': 0.42, 'Fd': 0.50}
                },
                'Fluted vane (70°)': {
                    'Either': {'Fl': 0.67, 'xt': 0.38, 'Fd': 0.30}
                }
            },
            'High Performance Butterfly (eccentric shaft)': {
                'Offset seat (70°)': {
                    'Either': {'Fl': 0.67, 'xt': 0.35, 'Fd': 0.57}
                }
            },
            'Ball': {
                'Full bore (70°)': {
                    'Either': {'Fl': 0.74, 'xt': 0.42, 'Fd': 0.99}
                },
                'Segmented ball': {
                    'Either': {'Fl': 0.60, 'xt': 0.30, 'Fd': 0.98}
                }
            }
        }

    @classmethod
    def get_instance(cls):
        """
        Get or create the ValveProperties instance (singleton pattern)
        """
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def get_properties(self, valve_type, trim_type, flow_direction):
        """
        Get valve properties based on valve type, trim type, and flow direction.

        Args:
            valve_type (str): Type of valve
            trim_type (str): Type of trim
            flow_direction (str): Direction of flow

        Returns:
            dict: Dictionary containing 'Fl', 'xt', and 'Fd' values

        Raises:
            KeyError: If the combination of parameters is not found in the data
        """
        try:
            # Special handling for tapered needle with formula
            if valve_type == 'Globe, small flow trim' and trim_type == 'Tapered needle' and flow_direction == 'Open':
                properties = self.valve_data[valve_type][trim_type][flow_direction].copy()
                return properties  # Note: Fd requires additional calculation with N₁₉(Cᵥ)⁰·⁵/Dᵥ

            return self.valve_data[valve_type][trim_type][flow_direction]
        except KeyError:
            raise KeyError(
                f"No data found for Valve Type: '{valve_type}', Trim Type: '{trim_type}', Flow Direction: '{flow_direction}'")

    def calculate_fd_for_tapered_needle(self, kv, dv, FL=0.95):
        """
        Calculate Fd for tapered needle using the formula N₁₉(Cᵥ)⁰·⁵/Dᵥ

        Args:
            kv (float): Flow coefficient
            dv (float): Valve diameter [mm]
            FL (float): Liquid pressure recovery factor

        Returns:
            float: Calculated Fd value
        """
        N19 = 2.5  # for KV
        return N19 * ((kv * FL) ** 0.5) / (dv*1000)

    def add_custom_valve(self, valve_type, trim_type, flow_direction, fl, xt, fd):
        """
        Add custom valve data to the dictionary

        Args:
            valve_type (str): Type of valve
            trim_type (str): Type of trim
            flow_direction (str): Direction of flow
            fl (float): Fl value
            xt (float): xt value
            fd (float): Fd value
        """
        # Create nested dictionaries if they don't exist
        if valve_type not in self.valve_data:
            self.valve_data[valve_type] = {}

        if trim_type not in self.valve_data[valve_type]:
            self.valve_data[valve_type][trim_type] = {}

        # Add or update the data
        self.valve_data[valve_type][trim_type][flow_direction] = {
            'Fl': fl,
            'xt': xt,
            'Fd': fd
        }

    @staticmethod
    def get_all_valve_types():
        """
        Get a list of all available valve types

        Returns:
            list: List of valve types
        """
        return list(ValveProperties.get_instance().valve_data.keys())

    @staticmethod
    def get_trim_types(valve_type):
        """
        Get a list of trim types for a specific valve type

        Args:
            valve_type (str): Type of valve

        Returns:
            list: List of trim types
        """
        if valve_type in ValveProperties.get_instance().valve_data:
            return list(ValveProperties.get_instance().valve_data[valve_type].keys())
        return []

    @staticmethod
    def get_flow_directions(valve_type, trim_type):
        """
        Get a list of flow directions for a specific valve and trim type

        Args:
            valve_type (str): Type of valve
            trim_type (str): Type of trim

        Returns:
            list: List of flow directions
        """
        if (valve_type in ValveProperties.get_instance().valve_data and
                trim_type in ValveProperties.get_instance().valve_data[valve_type]):
            return list(ValveProperties.get_instance().valve_data[valve_type][trim_type].keys())
        return []


class ValvePropertiesSelector:
    def __init__(self, valve_type, trim_type, flow_direction, pipe_size):
        """
        Initialize a valve selector with specific properties

        Args:
            valve_type (str): Type of valve
            trim_type (str): Type of trim
            flow_direction (str): Direction of flow
            pipe_size (float): Pipe size in mm
        """
        self.valve_type = valve_type
        self.trim_type = trim_type
        self.flow_direction = flow_direction
        self.pipe_size = pipe_size
        self.valve_props = ValveProperties.get_instance()

        # Get basic properties
        self.properties = self.valve_props.get_properties(valve_type, trim_type, flow_direction)

        # Calculate Fd for tapered needle
        if valve_type == 'Globe, small flow trim' and trim_type == 'Tapered needle' and flow_direction == 'Open':
            # Default kv value - in a real app, you would get this from user input
            self.kv = 10.0  # this should be provided by the user
            self.properties['Fd'] = self.valve_props.calculate_fd_for_tapered_needle(
                self.kv, self.pipe_size, self.properties['Fl']
            )

    def get_fl(self):
        """Get the Fl (liquid pressure recovery factor)"""
        return self.properties['Fl']

    def get_xt(self):
        """Get the xt (pressure differential ratio factor)"""
        return self.properties['xt']

    def get_fd(self):
        """Get the Fd (valve style modifier)"""
        return self.properties['Fd']

    def get_all_properties(self):
        """Get all valve properties as a dictionary"""
        return {
            'valve_type': self.valve_type,
            'trim_type': self.trim_type,
            'flow_direction': self.flow_direction,
            'pipe_size': self.pipe_size,
            'Fl': self.get_fl(),
            'xt': self.get_xt(),
            'Fd': self.get_fd()
        }


class Valve:
    """
    Class for handling valve sizing calculations following the ISA 75 standard.
    Implements the complete calculation scheme from the ISA flowchart for both
    incompressible and compressible fluids.
    """

    def __init__(self, props: ValvePropertiesSelector, fluid: Fluid):
        """
        Initialize the valve sizing calculator.

        Args:
            props: ValvePropertiesSelector object containing valve properties
            fluid: Fluid object containing fluid properties
        """
        self.fluid = fluid
        self.props = props
        self.valve_diameter = props.pipe_size  # In meters
        self.pipe_diameter = props.pipe_size  # In meters, assuming equal to valve size initially

        # Get valve properties
        self.Fd = props.get_fd()
        self.FL = props.get_fl()
        self.Xt = props.get_xt()

        # Initialize other parameters that will be calculated
        self.FP = None
        self.FR = None
        self.pc = None
        self.pv = None
        self.FF = None
        self.choking_delta_p = None
        self.Reynolds_number = None
        self.Cv = None
        self.Kv = None
        self.flow_regime = None
        self.is_choked = None

        # Constants for conversion
        self.N1 = 0.1  # For metric units (m³/h, kPa, kg/m³)
        self.N2 = 1.6*1e-3  # For metric units
        self.N4 = 7.07*1e-2  # For metric units (Cv to Kv conversion)
        self.N19 = 2.5  # For metric units (Kv calculation)
        self.N32 = 1.4e2  # For metric units (gas flow)
        self.rho0 = PropsSI('D', 'P', 101325, 'T', 288.15, 'water')  # Reference density for liquid

    def calculate_fp(self):
        """
        Calculate piping geometry factor (FP) using equation 31 from ISA 75.01.

        Returns:
            float: Piping geometry factor
        """
        d1 = self.valve_diameter
        D1 = self.pipe_diameter

        if d1 >= D1:
            return 1.0
        else:
            return 1.0 / (1.0 - (d1 / D1) ** 4) ** 0.5

    def calculate_FF(self):
        """
        Calculate FF, critical pressure and vapour pressure of the fluid.

        Returns:
            float: FF
            float: critical pressure
            float: vapour pressure
        """
        try:
            self.pc = self.fluid.get_critical_pressure()
            self.pv = self.fluid.get_vapour_pressure()
            self.FF = 0.96 - 0.28*(self.pv/self.pc)**0.5
        except:
            st.warning(f"Could not calculate FF for {self.fluid.name}. Using default value.")
            self.FF = 0.0
        return self.FF, self.pv, self.pc


    def calculate_reynolds_number(self, flow_rate):
        """
        Calculate Reynolds number using equation 28 from ISA 75.01.

        Args:
            flow_rate: Flow rate in m³/h

        Returns:
            float: Reynolds number
        """
        N2 = self.N2
        N4 = self.N4
        Fd = self.Fd
        density = self.fluid.get_density()
        viscosity = self.fluid.get_viscosity()

        # Reynolds number calculation (equation 28)
        return (N4 * Fd*  flow_rate)/(viscosity/density*(self.Kv * self.FL)**0.5)*((self.Kv**2*self.FL**2)/(N2*self.pipe_diameter**4)+1)**0.25

    def check_if_choked_liquid(self, delta_p, p1):
        """
        Check if flow is choked for liquid (incompressible fluid).

        Args:
            delta_p: Pressure differential (p1-p2) in Pa
            p1: Inlet pressure in Pa

        Returns:
            bool: True if flow is choked, False otherwise
        """
        FL = self.FL
        FP = self.FP if self.FP is not None else self.calculate_fp()

        # Calculate FF and get critical/vapor pressures
        self.calculate_FF()  # This will set self.FF, self.pv, and self.pc

        # Use the calculated values
        FF = self.FF if self.FF is not None else 0.0
        pv = self.pv if self.pv is not None else 0.0

        # Store FP value
        self.FP = FP

        self.choking_delta_p = FL**2 * (p1 - FF * pv)

        # Choked flow check: ΔP ≥ FL²·FP²·(P1 - FF·Pv)
        return delta_p >= self.choking_delta_p, self.choking_delta_p

    def check_if_choked_gas(self, x, Fk):
        """
        Check if flow is choked for gas (compressible fluid).

        Args:
            x: Pressure drop ratio (p1-p2)/p1
            Fk: Calculated Fk value

        Returns:
            bool: True if flow is choked, False otherwise
        """
        # Choked flow check: x ≥ Fk·Xt
        return x >= (Fk * self.Xt)

    def calculate_cv_liquid_non_choked(self, flow_rate, delta_p, p1):
        """
        Calculate Cv for non-choked liquid flow using equation 1 from ISA 75.01.

        Args:
            flow_rate: Flow rate in m³/h
            delta_p: Pressure differential (p1-p2) in Pa
            p1: Inlet pressure in Pa
            rho0: reference density of liquid in kg/m³

        Returns:
            float: Flow coefficient (Cv)
        """
        N1 = self.N1
        FL = self.FL
        FP = self.FP if self.FP is not None else self.calculate_fp()
        rho0 = self.rho0

        # Store FP value
        self.FP = FP

        # Convert pressure to kPa for calculation
        delta_p_kPa = delta_p / 1000.0

        # Non-choked liquid flow coefficient (equation 1)
        return flow_rate /N1 * ((self.fluid.get_density()/rho0)/(delta_p_kPa))**0.5

    def calculate_cv_liquid_choked(self, flow_rate, delta_p):
        """
        Calculate Cv for choked liquid flow using equation 3 from ISA 75.01.

        Args:
            flow_rate: Flow rate in m³/h
            p1: Inlet pressure in Pa
            gravity: Specific gravity of liquid

        Returns:
            float: Flow coefficient (Cv)
        """
        N1 = self.N1
        FL = self.FL
        FP = self.FP if self.FP is not None else self.calculate_fp()
        rho0 = self.rho0

        # Store FP value
        self.FP = FP

        # Convert pressure to kPa for calculation
        choking_delta_p_kPa =  self.choking_delta_p/1e3

        # Choked liquid flow coefficient (equation 3)
        return flow_rate /N1 * ((self.fluid.get_density()/rho0)/(choking_delta_p_kPa)) ** 0.5

    def calculate_cv_gas_non_choked(self, flow_rate, p1, p2, T1, Z=1.0, MW=29.0):
        """
        Calculate Cv for non-choked gas flow using equations from ISA 75.01.

        Args:
            flow_rate: Flow rate in m³/h at standard conditions
            p1: Inlet pressure in Pa
            p2: Outlet pressure in Pa
            T1: Inlet temperature in K
            Z: Compressibility factor
            MW: Molecular weight in kg/kmol

        Returns:
            float: Flow coefficient (Cv)
        """
        # Calculate pressure drop ratio x
        x = (p1 - p2) / p1

        # Calculate expansion factor Y
        Y = 1.0 - (x / (3.0 * self.Xt))

        # Calculate specific gravity of gas relative to air
        Gg = MW / 29.0

        # Convert pressures to kPa
        p1_kPa = p1 / 1000.0
        p2_kPa = p2 / 1000.0

        # Calculate Cv using equation 32 for non-choked gas flow
        return (flow_rate / (self.N32 * Y)) * ((T1 * Z * Gg) / (p1_kPa * (p1_kPa - p2_kPa))) ** 0.5

    def calculate_cv_gas_choked(self, flow_rate, p1, T1, Z=1.0, MW=29.0):
        """
        Calculate Cv for choked gas flow using equations from ISA 75.01.

        Args:
            flow_rate: Flow rate in m³/h at standard conditions
            p1: Inlet pressure in Pa
            T1: Inlet temperature in K
            Z: Compressibility factor
            MW: Molecular weight in kg/kmol

        Returns:
            float: Flow coefficient (Cv)
        """
        # Calculate specific gravity of gas relative to air
        Gg = MW / 29.0

        # Calculate Fk (similar to FL but for gases)
        Fk = self.FL * 0.9  # Approximation, typically Fk ≈ 0.9*FL

        # Convert pressure to kPa
        p1_kPa = p1 / 1000.0

        # Calculate Cv for choked gas flow (equations 12-14 or 15-17 depending on x)
        return flow_rate / (self.N32 * p1_kPa * Fk) * ((T1 * Z * Gg) / p1_kPa) ** 0.5

    def calculate_fr(self, ReV):
        """
        Calculate Reynolds number factor (FR) based on valve Reynolds number.

        Args:
            ReV: Valve Reynolds number

        Returns:
            float: Reynolds number factor
        """
        if ReV >= 10000:
            return 1.0
        elif ReV <= 10:
            return 0.1  # Approximation for very low Reynolds numbers
        else:
            # Approximate FR based on typical FR vs. ReV curve (would use Figure 3a/3b in full implementation)
            return 0.1 + 0.9 * (ReV / 10000.0) ** 0.25

    def size_valve_incompressible(self, flow_rate, inlet_pressure, outlet_pressure):
        """
        Size valve for incompressible fluid following ISA 75.01 flowchart.

        Args:
            flow_rate: Flow rate in m³/h
            inlet_pressure: Inlet pressure in Pa
            outlet_pressure: Outlet pressure in Pa

        Returns:
            dict: Dictionary containing calculated parameters
        """
        # Calculate pressure differential
        delta_p = inlet_pressure - outlet_pressure

        # Calculate FP (piping geometry factor)
        self.FP = self.calculate_fp()

        # Check if flow is choked
        self.is_choked, choking_delta_p = self.check_if_choked_liquid(delta_p, inlet_pressure)

        # Calculate initial Cv value
        if self.is_choked:
            self.flow_regime = "Choked flow"
            self.Kv = self.calculate_cv_liquid_choked(flow_rate, inlet_pressure)
        else:
            self.flow_regime = "Non-choked flow"
            self.Kv = self.calculate_cv_liquid_non_choked(flow_rate, delta_p, inlet_pressure)

        self.Cv = self.Kv * 1.156  # Conversion factor for metric units

        # Calculate Reynolds number
        self.Reynolds_number = self.calculate_reynolds_number(flow_rate)

        # Iterative process for Reynolds number correction
        if self.Reynolds_number < 10000:
            # For non-turbulent flow
            if self.valve_diameter < 2 * self.pipe_diameter:
                # Initial C' approximation
                C_prime = 1.3 * self.Cv

                # Calculate ReV using C'
                ReV_prime = self.calculate_reynolds_number(flow_rate)

                # Check condition for FR calculation method
                if C_prime ** 2 > 0.016 * self.N19 * self.valve_diameter:
                    # Calculate FR using Figure 3a/3b or equations
                    self.FR = self.calculate_fr(ReV_prime)
                else:
                    # Use FR from F.4 Annex
                    self.FR = 0.026 / (C_prime / (self.N19 * self.valve_diameter))

                # Check if correction needed
                if self.Cv * self.FR < self.Cv:
                    # Increase C by 30%
                    self.Cv *= 1.3
            else:
                # Use calculated C as flow coefficient
                pass

        # Convert Cv to Kv
        self.valve_diameter = (4*(self.Kv/36000)/np.pi)**0.5  # Kv/36000 is Area in m^2
        self.A = np.pi*(self.valve_diameter/2)**2

        return {
            "Cv": self.Cv,
            "Kv": self.Kv,
            "Area": self.A,
            "flow_regime": self.flow_regime,
            "is_choked": self.is_choked,
            "Reynolds_number": self.Reynolds_number,
            "FP": self.FP,
            "FR": self.FR
        }

    def size_valve_compressible(self, flow_rate, inlet_pressure, outlet_pressure, inlet_temperature):
        """
        Size valve for compressible fluid following ISA 75.01 flowchart.

        Args:
            flow_rate: Flow rate in m³/h at standard conditions
            inlet_pressure: Inlet pressure in Pa
            outlet_pressure: Outlet pressure in Pa
            inlet_temperature: Inlet temperature in K

        Returns:
            dict: Dictionary containing calculated parameters
        """
        # Get fluid properties
        z_factor = self.fluid.get_z_factor()
        molecular_weight = self.fluid.get_molecular_weight()

        # Calculate pressure ratio
        x = (inlet_pressure - outlet_pressure) / inlet_pressure

        # Calculate Fk (similar to FL but for gases)
        Fk = self.Xt ** 0.5  # Approximation per equation 34

        # Check if flow is choked
        self.is_choked = self.check_if_choked_gas(x, Fk)

        # Calculate initial Cv value
        if self.is_choked:
            self.flow_regime = "Choked flow"
            # Critical flow factor Y = 0.667 for choked flow
            Y = 0.667
            self.Cv = self.calculate_cv_gas_choked(flow_rate, inlet_pressure, inlet_temperature, z_factor,
                                                   molecular_weight)
        else:
            self.flow_regime = "Non-choked flow"
            self.Cv = self.calculate_cv_gas_non_choked(flow_rate, inlet_pressure, outlet_pressure, inlet_temperature,
                                                       z_factor, molecular_weight)

        # Calculate Reynolds number
        self.Reynolds_number = self.calculate_reynolds_number(flow_rate)

        # Iterative process for Reynolds number correction
        if self.Reynolds_number < 10000:
            # For non-turbulent flow
            if self.valve_diameter < 2 * self.pipe_diameter:
                # Initial C' approximation
                C_prime = 1.3 * self.Cv

                # Calculate ReV using C'
                ReV_prime = self.calculate_reynolds_number(flow_rate)

                # Check condition for FR calculation method
                if C_prime ** 2 > 0.016 * self.N19 * self.valve_diameter:
                    # Calculate FR using Figure 3a/3b or equations
                    self.FR = self.calculate_fr(ReV_prime)
                else:
                    # Use FR from F.4 Annex
                    self.FR = 0.026 / (C_prime / (self.N19 * self.valve_diameter))

                # Check if correction needed
                if self.Cv * self.FR < self.Cv:
                    # Increase C by 30%
                    self.Cv *= 1.3
            else:
                # Use calculated C as flow coefficient
                pass

        # Convert Cv to Kv
        self.Kv = self.Cv / 1.156
        self.valve_diameter = (4*(self.Kv/36000)/np.pi)**0.5  # Kv/36000 is Area in m^2
        self.A = np.pi*(self.valve_diameter/2)**2

        return {
            "Cv": self.Cv,
            "Kv": self.Kv,
            "Area": self.A,
            "flow_regime": self.flow_regime,
            "is_choked": self.is_choked,
            "choking_delta_p": self.choking_delta_p,
            "Reynolds_number": self.Reynolds_number,
            "x": x,
            "Fk": Fk,
            "FR": self.FR
        }

    def size_valve(self, flow_rate, inlet_pressure, outlet_pressure):
        """
        Size valve based on fluid type (incompressible or compressible).

        Args:
            flow_rate: Flow rate in m³/h (standard conditions for gas)
            inlet_pressure: Inlet pressure in Pa
            outlet_pressure: Outlet pressure in Pa

        Returns:
            dict: Dictionary containing calculated parameters
        """
        if self.fluid.is_compressible:
            return self.size_valve_compressible(flow_rate, inlet_pressure, outlet_pressure, self.fluid.T)
        else:
            return self.size_valve_incompressible(flow_rate, inlet_pressure, outlet_pressure)

    def calculate_max_flow_rate(self, cv, inlet_pressure, outlet_pressure):
        """
        Calculate maximum flow rate for a given Cv and pressure conditions.

        Args:
            cv: Flow coefficient (Cv)
            inlet_pressure: Inlet pressure in Pa
            outlet_pressure: Outlet pressure in Pa

        Returns:
            float: Maximum flow rate in m³/h
        """
        delta_p = inlet_pressure - outlet_pressure

        if self.fluid.is_compressible:
            # For compressible fluids
            x = delta_p / inlet_pressure
            z_factor = self.fluid.get_z_factor()
            molecular_weight = self.fluid.get_molecular_weight()

            # Calculate Fk (similar to FL but for gases)
            Fk = self.Xt ** 0.5

            # Check if flow is choked
            is_choked = x >= (Fk * self.Xt)

            # Convert pressure to kPa
            inlet_pressure_kPa = inlet_pressure / 1000.0
            outlet_pressure_kPa = outlet_pressure / 1000.0

            if is_choked:
                # Critical flow factor Y = 0.667 for choked flow
                Y = 0.667

                # Calculate specific gravity of gas relative to air
                Gg = molecular_weight / 29.0

                # Maximum flow rate for choked flow
                return cv * self.N32 * inlet_pressure_kPa * Fk * (
                            inlet_pressure_kPa / (self.fluid.T * z_factor * Gg)) ** 0.5
            else:
                # Calculate expansion factor Y
                Y = 1.0 - (x / (3.0 * self.Xt))

                # Calculate specific gravity of gas relative to air
                Gg = molecular_weight / 29.0

                # Maximum flow rate for non-choked flow
                return cv * self.N32 * Y * ((inlet_pressure_kPa * (inlet_pressure_kPa - outlet_pressure_kPa)) / (
                            self.fluid.T * z_factor * Gg)) ** 0.5
        else:
            # For incompressible fluids
            FP = self.FP if self.FP is not None else self.calculate_fp()

            # Check if flow is choked
            is_choked = delta_p >= self.choking_delta_p

            if is_choked:
                # Maximum flow rate for choked flow
                return 3600*self.A*(self.choking_delta_p/self.fluid.get_density())** 0.5
            else:
                # Maximum flow rate for non-choked flow
                return 3600*self.A*(delta_p/self.fluid.get_density())** 0.5
