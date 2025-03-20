import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from CoolProp.CoolProp import PropsSI

st.set_page_config(page_title="Control Valve Sizing Tool", layout="wide")

def main():
    st.title("Control Valve Sizing Tool (ISA Standard)")
    st.markdown("*Based on ISA Standard for Industrial-Process Control Valves with CoolProp fluid properties*")

    # Sidebar with calculation type selector
    st.sidebar.header("Calculation Settings")
    calculation_type = st.sidebar.radio(
        "Select Flow Type:",
        ["Incompressible Flow (Liquid)", "Compressible Flow (Gas/Vapor)"]
    )

    # Constants from Table 1 - Numerical constants N
    constants = {
        "N1": 0.865,  # For Cv, Q in m³/h, P in bar
        "N2": 890.0,  # For Cv, W in kg/h, P in bar
        "N4": 76.0,  # For Cv, Q in m³/h, P in kPa
        "N6": 27.3,  # For Cv, Q in m³/h, P in psi
        "N7": 63.3,  # For Cv, W in lb/h, P in psi
        "N8": 28.0,  # For Cv, W in kg/h, P in kPa
        "N9": 0.00214  # For Reynolds number calculation
    }

    # Main content
    with st.container(border=True):
        st.header('Input Parameters')
        col1, col2, col3 = st.columns([1, 1, 1], border=True)

        with col1:
            st.subheader("Process Parameters")
            units_system = st.radio("Units System", ["Metric", "US/Imperial"])

            # Fluid selection section
            st.subheader("Fluid Properties")

            # CoolProp fluid list
            coolprop_fluids = [
            "Water", "Air", "Nitrogen", "Oxygen", "Hydrogen", "CarbonDioxide", "Methane",
            "Ethane", "Propane", "Butane", "Ammonia", "R134a", "R22", "R410A", "Helium",
            "Argon", "Steam", "Ethanol", "Methanol", "Benzene", "Toluene", "Acetone"
            ]

            # Common fluids by type for easier selection
            if calculation_type == "Incompressible Flow (Liquid)":
                suggested_fluids = ["Water", "Ethanol", "Methanol", "Ammonia", "Benzene", "Toluene", "Acetone"]
            else:  # Compressible flow
                suggested_fluids = ["Air", "Nitrogen", "Oxygen", "Hydrogen", "CarbonDioxide", "Methane", "Propane", "Steam"]

            # Allow direct selection or custom fluid
            fluid_selection_method = st.radio("Fluid Selection Method", ["Select Common Fluid", "Enter Custom Properties"])

        with col2:
            if fluid_selection_method == "Select Common Fluid":
                selected_fluid = st.selectbox("Select Fluid",
                                          suggested_fluids + [f for f in coolprop_fluids if f not in suggested_fluids])

                if calculation_type == "Incompressible Flow (Liquid)":
                    # Liquid parameters in selected units
                    if units_system == "Metric":
                        fluid_temp = st.number_input("Fluid Temperature [°C]", min_value = -273.15, value=90.0)
                        fluid_temp_k = fluid_temp + 273.15  # Convert to Kelvin
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [bar]", min_value=0.1, value=6.8, step=0.1)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [bar]", min_value=0.1, value=2.2, step=0.1)
                        inlet_pressure_pa = inlet_pressure * 1e5  # Convert to Pa
                        pressure_drop = inlet_pressure - outlet_pressure
                        flow_rate = st.number_input("Flow Rate (Q) [m³/h]", min_value=0.1, value=360.0, step=1.0)
                    else:  # US/Imperial
                        fluid_temp = st.number_input("Fluid Temperature [°F]", value=70.0)
                        fluid_temp_c = (fluid_temp - 32) * 5 / 9
                        fluid_temp_k = fluid_temp_c + 273.15  # Convert to Kelvin
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [psi]", min_value=0.1, value=75.0, step=1.0)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [psi]", min_value=0.1, value=30.0, step=1.0)
                        pressure_drop = inlet_pressure - outlet_pressure
                        inlet_pressure_pa = inlet_pressure * 6894.76  # Convert to Pa
                        flow_rate = st.number_input("Flow Rate (Q) [gpm]", min_value=0.1, value=50.0, step=1.0)
                        # Convert to m³/h for calculations
                        flow_rate_m3h = flow_rate * 0.2271  # gpm to m³/h

                    # Calculate fluid properties using CoolProp
                    try:
                        # Density in kg/m³
                        density = PropsSI('D', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid)
                        # Viscosity in Pa·s
                        viscosity = PropsSI('V', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid)
                        # Convert to cP
                        viscosity_cp = viscosity * 1000
                        # Specific gravity (relative to water at 4°C)
                        water_density = PropsSI('D', 'T', 277.15, 'P', 101325, 'Water')
                        specific_gravity = density / water_density

                        st.success(f"CoolProp Properties for {selected_fluid}:")
                        st.info(f"Density: {density:.2f} kg/m³")
                        st.info(f"Viscosity: {viscosity_cp:.2f} cP")
                        st.info(f"Specific Gravity: {specific_gravity:.4f}")

                        liquid_sg = specific_gravity
                        fluid_viscosity = viscosity_cp
                    except Exception as e:
                        st.error(f"Error calculating properties: {str(e)}")
                        st.warning("Using default values instead.")
                        liquid_sg = 1.0
                        fluid_viscosity = 1.0

                else:  # Compressible Flow (Gas/Vapor)
                    if units_system == "Metric":
                        fluid_temp = st.number_input("Gas Temperature [°C]", value=25.0)
                        fluid_temp_k = fluid_temp + 273.15  # Convert to Kelvin
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [bar]", min_value=0.1, value=10.0, step=0.1)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [bar]", min_value=0.1, value=5.0, step=0.1)
                        inlet_pressure_pa = inlet_pressure * 1e5  # Convert to Pa
                        pressure_drop = inlet_pressure - outlet_pressure


                        # Choose between mass flow and volumetric flow
                        flow_type = st.radio("Flow Input Type", ["Volumetric Flow", "Mass Flow"])
                        if flow_type == "Volumetric Flow":
                            flow_rate = st.number_input("Flow Rate (Q) [m³/h at standard conditions]", min_value=0.1,
                                                    value=50.0, step=1.0)
                            mass_flow = 0.0  # Initialize as zero
                        else:
                            mass_flow = st.number_input("Mass Flow Rate (W) [kg/h]", min_value=0.1, value=100.0, step=1.0)
                            flow_rate = 0.0  # Initialize as zero
                    else:  # US/Imperial
                        fluid_temp = st.number_input("Gas Temperature [°F]", value=75.0)
                        fluid_temp_c = (fluid_temp - 32) * 5 / 9
                        fluid_temp_k = fluid_temp_c + 273.15  # Convert to Kelvin
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [psi]", min_value=0.1, value=150.0, step=1.0)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [psi]", min_value=0.1, value=75.0, step=1.0)
                        inlet_pressure_pa = inlet_pressure * 6894.76  # Convert to Pa

                        # Choose between mass flow and volumetric flow
                        flow_type = st.radio("Flow Input Type", ["Volumetric Flow", "Mass Flow"])
                        if flow_type == "Volumetric Flow":
                            flow_rate = st.number_input("Flow Rate (Q) [scfh]", min_value=0.1, value=1000.0, step=10.0)
                            # Convert to m³/h for calculations
                            flow_rate_m3h = flow_rate * 0.02832  # scfh to m³/h
                            mass_flow = 0.0  # Initialize as zero
                        else:
                            mass_flow = st.number_input("Mass Flow Rate (W) [lb/h]", min_value=0.1, value=200.0, step=1.0)
                            # Convert to kg/h for calculations
                            mass_flow_kg = mass_flow * 0.4536  # lb/h to kg/h
                            flow_rate = 0.0  # Initialize as zero

                    # Calculate gas properties using CoolProp
                    try:
                        # Molar mass in g/mol
                        molecular_weight = PropsSI('M', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid) * 1000
                        # Compressibility factor
                        z_factor = PropsSI('Z', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid)
                        # Specific heat ratio (Cp/Cv)
                        cp = PropsSI('C', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid)
                        cv = PropsSI('O', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid)
                        specific_heat_ratio = cp / cv
                        # Viscosity in Pa·s
                        viscosity = PropsSI('V', 'T', fluid_temp_k, 'P', inlet_pressure_pa, selected_fluid)
                        # Convert to cP
                        viscosity_cp = viscosity * 1000

                        st.success(f"CoolProp Properties for {selected_fluid}:")
                        st.info(f"Molecular Weight: {molecular_weight:.2f} g/mol")
                        st.info(f"Compressibility Factor (Z): {z_factor:.4f}")
                        st.info(f"Specific Heat Ratio (k = Cp/Cv): {specific_heat_ratio:.4f}")
                        st.info(f"Viscosity: {viscosity_cp:.4f} cP")
                    except Exception as e:
                        st.error(f"Error calculating properties: {str(e)}")
                        st.warning("Using default values instead.")
                        molecular_weight = 28.0  # Default for air
                        z_factor = 1.0
                        specific_heat_ratio = 1.4
                        viscosity_cp = 0.018

            else:  # Enter Custom Properties
                if calculation_type == "Incompressible Flow (Liquid)":
                    if units_system == "Metric":
                        fluid_temp = st.number_input("Fluid Temperature [°C]", value=20.0)
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [bar]", min_value=0.0, value=5.0, step=0.1)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [bar]", min_value=0.0, value=2.0, step=0.1)
                        pressure_drop = inlet_pressure - outlet_pressure
                        st.info(f"Pressure Drop (ΔP): {pressure_drop:.2f} bar")
                        flow_rate = st.number_input("Flow Rate (Q) [m³/h]", min_value=0.0, value=10.0, step=1.0)

                        # Manual property inputs
                        liquid_sg = st.number_input("Liquid Specific Gravity (relative to water)", min_value=0.1, value=1.0,
                                                    step=0.1)
                        fluid_viscosity = st.number_input("Fluid Viscosity [cP]", min_value=0.1, value=1.0, step=0.1)

                    else:  # US/Imperial
                        fluid_temp = st.number_input("Fluid Temperature [°F]", value=70.0)
                        # Convert to °C for calculations
                        fluid_temp_c = (fluid_temp - 32) * 5 / 9
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [psi]", min_value=0.0, value=75.0, step=1.0)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [psi]", min_value=0.0, value=30.0, step=1.0)
                        pressure_drop = inlet_pressure - outlet_pressure
                        st.info(f"Pressure Drop (ΔP): {pressure_drop:.2f} psi")
                        flow_rate = st.number_input("Flow Rate (Q) [gpm]", min_value=0.0, value=50.0, step=1.0)

                        # Manual property inputs
                        liquid_sg = st.number_input("Liquid Specific Gravity (relative to water)", min_value=0.1, value=1.0,
                                                    step=0.1)
                        fluid_viscosity = st.number_input("Fluid Viscosity [cP]", min_value=0.1, value=1.0, step=0.1)

                else:  # Compressible Flow (Gas/Vapor)
                    if units_system == "Metric":
                        fluid_temp = st.number_input("Gas Temperature [°C]", value=25.0)
                        fluid_temp_k = fluid_temp + 273.15  # Convert to Kelvin
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [bar]", min_value=0.0, value=10.0, step=0.1)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [bar]", min_value=0.0, value=5.0, step=0.1)
                        pressure_drop = inlet_pressure - outlet_pressure
                        pressure_ratio = outlet_pressure / inlet_pressure
                        st.info(f"Pressure Drop (ΔP): {pressure_drop:.2f} bar")
                        st.info(f"Pressure Ratio (P₂/P₁): {pressure_ratio:.3f}")

                        # Choose between mass flow and volumetric flow
                        flow_type = st.radio("Flow Input Type", ["Volumetric Flow", "Mass Flow"])
                        if flow_type == "Volumetric Flow":
                            flow_rate = st.number_input("Flow Rate (Q) [m³/h at standard conditions]", min_value=0.0,
                                                        value=50.0, step=1.0)
                            mass_flow = 0.0  # Initialize as zero
                        else:
                            mass_flow = st.number_input("Mass Flow Rate (W) [kg/h]", min_value=0.0, value=100.0, step=1.0)
                            flow_rate = 0.0  # Initialize as zero

                        # Manual property inputs
                        molecular_weight = st.number_input("Molecular Weight [g/mol]", min_value=1.0, value=28.0, step=0.1)
                        z_factor = st.number_input("Compressibility Factor (Z)", min_value=0.1, max_value=1.0, value=1.0,
                                                   step=0.01)
                        specific_heat_ratio = st.number_input("Specific Heat Ratio (k = Cp/Cv)", min_value=1.0, value=1.4,
                                                              step=0.01)
                        viscosity_cp = st.number_input("Viscosity [cP]", min_value=0.001, value=0.018, step=0.001,
                                                       format="%.4f")

                    else:  # US/Imperial
                        fluid_temp = st.number_input("Gas Temperature [°F]", value=75.0)
                        # Convert to Kelvin for calculations
                        fluid_temp_c = (fluid_temp - 32) * 5 / 9
                        fluid_temp_k = fluid_temp_c + 273.15
                        inlet_pressure = st.number_input("Inlet Pressure (P₁) [psi]", min_value=0.0, value=150.0, step=1.0)
                        outlet_pressure = st.number_input("Outlet Pressure (P₂) [psi]", min_value=0.0, value=75.0, step=1.0)
                        pressure_drop = inlet_pressure - outlet_pressure
                        pressure_ratio = outlet_pressure / inlet_pressure
                        st.info(f"Pressure Drop (ΔP): {pressure_drop:.2f} psi")
                        st.info(f"Pressure Ratio (P₂/P₁): {pressure_ratio:.3f}")

                        # Choose between mass flow and volumetric flow
                        flow_type = st.radio("Flow Input Type", ["Volumetric Flow", "Mass Flow"])
                        if flow_type == "Volumetric Flow":
                            flow_rate = st.number_input("Flow Rate (Q) [scfh]", min_value=0.0, value=1000.0, step=10.0)
                            # Initialize as zero
                            mass_flow = 0.0
                        else:
                            mass_flow = st.number_input("Mass Flow Rate (W) [lb/h]", min_value=0.0, value=200.0, step=1.0)
                            # Initialize as zero
                            flow_rate = 0.0

                        # Manual property inputs
                        molecular_weight = st.number_input("Molecular Weight [g/mol]", min_value=1.0, value=28.0, step=0.1)
                        z_factor = st.number_input("Compressibility Factor (Z)", min_value=0.1, max_value=1.0, value=1.0,
                                                   step=0.01)
                        specific_heat_ratio = st.number_input("Specific Heat Ratio (k = Cp/Cv)", min_value=1.0, value=1.4,
                                                              step=0.01)
                        viscosity_cp = st.number_input("Viscosity [cP]", min_value=0.001, value=0.018, step=0.001,
                                                       format="%.4f")
        with col3:
            # Pipe and valve parameters section
            st.subheader("Pipe and Valve Parameters")

            if units_system == "Metric":
                pipe_diameter = st.number_input("Pipe Diameter (D) [mm]", min_value=1.0, value=50.0, step=1.0)
                pipe_diameter_m = pipe_diameter / 1000  # Convert to meters for calculations
            else:  # US/Imperial
                pipe_diameter = st.number_input("Pipe Diameter (D) [inches]", min_value=0.1, value=2.0, step=0.1)
                # Convert to mm for calculations
                pipe_diameter_mm = pipe_diameter * 25.4  # inches to mm
                pipe_diameter_m = pipe_diameter_mm / 1000  # mm to m

            # Valve parameters
            valve_type = st.selectbox(
                "Valve Type",
                ["Globe - Single Seated", "Globe - Double Seated", "Butterfly", "Ball - Segmented",
                 "Eccentric Plug", "Angle Valve"]
            )

            # Get Fd based on valve type (simplified from Table 2)
            fd_values = {
                "Globe - Single Seated": 0.9,
                "Globe - Double Seated": 0.85,
                "Butterfly": 0.7,
                "Ball - Segmented": 0.5,
                "Eccentric Plug": 0.6,
                "Angle Valve": 0.8
            }
            fd = fd_values[valve_type]

            if calculation_type == "Incompressible Flow (Liquid)":
                fl_values = {
                    "Globe - Single Seated": 0.85,
                    "Globe - Double Seated": 0.75,
                    "Butterfly": 0.52,
                    "Ball - Segmented": 0.45,
                    "Eccentric Plug": 0.57,
                    "Angle Valve": 0.8
                }
                fl = fl_values[valve_type]

                xt_values = {
                    "Globe - Single Seated": 0.75,
                    "Globe - Double Seated": 0.65,
                    "Butterfly": 0.50,
                    "Ball - Segmented": 0.30,
                    "Eccentric Plug": 0.45,
                    "Angle Valve": 0.72
                }
                xt = xt_values[valve_type]
            else:  # Compressible Flow (Gas/Vapor)
                xtp_values = {
                    "Globe - Single Seated": 0.75,
                    "Globe - Double Seated": 0.65,
                    "Butterfly": 0.50,
                    "Ball - Segmented": 0.30,
                    "Eccentric Plug": 0.45,
                    "Angle Valve": 0.72
                }
                xtp = xtp_values[valve_type]

            # Additional parameters
            if units_system == "Metric":
                valve_diameter = st.number_input("Valve Diameter (d) [mm]", min_value=1.0, value=pipe_diameter, step=1.0)
                valve_diameter_m = valve_diameter / 1000  # Convert to meters
            else:  # US/Imperial
                valve_diameter = st.number_input("Valve Diameter (d) [inches]", min_value=0.1, value=pipe_diameter,
                                                 step=0.1)
                # Convert to mm for calculations
                valve_diameter_mm = valve_diameter * 25.4  # inches to mm
                valve_diameter_m = valve_diameter_mm / 1000  # mm to m

            # Calculate d/D ratio
            d_D_ratio = valve_diameter / pipe_diameter
            st.info(f"d/D Ratio: {d_D_ratio:.3f}")

            # Fittings
            include_fittings = st.checkbox("Include Pipe Fittings")

            if include_fittings:
                st.warning("Pipe fittings functionality is not fully implemented yet.")

    with st.container():
        # CALCULATE BUTTON
        if st.button("Calculate Valve Coefficient (Cv)"):
            st.subheader("Calculation Results")
            if calculation_type == "Incompressible Flow (Liquid)":
                # Prepare variables for calculation based on units
                if units_system == "Metric":
                    Q = flow_rate
                    P1 = inlet_pressure
                    P2 = outlet_pressure
                    dP = pressure_drop
                    SG = liquid_sg
                    D = pipe_diameter_m
                    d = valve_diameter_m
                    mu = fluid_viscosity
                    N1 = constants["N1"]
                    N4 = constants["N4"]
                    N9 = constants["N9"]
                else:  # US/Imperial
                    Q = flow_rate_m3h
                    P1 = inlet_pressure
                    P2 = outlet_pressure
                    dP = pressure_drop
                    SG = liquid_sg
                    D = pipe_diameter_m
                    d = valve_diameter_m
                    mu = fluid_viscosity
                    N1 = constants["N1"]
                    N4 = constants["N4"]
                    N9 = constants["N9"]

                # Calculate Fp (piping geometry factor) based on d/D ratio
                # Simplified from Figure 2a
                Fp = max(1 - (d / D) * (1 - d / D), 0.5)

                # Calculate liquid pressure recovery factor FL
                FL = fl

                # Calculate pressure ratio factor Xt
                XT = xt

                # Calculate Y factor (expansion factor) - for liquids, Y=1
                Y = 1.0

                # Calculate FF (liquid critical pressure ratio factor)
                FF = 0.96 - 0.28 * math.sqrt(XT / FL)

                # Calculate x (pressure differential ratio)
                x = dP / (P1 * FL ** 2)

                # Calculate choked pressure drop
                dP_choked = FL ** 2 * P1

                # Check if flow is choked
                is_choked = (dP >= dP_choked)

                if is_choked:
                    # Use choked flow equation
                    Cv = Q / (N1 * FL * Fp * Y * math.sqrt(dP_choked * SG))
                else:
                    # Use non-choked flow equation
                    Cv = Q / (N1 * Fp * Y * math.sqrt(dP * SG))

                # Calculate Reynolds number factor FR
                # Simplified calculation for FR based on Reynolds number
                Rev = (N9 * Q * SG) / (mu * math.sqrt(Cv * d ** 2))

                # Simplified FR calculation based on Figure 3
                if Rev >= 10000:
                    FR = 1.0
                elif Rev <= 10:
                    FR = 0.1
                else:
                    FR = 0.1 + 0.9 * (math.log10(Rev / 10) / 3)

                # Apply Reynolds number correction to Cv
                Cv_laminar = Cv / FR

                # Display results
                st.success(f"Standard Valve Coefficient (Cv): {Cv:.2f}")

                if is_choked:
                    st.warning("⚠️ Flow is choked (at critical pressure drop)")
                    st.info(f"Critical Pressure Drop: {dP_choked:.2f} bar")

                if Rev < 10000:
                    st.warning(f"⚠️ Flow is in laminar or transitional regime (Reynolds number = {Rev:.0f})")
                    st.info(f"Reynolds Number Factor (FR): {FR:.4f}")
                    st.info(f"Corrected Cv for laminar flow: {Cv_laminar:.2f}")

                # Create a dataframe for results
                results_df = pd.DataFrame({
                    "Parameter": ["Valve Coefficient (Cv)", "Piping Geometry Factor (Fp)",
                                  "Liquid Pressure Recovery Factor (FL)", "Pressure Ratio Factor (Xt)",
                                  "Reynolds Number", "Reynolds Number Factor (FR)",
                                  "Critical Pressure Drop", "Is Flow Choked?"],
                    "Value": [f"{Cv:.2f}", f"{Fp:.3f}", f"{FL:.3f}", f"{XT:.3f}",
                              f"{Rev:.0f}", f"{FR:.4f}", f"{dP_choked:.2f} bar",
                              "Yes" if is_choked else "No"]
                })

                st.table(results_df)

                # Plot Cv vs. Flow Rate
                flow_range = np.linspace(0.1 * flow_rate, 2 * flow_rate, 50)
                cv_values = []

                for q in flow_range:
                    if is_choked:
                        cv = q / (N1 * FL * Fp * Y * math.sqrt(dP_choked * SG))
                    else:
                        cv = q / (N1 * Fp * Y * math.sqrt(dP * SG))
                    cv_values.append(cv)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(flow_range, cv_values)
                ax.set_xlabel("Flow Rate (m³/h)")
                ax.set_ylabel("Valve Coefficient (Cv)")
                ax.set_title("Cv vs. Flow Rate")
                ax.grid(True)
                ax.axhline(y=Cv, color='r', linestyle='-')
                ax.axvline(x=flow_rate, color='r', linestyle='-')
                st.pyplot(fig)

            else:  # Compressible Flow (Gas/Vapor)
                # Prepare variables for calculation based on units
                if units_system == "Metric":
                    Q = flow_rate
                    W = mass_flow
                    P1 = inlet_pressure
                    P2 = outlet_pressure
                    T = fluid_temp_k
                    M = molecular_weight
                    Z = z_factor
                    k = specific_heat_ratio
                    D = pipe_diameter_m
                    d = valve_diameter_m
                    N2 = constants["N2"]
                    N4 = constants["N4"]
                    N8 = constants["N8"]
                    N9 = constants["N9"]
                else:  # US/Imperial
                    Q = flow_rate_m3h
                    W = mass_flow_kg
                    P1 = inlet_pressure
                    P2 = outlet_pressure
                    T = fluid_temp_k
                    M = molecular_weight
                    Z = z_factor
                    k = specific_heat_ratio
                    D = pipe_diameter_m
                    d = valve_diameter_m
                    N2 = constants["N2"]
                    N4 = constants["N4"]
                    N8 = constants["N8"]
                    N9 = constants["N9"]

                # Calculate Fp (piping geometry factor) based on d/D ratio
                # Simplified from Figure 2a
                Fp = max(1 - (d / D) * (1 - d / D), 0.5)

                # Calculate pressure ratio x
                x = (P1 - P2) / P1

                # Calculate the expansion factor Y for compressible flow
                Fk = k / 1.4  # Specific heat ratio factor
                xTP = xtp  # Pressure differential ratio factor at choked flow

                # Calculate Y based on the ISA formula
                if x < xTP:
                    Y = 1 - (x / (3 * xTP)) * (1 - (P2 / P1))
                else:
                    Y = 2 / 3

                # Check if using mass flow or volumetric flow
                if W > 0:
                    # Use mass flow formula
                    Cv = W / (N8 * Fp * Y * math.sqrt(P1 * M / (Z * T)))
                else:
                    # Use volumetric flow formula
                    Cv = Q / (N4 * Fp * Y * P1 * math.sqrt(1 / (M * T * Z)))

                # Calculate Reynolds number factor FR
                # Simplified calculation for FR based on Reynolds number
                if W > 0:
                    Rev = (N9 * W) / (mu * math.sqrt(Cv * d ** 2 * M))
                else:
                    # Estimate density for Reynolds calculation
                    rho = (P1 * M) / (0.08314 * T)  # kg/m³
                    Rev = (N9 * Q * rho) / (mu * math.sqrt(Cv * d ** 2))

                # Simplified FR calculation based on Figure 3
                if Rev >= 10000:
                    FR = 1.0
                elif Rev <= 10:
                    FR = 0.1
                else:
                    FR = 0.1 + 0.9 * (math.log10(Rev / 10) / 3)

                # Apply Reynolds number correction to Cv
                Cv_laminar = Cv / FR

                # Display results
                st.success(f"Standard Valve Coefficient (Cv): {Cv:.2f}")

                if x >= xTP:
                    st.warning("⚠️ Flow is choked (at critical pressure ratio)")
                    choked_dp = P1 * xTP
                    st.info(f"Critical Pressure Drop: {choked_dp:.2f} bar")

                if Rev < 10000:
                    st.warning(f"⚠️ Flow is in laminar or transitional regime (Reynolds number = {Rev:.0f})")
                    st.info(f"Reynolds Number Factor (FR): {FR:.4f}")
                    st.info(f"Corrected Cv for laminar flow: {Cv_laminar:.2f}")

                # Create a dataframe for results
                results_df = pd.DataFrame({
                    "Parameter": ["Valve Coefficient (Cv)", "Piping Geometry Factor (Fp)",
                                  "Expansion Factor (Y)", "Pressure Differential Ratio (x)",
                                  "Critical Pressure Differential Ratio (xTP)",
                                  "Reynolds Number", "Reynolds Number Factor (FR)",
                                  "Is Flow Choked?"],
                    "Value": [f"{Cv:.2f}", f"{Fp:.3f}", f"{Y:.3f}", f"{x:.3f}",
                              f"{xTP:.3f}", f"{Rev:.0f}", f"{FR:.4f}",
                              "Yes" if x >= xTP else "No"]
                })

                st.table(results_df)

                # Plot for compressible flow - Cv vs. Pressure Ratio
                pressure_ratios = np.linspace(0.1, 1.0, 50)
                cv_values = []

                for ratio in pressure_ratios:
                    p2 = P1 * ratio
                    x_test = (P1 - p2) / P1

                    # Calculate Y based on ratio
                    if x_test < xTP:
                        y_val = 1 - (x_test / (3 * xTP)) * (1 - ratio)
                    else:
                        y_val = 2 / 3

                    if W > 0:
                        # Use mass flow formula
                        cv = W / (N8 * Fp * y_val * math.sqrt(P1 * M / (Z * T)))
                    else:
                        # Use volumetric flow formula
                        cv = Q / (N4 * Fp * y_val * P1 * math.sqrt(1 / (M * T * Z)))

                    cv_values.append(cv)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(pressure_ratios, cv_values)
                ax.set_xlabel("Pressure Ratio (P₂/P₁)")
                ax.set_ylabel("Valve Coefficient (Cv)")
                ax.set_title("Cv vs. Pressure Ratio")
                ax.grid(True)
                ax.axhline(y=Cv, color='r', linestyle='-')
                ax.axvline(x=P2 / P1, color='r', linestyle='-')
                st.pyplot(fig)

    # Documentation section
#    with st.expander:
#        st.write('tbd')

if __name__ == '__main__':
    main()