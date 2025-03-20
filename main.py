import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from classes import Fluid, ValveProperties, ValvePropertiesSelector, Valve

st.set_page_config(
    page_title="Control Valve Sizing",
    layout="wide",
    page_icon="🔧"
)

import streamlit as st
import numpy as np

if not 'Valve' in st.session_state:
    st.session_state['Valve'] = None
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 0

st.title("Control Valve Sizing Tool")
st.markdown("*Based on ISA Standard 75.01.01-2007 for Industrial-Process Control Valves with CoolProp calculated fluid properties*")

# Create tabs for input and results
if st.session_state['active_tab'] == 0:
    tab1, tab2 = st.tabs(["Input Parameters", "Results & Flow Curve"])
else:
    tab2, tab1 = st.tabs(["Results & Flow Curve", "Input Parameters"])


with tab1:
    cols = st.columns([1, 1 ,1])

    with cols[0]:
        st.subheader('Fluid and Inlet Conditions')
        coolprop_fluids = [
            "Water", "Air", "Nitrogen", "Oxygen", "Hydrogen", "CarbonDioxide", "Methane",
            "Ethane", "Propane", "Butane", "Ammonia", "R134a", "R22", "R410A", "Helium",
            "Argon", "Steam", "Ethanol", "Methanol", "Benzene", "Toluene", "Acetone"
        ]
        fluid_name = st.selectbox('Select Fluid', coolprop_fluids)
        fluid_temperature = 273.15 + st.number_input('Inlet Temperature [°C]',
                                                     min_value=-30.0,
                                                     value=90.0)
        fluid_pressure = 1e5 * (st.number_input('Inlet Pressure [bar abs]',
                                                    min_value=1.0,
                                                    value=6.8))

    with cols[1]:
        st.subheader('Flow Outlet Conditions')
        flow_rate = st.number_input('Flow Rate [m³/h]',
                                    min_value=0.1,
                                    max_value=10000.0,
                                    value=360.0)
        outlet_pressure = 1e5 * (st.number_input('Outlet Pressure [bar abs]',
                                                     min_value=0.0,
                                                     max_value=float(fluid_pressure / 1e5 - 0.1),
                                                     value=2.2))

    with cols[2]:
        st.subheader('Valve')
        valve_type = st.selectbox('Choose Valve Type', ValveProperties.get_all_valve_types())
        trim_type = st.selectbox('Choose Trim Type', ValveProperties.get_trim_types(valve_type))
        flow_direction = st.selectbox('Choose Flow Direction',
                                      ValveProperties.get_flow_directions(valve_type, trim_type))
        pipe_size = st.number_input('Pipe Size [mm]',
                                           min_value=1.0,
                                           value=150.0)

    # Create fluid and valve objects
    st.session_state['Valve'] = Valve(
        fluid=Fluid(
            fluid_name=fluid_name,
            fluid_pressure=fluid_pressure,
            fluid_temperature=fluid_temperature),
        props=ValvePropertiesSelector(
            valve_type=valve_type,
            trim_type=trim_type,
            flow_direction=flow_direction,
            pipe_size=pipe_size)
    )

    # Add button to calculate valve parameters
    def switch_to_results():
        st.session_state.active_tab = "Results & Flow Curve"

    calculate_button = st.button('Size Valve',
                                 type='primary',
                                 on_click=switch_to_results)

# Results and flow curve tab
with tab2:
    if calculate_button or 'valve_results' in st.session_state:
        # Use the Valve.size_valve method to calculate valve parameters
        valve = st.session_state['Valve']
        results = valve.size_valve(flow_rate, fluid_pressure, outlet_pressure)
        st.session_state['valve_results'] = results

        # Display results
        st.subheader('Valve Sizing Results')

        # Create columns for results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader('Inlet')
            st.metric("Inlet Pressure [kPa]", f"{fluid_pressure / 1000:.1f}")
            st.metric("Inlet Temperature [°C]", f"{valve.fluid.T - 273.15:.1f}")
            st.metric("Density [kg/m³]", f"{valve.fluid.get_density():.1f}")
            st.metric("Vapor Pressure [kPa]", f"{valve.fluid.get_vapour_pressure() / 1000:.1f}")
            st.metric("Critical Pressure [kPa]", f"{valve.fluid.get_critical_pressure() / 1000:.1f}")
            st.metric("Kinematic Viscosity [m²/s]", f"{valve.fluid.get_viscosity()/valve.fluid.get_density():.2e}")

        with col2:
            st.subheader('Outlet')
            st.metric("Outlet Pressure [kPa]", f"{outlet_pressure / 1000:.1f}")
            st.metric("Pressure Drop [kPa]", f"{(fluid_pressure - outlet_pressure) / 1000:.1f}")
            st.metric("Choking Pressure Drop [kPa]", f"{valve.choking_delta_p/1e3:.1f}")
            st.metric("Liquid Pressure Recovery (FL)", f"{valve.FL:.3f}")
            st.metric("Critical Pressure Ratio Factor (FF)", f"{valve.FF:.3f}")
            if 'x' in results:
                st.metric("Pressure Drop Ratio (x)", f"{results['x']:.3f}")
            if 'Fk' in results:
                st.metric("Fk Factor", f"{results['Fk']:.3f}")
            st.metric("Flow Rate [m³/h]", f"{flow_rate:.1f}")

        with col3:
            st.subheader('Valve Dimensions')
            st.metric("Pipe Size [mm]", f"{pipe_size:.1f}")
            st.metric("Flow Coefficient (Cv)", f"{results['Cv']:.2f}")
            st.metric("Flow Coefficient (Kv)", f"{results['Kv']:.2f}")
            st.metric("Valve Area [mm²]", f"{results['Area'] * 1e6:.2f}")
            st.metric("Valve Style Modifier (Fd)", f"{valve.Fd:.3f}")

        with col4:
            st.subheader('Flow Regime')
            st.metric("Flow Regime", results['flow_regime'])
            st.metric("Is Flow Choked?", "Yes" if results['is_choked'] else "No")
            st.metric("Reynolds Number", f"{results['Reynolds_number']:.4e}")
            st.metric("Pipe Geometry Factor (FP)", f"{results['FP']:.3f}")
            if 'FR' in results and results['FR'] is not None:
                st.metric("Reynolds Number Factor (FR)", f"{results['FR']:.3f}")


        # Generate flow curve: Q vs. dp
        st.subheader('Flow Curve: Q vs. Pressure Drop')

        # Create array of pressure drops (in bar)
        delta_p_bar_array = np.linspace(0.1, max(7.0, (fluid_pressure - outlet_pressure)*1.1 / 1e5), 50)

        # Calculate flow rates for each pressure drop
        flow_rates = []
        for dp_bar in delta_p_bar_array:
            dp_pa = dp_bar * 1e5
            outlet_p = fluid_pressure - dp_pa
            # Don't allow negative outlet pressure
            if outlet_p < 0:
                outlet_p = 0

            # Calculate maximum flow rate for this Cv and pressure condition
            max_flow = valve.calculate_max_flow_rate(results['Cv'], fluid_pressure, outlet_p)
            flow_rates.append(max_flow)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(delta_p_bar_array, flow_rates, 'b-', linewidth=2)

        # Mark the design point
        design_dp = (fluid_pressure - outlet_pressure) / 1e5
        ax.plot(design_dp, flow_rate, 'ro', markersize=8, label='Design Point')

        # Check if flow is choked and mark choked flow region
        if results['is_choked']:
            # Find where flow becomes choked
            if valve.fluid.is_compressible:
                # For compressible fluids, calculate based on pressure ratio
                Fk = results.get('Fk', valve.Xt ** 0.5)
                choked_x = Fk * valve.Xt
                choked_dp_bar = choked_x * fluid_pressure / 1e5
            else:
                # For incompressible fluids
                FP = results['FP']
                FL = valve.FL
                choked_dp_bar = ((FL * FP) ** 2) * fluid_pressure / 1e5

            # Add vertical line for choked flow
            if choked_dp_bar < max(delta_p_bar_array):
                ax.axvline(x=choked_dp_bar, color='r', linestyle='--', label='Choked Flow Threshold')

                # Shade the choked flow region
                choked_indices = [i for i, dp in enumerate(delta_p_bar_array) if dp >= choked_dp_bar]
                if choked_indices:
                    ax.fill_between(delta_p_bar_array[min(choked_indices):],
                                    0,
                                    flow_rates[min(choked_indices):],
                                    color='red', alpha=0.2)

        # Add labels and title
        ax.set_xlabel('Pressure Drop [bar]')
        ax.set_ylabel('Flow Rate [m³/h]')
        ax.set_title(f'Flow Curve for {valve_type} Valve (Cv={results["Cv"]:.2f})')
        ax.grid(True)
        ax.legend()

        # Display the plot
        st.pyplot(fig)

        # Additional information
        st.subheader('Valve Information')
        st.markdown(f"**Fluid:** {fluid_name} ({'Compressible' if valve.fluid.is_compressible else 'Incompressible'})")
        st.markdown(f"**Valve Type:** {valve_type} with {trim_type} trim")
        st.markdown(f"**Flow Direction:** {flow_direction}")

        # Display valve sizing parameters
        st.subheader('Valve Sizing Parameters')
        st.markdown(f"**Fd (valve style modifier):** {valve.Fd:.3f}")
        st.markdown(f"**FL (pressure recovery factor):** {valve.FL:.3f}")
        st.markdown(f"**Xt (pressure drop ratio factor):** {valve.Xt:.3f}")
    else:
        st.info("Please enter input parameters and click 'Size Valve' to see results.")