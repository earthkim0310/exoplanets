# RV Exoplanet Simulator - Complete Rewrite
# Simple and clear visualization of exoplanet radial velocity detection

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Set page config
st.set_page_config(page_title="RV Exoplanet Simulator", layout="wide")

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_sun = 1.98847e30  # kg
M_jup = 1.89813e27  # kg
AU = 1.495978707e11  # m
c = 299792458.0  # m/s

def mjup_to_msun(mj):
    """Convert Jupiter mass to Solar mass"""
    return mj * (M_jup / M_sun)

def orbital_period(a_AU, M_star, M_planet):
    """Calculate orbital period in years"""
    M_total = M_star + M_planet
    return np.sqrt(a_AU**3 / M_total)

def rv_amplitude(a_AU, M_star, M_planet, e, i):
    """Calculate RV semi-amplitude in m/s"""
    P = orbital_period(a_AU, M_star, M_planet)
    P_sec = P * 365.25 * 24 * 3600
    M_total_kg = (M_star + M_planet) * M_sun
    M_planet_kg = M_planet * M_sun
    
    K = ((2 * np.pi * G) / P_sec)**(1/3) * (M_planet_kg * np.sin(i)) / (M_total_kg**(2/3)) * 1/np.sqrt(1 - e**2)
    return K

def kepler_equation_solver(M, e, max_iter=100):
    """Solve Kepler's equation using Newton-Raphson method"""
    E = M  # Initial guess
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        if abs(f_prime) < 1e-12:
            break
        E = E - f / f_prime
        if abs(f) < 1e-10:
            break
    return E

def true_anomaly(M, e):
    """Calculate true anomaly from mean anomaly"""
    E = kepler_equation_solver(M, e)
    cos_f = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_f = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    return np.arctan2(sin_f, cos_f)

def radial_velocity(t, a_AU, M_star, M_planet, e, omega, i, gamma=0):
    """Calculate radial velocity at time t"""
    P = orbital_period(a_AU, M_star, M_planet)
    P_sec = P * 365.25 * 24 * 3600
    
    # Mean anomaly
    M = 2 * np.pi * t / P_sec
    
    # True anomaly
    f = true_anomaly(M, e)
    
    # RV amplitude
    K = rv_amplitude(a_AU, M_star, M_planet, e, i)
    
    # Radial velocity
    vr = K * (np.cos(omega + f) + e * np.cos(omega)) + gamma
    return vr, f

def doppler_shift(lambda0, vr):
    """Calculate Doppler shift in wavelength"""
    beta = vr / c
    return lambda0 * np.sqrt((1 + beta) / (1 - beta))

# Streamlit UI
st.title("🌌 RV Exoplanet Simulator")
st.caption("Visualize how exoplanets are detected using radial velocity method")

# Sidebar controls
st.sidebar.header("System Parameters")

# Star parameters
M_star = st.sidebar.slider("Star Mass (M☉)", 0.1, 5.0, 1.0, 0.1)

# Planet parameters
planet_mass_mode = st.sidebar.radio("Planet Mass Unit", ["Jupiter Mass", "Solar Mass"])
if planet_mass_mode == "Jupiter Mass":
    M_planet_jup = st.sidebar.slider("Planet Mass (MJup)", 0.1, 20.0, 1.0, 0.1)
    M_planet = mjup_to_msun(M_planet_jup)
else:
    M_planet = st.sidebar.slider("Planet Mass (M☉)", 0.001, 0.1, 0.003, 0.001)

# Orbital parameters
a_AU = st.sidebar.slider("Semi-major Axis (AU)", 0.02, 5.0, 1.0, 0.01)
e = st.sidebar.slider("Eccentricity", 0.0, 0.95, 0.0, 0.01)
i_deg = st.sidebar.slider("Inclination (degrees)", 0.0, 90.0, 60.0, 1.0)
omega_deg = st.sidebar.slider("Argument of Periastron (degrees)", 0.0, 360.0, 0.0, 1.0)
gamma = st.sidebar.slider("System Velocity (m/s)", -50000.0, 50000.0, 0.0, 10.0)

# Convert to radians
i = np.radians(i_deg)
omega = np.radians(omega_deg)

# Animation controls
st.sidebar.header("Animation")
use_animation = st.sidebar.checkbox("Enable Animation", value=False)
if use_animation:
    speed = st.sidebar.slider("Animation Speed", 0.1, 10.0, 2.0, 0.1)
    if st.sidebar.button("Play/Pause"):
        if 'playing' not in st.session_state:
            st.session_state.playing = False
        st.session_state.playing = not st.session_state.playing
        if st.session_state.playing:
            st.session_state.start_time = time.time()

# Current phase
if use_animation and st.session_state.get('playing', False):
    elapsed = (time.time() - st.session_state.start_time) * speed
    P = orbital_period(a_AU, M_star, M_planet)
    P_sec = P * 365.25 * 24 * 3600
    phase = (elapsed / P_sec) % 1.0
    st.write(f"Current Phase: {phase:.3f}")
else:
    phase = st.sidebar.slider("Phase (0-1)", 0.0, 1.0, 0.0, 0.01)

# Calculate orbital period
P = orbital_period(a_AU, M_star, M_planet)
P_days = P * 365.25

# Current time
t_current = phase * P * 365.25 * 24 * 3600  # seconds

# Calculate current RV and true anomaly
vr_current, f_current = radial_velocity(t_current, a_AU, M_star, M_planet, e, omega, i, gamma)

# Generate RV curve for one period
t_period = np.linspace(0, P * 365.25 * 24 * 3600, 1000)
vr_curve, f_curve = radial_velocity(t_period, a_AU, M_star, M_planet, e, omega, i, gamma)

# Calculate orbital positions
a_star = a_AU * (M_planet / (M_star + M_planet))
a_planet = a_AU * (M_star / (M_star + M_planet))

# Current positions
r_star = a_star * (1 - e**2) / (1 + e * np.cos(f_current))
r_planet = a_planet * (1 - e**2) / (1 + e * np.cos(f_current + np.pi))

x_star = r_star * np.cos(f_current) * np.cos(omega) - r_star * np.sin(f_current) * np.sin(omega)
y_star = r_star * np.cos(f_current) * np.sin(omega) + r_star * np.sin(f_current) * np.cos(omega)
x_planet = r_planet * np.cos(f_current + np.pi) * np.cos(omega) - r_planet * np.sin(f_current + np.pi) * np.sin(omega)
y_planet = r_planet * np.cos(f_current + np.pi) * np.sin(omega) + r_planet * np.sin(f_current + np.pi) * np.cos(omega)

# Project to sky plane
x_star_proj = x_star
y_star_proj = y_star * np.cos(i)
x_planet_proj = x_planet
y_planet_proj = y_planet * np.cos(i)

# Create plots
col1, col2, col3 = st.columns([1, 1, 1])

# Orbit plot
with col1:
    st.subheader("Orbital Motion")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    
    # Plot orbits
    f_full = np.linspace(0, 2*np.pi, 1000)
    r_star_full = a_star * (1 - e**2) / (1 + e * np.cos(f_full))
    r_planet_full = a_planet * (1 - e**2) / (1 + e * np.cos(f_full + np.pi))
    
    x_star_full = r_star_full * np.cos(f_full) * np.cos(omega) - r_star_full * np.sin(f_full) * np.sin(omega)
    y_star_full = r_star_full * np.cos(f_full) * np.sin(omega) + r_star_full * np.sin(f_full) * np.cos(omega)
    x_planet_full = r_planet_full * np.cos(f_full + np.pi) * np.cos(omega) - r_planet_full * np.sin(f_full + np.pi) * np.sin(omega)
    y_planet_full = r_planet_full * np.cos(f_full + np.pi) * np.sin(omega) + r_planet_full * np.sin(f_full + np.pi) * np.cos(omega)
    
    # Project to sky plane
    x_star_proj_full = x_star_full
    y_star_proj_full = y_star_full * np.cos(i)
    x_planet_proj_full = x_planet_full
    y_planet_proj_full = y_planet_full * np.cos(i)
    
    ax1.plot(x_planet_proj_full/AU, y_planet_proj_full/AU, 'b-', label='Planet Orbit', alpha=0.7)
    ax1.plot(x_star_proj_full/AU, y_star_proj_full/AU, 'r-', label='Star Orbit', alpha=0.7)
    
    # Current positions
    ax1.scatter(x_planet_proj/AU, y_planet_proj/AU, c='blue', s=100, label='Planet (now)')
    ax1.scatter(x_star_proj/AU, y_star_proj/AU, c='red', s=100, marker='*', label='Star (now)')
    ax1.scatter(0, 0, c='black', s=50, label='Center of Mass')
    
    ax1.set_xlabel('X (AU)')
    ax1.set_ylabel('Y (AU)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Orbital Motion (Sky View)')
    
    st.pyplot(fig1)

# RV curve
with col2:
    st.subheader("Radial Velocity Curve")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    t_days = t_period / (24 * 3600)
    ax2.plot(t_days, vr_curve, 'b-', linewidth=2)
    ax2.axvline(t_current / (24 * 3600), color='red', linestyle='--', alpha=0.7, label='Current Time')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Radial Velocity (m/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('RV vs Time')
    
    # Add info text
    K = rv_amplitude(a_AU, M_star, M_planet, e, i)
    info_text = f'Period: {P:.2f} years\nK = {K:.1f} m/s\nCurrent RV: {vr_current:.1f} m/s'
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    st.pyplot(fig2)

# Spectrum
with col3:
    st.subheader("Doppler Shift in Spectrum")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    
    # Wavelength range
    lambda0 = 656.28  # H-alpha line in nm
    vr_max = np.max(np.abs(vr_curve))
    delta_lambda_max = lambda0 * vr_max / c
    lambda_range = np.linspace(lambda0 - 3*delta_lambda_max, lambda0 + 3*delta_lambda_max, 1000)
    
    # Current Doppler shift
    lambda_current = doppler_shift(lambda0, vr_current)
    
    # Create spectrum with rainbow background
    continuum = np.ones_like(lambda_range)
    absorption_depth = 0.3
    sigma = delta_lambda_max / 10
    
    # Absorption line
    absorption_line = 1 - absorption_depth * np.exp(-0.5 * ((lambda_range - lambda_current) / sigma)**2)
    
    # Plot with rainbow colors
    for i in range(len(lambda_range)-1):
        color = plt.cm.rainbow(i / len(lambda_range))
        ax3.plot([lambda_range[i], lambda_range[i+1]], [absorption_line[i], absorption_line[i+1]], 
                color=color, linewidth=3, alpha=0.8)
    
    # Highlight absorption line
    ax3.plot(lambda_range, absorption_line, 'k-', linewidth=2, alpha=0.9, label='Absorption Line')
    
    # Reference lines
    ax3.axvline(lambda0, color='gray', linestyle='--', alpha=0.7, label='Rest Wavelength')
    ax3.axvline(lambda_current, color='red', linestyle='--', alpha=0.9, linewidth=2, label='Current Wavelength')
    
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Relative Intensity')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title('Spectral Line Doppler Shift')
    
    # Add shift info
    shift_nm = lambda_current - lambda0
    if vr_current > 0:
        shift_text = f'REDSHIFT\nΔλ = +{shift_nm:.4f} nm\nMoving Away'
        color = 'red'
    elif vr_current < 0:
        shift_text = f'BLUESHIFT\nΔλ = {shift_nm:.4f} nm\nMoving Toward'
        color = 'blue'
    else:
        shift_text = f'NO SHIFT\nΔλ = 0 nm\nNo Motion'
        color = 'gray'
    
    ax3.text(0.02, 0.98, shift_text, transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    st.pyplot(fig3)

# Auto-refresh for animation
if use_animation and st.session_state.get('playing', False):
    time.sleep(0.1)
    st.rerun()