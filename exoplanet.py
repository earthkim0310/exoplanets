# Improved RV Exoplanet Simulator with Better Animation
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better compatibility
matplotlib.use('Agg')
plt.style.use('default')

# Set safe matplotlib parameters
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Page config
st.set_page_config(page_title="RV Exoplanet Simulator", layout="wide")

# Constants
G = 6.67430e-11
M_sun = 1.98847e30
M_jup = 1.89813e27
AU = 1.495978707e11
c = 299792458.0

def mjup_to_msun(mj):
    """Convert Jupiter masses to Solar masses"""
    return mj * (M_jup / M_sun)

def orbital_period(a_AU, M_star, M_planet):
    """Calculate orbital period using Kepler's 3rd law"""
    M_total = M_star + M_planet
    return np.sqrt(a_AU**3 / M_total)

def rv_amplitude(a_AU, M_star, M_planet, e, i):
    """Calculate RV amplitude K"""
    P = orbital_period(a_AU, M_star, M_planet)
    P_sec = P * 365.25 * 24 * 3600
    M_total_kg = (M_star + M_planet) * M_sun
    M_planet_kg = M_planet * M_sun
    
    try:
        K = ((2 * np.pi * G) / P_sec)**(1/3) * (M_planet_kg * np.sin(i)) / (M_total_kg**(2/3)) * 1/np.sqrt(1 - e**2)
        return K
    except:
        return 0.0

def simple_rv(t, a_AU, M_star, M_planet, e, omega, i, gamma=0):
    """Calculate radial velocity"""
    try:
        P = orbital_period(a_AU, M_star, M_planet)
        P_sec = P * 365.25 * 24 * 3600
        
        # Mean anomaly
        M_anom = 2 * np.pi * t / P_sec
        
        # Simple eccentric anomaly approximation
        f = M_anom + 2 * e * np.sin(M_anom)
        
        K = rv_amplitude(a_AU, M_star, M_planet, e, i)
        vr = K * (np.cos(f + omega) + e * np.cos(omega)) + gamma
        return vr, f
    except:
        return gamma, 0.0

# Initialize session state for animation
if 'animation_time' not in st.session_state:
    st.session_state.animation_time = 0.0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

# UI Header
st.title("🌌 Enhanced RV Exoplanet Simulator")
st.caption("Observe real-time radial velocity changes and spectral shifts as planets orbit their stars")

# Sidebar Controls
st.sidebar.header("🔧 System Parameters")

# Star mass
M_star = st.sidebar.slider("Star Mass (M☉)", 0.1, 5.0, 1.0, 0.1, 
                          help="Mass of the host star in solar masses")

# Planet mass selection
planet_mass_mode = st.sidebar.radio("Planet Mass Unit", ["Jupiter Mass", "Solar Mass"])
if planet_mass_mode == "Jupiter Mass":
    M_planet_jup = st.sidebar.slider("Planet Mass (MJup)", 0.1, 20.0, 1.0, 0.1,
                                    help="Planet mass in Jupiter masses")
    M_planet = mjup_to_msun(M_planet_jup)
    st.sidebar.caption(f"≈ {M_planet:.4f} M☉")
else:
    M_planet = st.sidebar.slider("Planet Mass (M☉)", 0.001, 0.1, 0.003, 0.001,
                                help="Planet mass in solar masses")
    st.sidebar.caption(f"≈ {M_planet/mjup_to_msun(1):.1f} MJup")

# Orbital parameters
a_AU = st.sidebar.slider("Semi-major Axis (AU)", 0.02, 5.0, 1.0, 0.01,
                        help="Distance between star and planet")
e = st.sidebar.slider("Eccentricity", 0.0, 0.95, 0.0, 0.01,
                     help="0 = circular orbit, close to 1 = very elliptical")
i_deg = st.sidebar.slider("Inclination (degrees)", 0.0, 90.0, 60.0, 1.0,
                         help="Orbital tilt: 90° = edge-on, 0° = face-on")
omega_deg = st.sidebar.slider("Argument of Periastron (degrees)", 0.0, 360.0, 0.0, 1.0,
                             help="Orientation of elliptical orbit")
gamma = st.sidebar.slider("System Velocity (m/s)", -50000.0, 50000.0, 0.0, 10.0,
                         help="Bulk motion of the entire star system")

# Convert to radians
i = np.radians(i_deg)
omega = np.radians(omega_deg)

# Animation controls
st.sidebar.header("🎬 Animation Controls")
use_animation = st.sidebar.checkbox("Enable Real-time Animation", value=False)

if use_animation:
    speed_multiplier = st.sidebar.slider("Animation Speed Multiplier", 0.1, 50.0, 10.0, 0.1,
                                        help="How fast the animation plays")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    if col1.button("▶️", help="Play animation"):
        st.session_state.is_playing = True
        st.session_state.last_update = time.time()
    
    if col2.button("⏸️", help="Pause animation"):
        st.session_state.is_playing = False
    
    if col3.button("🔄", help="Reset to start"):
        st.session_state.animation_time = 0.0
        st.session_state.is_playing = False
        st.session_state.last_update = time.time()

# Calculate system properties
P = orbital_period(a_AU, M_star, M_planet)
P_days = P * 365.25
P_sec = P * 365.25 * 24 * 3600
K = rv_amplitude(a_AU, M_star, M_planet, e, i)

# Update animation time
if use_animation and st.session_state.is_playing:
    current_time = time.time()
    dt = current_time - st.session_state.last_update
    st.session_state.animation_time += dt * speed_multiplier
    st.session_state.last_update = current_time
    
    # Keep animation time within reasonable bounds
    if st.session_state.animation_time > P_sec:
        st.session_state.animation_time = st.session_state.animation_time % P_sec

# Current phase calculation
if use_animation:
    t_current = st.session_state.animation_time
    phase = (t_current / P_sec) % 1.0
    if st.session_state.is_playing:
        st.sidebar.success(f"🎬 **Playing** - Phase: {phase:.3f}")
    else:
        st.sidebar.info(f"⏸️ **Paused** - Phase: {phase:.3f}")
else:
    phase = st.sidebar.slider("Manual Phase (0-1)", 0.0, 1.0, 0.0, 0.01,
                             help="Manually control orbital position")
    t_current = phase * P_sec

# Calculate current state
vr_current, f_current = simple_rv(t_current, a_AU, M_star, M_planet, e, omega, i, gamma)

# Generate extended time series for plotting
n_periods = 2
t_extended = np.linspace(0, n_periods * P_sec, 1000)
vr_extended = np.array([simple_rv(t, a_AU, M_star, M_planet, e, omega, i, gamma)[0] 
                       for t in t_extended])

# Calculate positions for visualization
a_star = a_AU * (M_planet / (M_star + M_planet))  # Star's orbit radius
a_planet = a_AU * (M_star / (M_star + M_planet))  # Planet's orbit radius

try:
    r_star = a_star * (1 - e**2) / (1 + e * np.cos(f_current))
    r_planet = a_planet * (1 - e**2) / (1 + e * np.cos(f_current + np.pi))
    
    x_star = r_star * np.cos(f_current + omega)
    y_star = r_star * np.sin(f_current + omega) * np.cos(i)
    x_planet = r_planet * np.cos(f_current + omega + np.pi)
    y_planet = r_planet * np.sin(f_current + omega + np.pi) * np.cos(i)
except:
    x_star = y_star = x_planet = y_planet = 0.0

# Display system info
st.subheader("📊 Current System Status")
col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric("Orbital Period", f"{P:.2f} years", f"{P_days:.0f} days")

with col_info2:
    st.metric("Current RV", f"{vr_current:.1f} m/s")

with col_info3:
    st.metric("RV Semi-amplitude", f"{K:.1f} m/s")

with col_info4:
    st.metric("Orbital Phase", f"{phase:.3f}")

# Create visualization columns
col1, col2 = st.columns([1, 1])

# Orbital motion visualization
with col1:
    st.subheader("🪐 Orbital Motion")
    
    try:
        # Create figure
        fig1 = plt.figure(figsize=(6, 6), dpi=80, facecolor='white')
        ax1 = fig1.add_subplot(111)
        
        # Generate full orbital paths
        f_orbit = np.linspace(0, 2*np.pi, 200)
        r_star_orbit = a_star * (1 - e**2) / (1 + e * np.cos(f_orbit))
        r_planet_orbit = a_planet * (1 - e**2) / (1 + e * np.cos(f_orbit + np.pi))
        
        x_star_orbit = r_star_orbit * np.cos(f_orbit + omega)
        y_star_orbit = r_star_orbit * np.sin(f_orbit + omega) * np.cos(i)
        x_planet_orbit = r_planet_orbit * np.cos(f_orbit + omega + np.pi)
        y_planet_orbit = r_planet_orbit * np.sin(f_orbit + omega + np.pi) * np.cos(i)
        
        # Plot orbits
        ax1.plot(x_planet_orbit/AU, y_planet_orbit/AU, 'b-', alpha=0.5, linewidth=2, 
                label='Planet Orbit')
        ax1.plot(x_star_orbit/AU, y_star_orbit/AU, 'r-', alpha=0.5, linewidth=2,
                label='Star Orbit')
        
        # Plot current positions
        planet_size = 150 + 50 * (M_planet / mjup_to_msun(1))
        star_size = 200 + 50 * M_star
        
        ax1.scatter(x_planet/AU, y_planet/AU, c='blue', s=planet_size, 
                   label='Planet', zorder=5, edgecolors='darkblue', linewidth=1)
        ax1.scatter(x_star/AU, y_star/AU, c='gold', s=star_size, marker='*',
                   label='Star', zorder=5, edgecolors='orange', linewidth=1)
        ax1.scatter(0, 0, c='black', s=50, marker='x', label='Center of Mass', zorder=5)
        
        # Add motion indicator
        if abs(vr_current) > 1.0:
            motion_text = "→ Moving Away" if vr_current > 0 else "← Moving Toward"
            color = 'red' if vr_current > 0 else 'blue'
            ax1.text(0.02, 0.98, motion_text, transform=ax1.transAxes, 
                    verticalalignment='top', color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))
        
        ax1.set_xlabel('X Position (AU)')
        ax1.set_ylabel('Y Position (AU)')
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_title('Orbital Motion (View from Above)')
        
        # Set axis limits
        max_extent = max(a_planet, a_star) * (1 + e) / AU * 1.3
        ax1.set_xlim(-max_extent, max_extent)
        ax1.set_ylim(-max_extent, max_extent)
        
        st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)
        
    except Exception as e:
        st.error("Error creating orbital plot")

# Radial velocity and spectrum plots
with col2:
    st.subheader("📈 Radial Velocity & Spectrum")
    
    try:
        # Create figure with two subplots
        fig2 = plt.figure(figsize=(6, 8), dpi=80, facecolor='white')
        ax2 = fig2.add_subplot(2, 1, 1)
        ax3 = fig2.add_subplot(2, 1, 2)
        
        # RV curve plot
        t_days = t_extended / (24 * 3600)
        t_current_days = t_current / (24 * 3600)
        
        ax2.plot(t_days, vr_extended, 'b-', linewidth=2, label='RV Curve')
        ax2.axvline(t_current_days, color='red', linestyle='--', linewidth=3, alpha=0.7,
                   label='Current Time')
        ax2.scatter([t_current_days], [vr_current], color='red', s=80, zorder=5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        if gamma != 0:
            ax2.axhline(gamma, color='gray', linestyle=':', alpha=0.7, label='System Velocity')
        
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Radial Velocity (m/s)')
        ax2.legend(fontsize=8)
        ax2.set_title('Radial Velocity vs Time')
        
        # Add info text
        info_text = f'Period: {P:.2f} yr\nK: {K:.1f} m/s\nRV: {vr_current:.1f} m/s'
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Spectrum plot (Doppler shift visualization)
        lambda0 = 656.3  # H-alpha wavelength in nm
        vr_range = max(abs(np.min(vr_extended)), abs(np.max(vr_extended)))
        
        if vr_range > 0:
            delta_lambda_max = lambda0 * vr_range / c
            lambda_range = np.linspace(lambda0 - 3*delta_lambda_max, 
                                     lambda0 + 3*delta_lambda_max, 500)
            
            # Current shifted wavelength
            lambda_current = lambda0 * (1 + vr_current / c)
            
            # Create absorption line
            sigma = delta_lambda_max / 4
            absorption_line = 1 - 0.4 * np.exp(-0.5 * ((lambda_range - lambda_current) / sigma)**2)
            
            ax3.plot(lambda_range, absorption_line, 'k-', linewidth=3, label='Absorption Line')
            ax3.axvline(lambda0, color='gray', linestyle='--', alpha=0.7, linewidth=2,
                       label='Rest Wavelength')
            ax3.axvline(lambda_current, color='red', linestyle='-', alpha=0.9, linewidth=2,
                       label='Current Wavelength')
            
            # Show shift direction
            if abs(vr_current) > 1:
                if vr_current > 0:
                    ax3.fill_betweenx([0, 1.1], lambda0, lambda_current, alpha=0.2, color='red')
                    shift_text = '🔴 Redshift'
                else:
                    ax3.fill_betweenx([0, 1.1], lambda_current, lambda0, alpha=0.2, color='blue')
                    shift_text = '🔵 Blueshift'
                
                ax3.text(0.02, 0.98, shift_text, transform=ax3.transAxes,
                        verticalalignment='top', fontsize=10, fontweight='bold')
            
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Relative Intensity')
            ax3.set_ylim(0, 1.1)
            ax3.legend(fontsize=8)
            ax3.set_title('Spectral Line Doppler Shift')
            
            # Show quantitative shift
            shift_nm = lambda_current - lambda0
            if abs(shift_nm) > 1e-6:
                shift_text = f'Δλ = {shift_nm:.4f} nm'
                ax3.text(0.98, 0.02, shift_text, transform=ax3.transAxes,
                        horizontalalignment='right', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
        
    except Exception as e:
        st.error("Error creating RV/spectrum plots")

# Educational information
with st.expander("ℹ️ How Radial Velocity Detection Works"):
    st.markdown("""
    **The Radial Velocity Method** detects exoplanets by measuring tiny changes in a star's motion:
    
    1. **Gravitational Dance**: Both star and planet orbit their common center of mass
    2. **Doppler Effect**: Star's motion causes spectral lines to shift red/blue
    3. **Periodic Signal**: Regular shifts reveal planetary orbits
    4. **Planet Properties**: Signal strength and period tell us about the planet
    
    **Key Parameters:**
    - **K (Semi-amplitude)**: Maximum RV change (larger planets → larger K)
    - **Period**: Time for one orbit (farther planets → longer periods)  
    - **Inclination**: Edge-on orbits (90°) give strongest signals
    - **Eccentricity**: Elliptical orbits create asymmetric RV curves
    """)

# Auto-refresh for animation
if use_animation and st.session_state.is_playing:
    time.sleep(0.1)  # 100ms refresh rate for smoother animation
    st.rerun()