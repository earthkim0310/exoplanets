# Improved RV Exoplanet Simulator with Better Animation
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Page config
st.set_page_config(page_title="RV Exoplanet Simulator", layout="wide")

# Constants
G = 6.67430e-11
M_sun = 1.98847e30
M_jup = 1.89813e27
AU = 1.495978707e11
c = 299792458.0

def mjup_to_msun(mj):
    return mj * (M_jup / M_sun)

def orbital_period(a_AU, M_star, M_planet):
    M_total = M_star + M_planet
    return np.sqrt(a_AU**3 / M_total)

def rv_amplitude(a_AU, M_star, M_planet, e, i):
    P = orbital_period(a_AU, M_star, M_planet)
    P_sec = P * 365.25 * 24 * 3600
    M_total_kg = (M_star + M_planet) * M_sun
    M_planet_kg = M_planet * M_sun
    K = ((2 * np.pi * G) / P_sec)**(1/3) * (M_planet_kg * np.sin(i)) / (M_total_kg**(2/3)) * 1/np.sqrt(1 - e**2)
    return K

def simple_rv(t, a_AU, M_star, M_planet, e, omega, i, gamma=0):
    """Simplified RV calculation"""
    P = orbital_period(a_AU, M_star, M_planet)
    P_sec = P * 365.25 * 24 * 3600
    
    # Mean anomaly
    M_anom = 2 * np.pi * t / P_sec
    
    # For simplicity, use M_anom as true anomaly (circular approximation)
    f = M_anom + 2 * e * np.sin(M_anom)  # Simple eccentric correction
    
    K = rv_amplitude(a_AU, M_star, M_planet, e, i)
    vr = K * (np.cos(f + omega) + e * np.cos(omega)) + gamma
    return vr, f

# Initialize session state for animation
if 'animation_time' not in st.session_state:
    st.session_state.animation_time = 0.0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

# UI
st.title("🌌 개선된 RV 외계행성 시뮬레이터")
st.caption("시간 변화에 따른 시선속도 변화와 스펙트럼 편이를 실시간으로 관찰하세요")

# Sidebar
st.sidebar.header("시스템 매개변수")

# Star mass
M_star = st.sidebar.slider("별의 질량 (M☉)", 0.1, 5.0, 1.0, 0.1)

# Planet mass
planet_mass_mode = st.sidebar.radio("행성 질량 단위", ["목성 질량", "태양 질량"])
if planet_mass_mode == "목성 질량":
    M_planet_jup = st.sidebar.slider("행성 질량 (MJup)", 0.1, 20.0, 1.0, 0.1)
    M_planet = mjup_to_msun(M_planet_jup)
else:
    M_planet = st.sidebar.slider("행성 질량 (M☉)", 0.001, 0.1, 0.003, 0.001)

# Orbital parameters
a_AU = st.sidebar.slider("궤도 반지름 (AU)", 0.02, 5.0, 1.0, 0.01)
e = st.sidebar.slider("이심률", 0.0, 0.95, 0.0, 0.01)
i_deg = st.sidebar.slider("경사각 (도)", 0.0, 90.0, 60.0, 1.0)
omega_deg = st.sidebar.slider("근점 인수 (도)", 0.0, 360.0, 0.0, 1.0)
gamma = st.sidebar.slider("시스템 속도 (m/s)", -50000.0, 50000.0, 0.0, 10.0)

# Convert to radians
i = np.radians(i_deg)
omega = np.radians(omega_deg)

# Animation controls
st.sidebar.header("애니메이션 제어")
use_animation = st.sidebar.checkbox("실시간 애니메이션 활성화", value=False)

if use_animation:
    speed_multiplier = st.sidebar.slider("애니메이션 속도", 0.1, 20.0, 5.0, 0.1)
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    if col1.button("▶️ 재생"):
        st.session_state.is_playing = True
        st.session_state.last_update = time.time()
    
    if col2.button("⏸️ 일시정지"):
        st.session_state.is_playing = False
    
    if col3.button("🔄 초기화"):
        st.session_state.animation_time = 0.0
        st.session_state.is_playing = False
        st.session_state.last_update = time.time()

# Calculate orbital period
P = orbital_period(a_AU, M_star, M_planet)
P_days = P * 365.25
P_sec = P * 365.25 * 24 * 3600

# Update animation time
if use_animation and st.session_state.is_playing:
    current_time = time.time()
    dt = current_time - st.session_state.last_update
    st.session_state.animation_time += dt * speed_multiplier
    st.session_state.last_update = current_time
    
    # Keep animation time within one period for cleaner display
    if st.session_state.animation_time > P_sec:
        st.session_state.animation_time = st.session_state.animation_time % P_sec

# Current phase calculation
if use_animation:
    t_current = st.session_state.animation_time
    phase = (t_current / P_sec) % 1.0
    if st.session_state.is_playing:
        st.sidebar.success(f"🎬 **애니메이션 실행 중** - 위상: {phase:.3f}")
    else:
        st.sidebar.info(f"⏸️ **일시정지됨** - 위상: {phase:.3f}")
else:
    phase = st.sidebar.slider("수동 위상 조절 (0-1)", 0.0, 1.0, 0.0, 0.01)
    t_current = phase * P_sec

# Calculate current RV
vr_current, f_current = simple_rv(t_current, a_AU, M_star, M_planet, e, omega, i, gamma)

# Generate RV curve for multiple periods for better visualization
n_periods = 2
t_extended = np.linspace(0, n_periods * P_sec, 2000)
vr_extended, f_extended = simple_rv(t_extended, a_AU, M_star, M_planet, e, omega, i, gamma)

# Calculate orbital positions
a_star = a_AU * (M_planet / (M_star + M_planet))
a_planet = a_AU * (M_star / (M_star + M_planet))

# Current positions (improved calculation)
r_star = a_star * (1 - e**2) / (1 + e * np.cos(f_current))
r_planet = a_planet * (1 - e**2) / (1 + e * np.cos(f_current + np.pi))

x_star = r_star * np.cos(f_current + omega)
y_star = r_star * np.sin(f_current + omega) * np.cos(i)
x_planet = r_planet * np.cos(f_current + omega + np.pi)
y_planet = r_planet * np.sin(f_current + omega + np.pi) * np.cos(i)

# Display current system info
st.subheader("📊 현재 시스템 상태")
col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric("궤도 주기", f"{P:.2f} 년", f"{P_days:.1f} 일")

with col_info2:
    st.metric("현재 시선속도", f"{vr_current:.1f} m/s")

with col_info3:
    K = rv_amplitude(a_AU, M_star, M_planet, e, i)
    st.metric("RV 반폭 (K)", f"{K:.1f} m/s")

with col_info4:
    st.metric("현재 위상", f"{phase:.3f}")

# Create plots
col1, col2 = st.columns([1, 1])

# Orbit plot
with col1:
    st.subheader("🪐 궤도 운동")
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    # Plot full orbits
    f_full = np.linspace(0, 2*np.pi, 1000)
    r_star_full = a_star * (1 - e**2) / (1 + e * np.cos(f_full))
    r_planet_full = a_planet * (1 - e**2) / (1 + e * np.cos(f_full + np.pi))
    
    x_star_full = r_star_full * np.cos(f_full + omega)
    y_star_full = r_star_full * np.sin(f_full + omega) * np.cos(i)
    x_planet_full = r_planet_full * np.cos(f_full + omega + np.pi)
    y_planet_full = r_planet_full * np.sin(f_full + omega + np.pi) * np.cos(i)
    
    # Plot orbits
    ax1.plot(x_planet_full/AU, y_planet_full/AU, 'b-', label='행성 궤도', alpha=0.5, linewidth=2)
    ax1.plot(x_star_full/AU, y_star_full/AU, 'r-', label='별 궤도', alpha=0.5, linewidth=2)
    
    # Current positions with larger markers
    planet_size = 200 + 100 * (M_planet / mjup_to_msun(1))  # Scale with mass
    star_size = 300 + 100 * M_star
    
    ax1.scatter(x_planet/AU, y_planet/AU, c='blue', s=planet_size, 
               label=f'행성 (현재)', zorder=5, edgecolors='darkblue', linewidth=2)
    ax1.scatter(x_star/AU, y_star/AU, c='gold', s=star_size, marker='*', 
               label=f'별 (현재)', zorder=5, edgecolors='orange', linewidth=2)
    ax1.scatter(0, 0, c='black', s=100, label='질량중심', zorder=5, marker='x')
    
    # Add velocity vectors
    if abs(vr_current) > 0.1:  # Only show if significant
        scale = 0.5 / K  # Scale factor for velocity vectors
        if vr_current > 0:
            ax1.arrow(x_star/AU, y_star/AU, 0, scale*vr_current, head_width=0.05, 
                     head_length=0.03, fc='red', ec='red', alpha=0.8, linewidth=2)
            ax1.text(x_star/AU + 0.1, y_star/AU + 0.1, '멀어짐', color='red', fontweight='bold')
        else:
            ax1.arrow(x_star/AU, y_star/AU, 0, scale*vr_current, head_width=0.05, 
                     head_length=0.03, fc='blue', ec='blue', alpha=0.8, linewidth=2)
            ax1.text(x_star/AU + 0.1, y_star/AU + 0.1, '가까워짐', color='blue', fontweight='bold')
    
    ax1.set_xlabel('X (AU)')
    ax1.set_ylabel('Y (AU)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_title('궤도 운동 (하늘에서 본 모습)')
    
    # Set reasonable axis limits
    max_orbit = max(a_planet, a_star) * (1 + e) / AU * 1.2
    ax1.set_xlim(-max_orbit, max_orbit)
    ax1.set_ylim(-max_orbit, max_orbit)
    
    st.pyplot(fig1, use_container_width=True)

# RV curve and spectrum
with col2:
    st.subheader("📈 시선속도 곡선")
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 10))
    
    # RV curve
    t_days_extended = t_extended / (24 * 3600)
    t_current_days = t_current / (24 * 3600)
    
    ax2.plot(t_days_extended, vr_extended, 'b-', linewidth=2, label='시선속도 곡선')
    ax2.axvline(t_current_days, color='red', linestyle='--', alpha=0.8, linewidth=3, 
               label=f'현재 시점')
    ax2.scatter(t_current_days, vr_current, color='red', s=100, zorder=5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(gamma, color='gray', linestyle=':', alpha=0.5, label='시스템 속도')
    
    ax2.set_xlabel('시간 (일)')
    ax2.set_ylabel('시선속도 (m/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('시선속도 vs 시간')
    
    # Add phase information
    phase_text = f'주기: {P:.2f} 년 ({P_days:.1f} 일)\nK = {K:.1f} m/s\n현재 RV: {vr_current:.1f} m/s\n위상: {phase:.3f}'
    ax2.text(0.02, 0.98, phase_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Spectrum with Doppler shift
    ax3.set_title('🌈 스펙트럼 선의 도플러 편이')
    
    # Wavelength range around H-alpha
    lambda0 = 656.28  # H-alpha line in nm
    vr_max = max(abs(np.min(vr_extended)), abs(np.max(vr_extended)))
    delta_lambda_max = lambda0 * vr_max / c
    lambda_range = np.linspace(lambda0 - 4*delta_lambda_max, lambda0 + 4*delta_lambda_max, 1000)
    
    # Current Doppler shift
    lambda_current = lambda0 * (1 + vr_current / c)
    
    # Create spectrum
    continuum = np.ones_like(lambda_range)
    absorption_depth = 0.5
    sigma = delta_lambda_max / 6
    
    # Absorption line at current position
    absorption_line = 1 - absorption_depth * np.exp(-0.5 * ((lambda_range - lambda_current) / sigma)**2)
    
    # Plot spectrum with colors
    ax3.plot(lambda_range, absorption_line, 'k-', linewidth=3, label='흡수선')
    ax3.axhline(1, color='gray', linestyle='-', alpha=0.3)
    
    # Reference lines
    ax3.axvline(lambda0, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='정지 파장')
    ax3.axvline(lambda_current, color='red', linestyle='-', alpha=0.9, linewidth=3, 
               label='현재 파장')
    
    # Fill area to show shift direction
    if vr_current > 0:
        ax3.fill_betweenx([0, 1.1], lambda0, lambda_current, alpha=0.2, color='red', 
                         label='적색편이 (멀어짐)')
    elif vr_current < 0:
        ax3.fill_betweenx([0, 1.1], lambda_current, lambda0, alpha=0.2, color='blue', 
                         label='청색편이 (가까워짐)')
    
    ax3.set_xlabel('파장 (nm)')
    ax3.set_ylabel('상대 강도')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add shift information
    shift_nm = lambda_current - lambda0
    shift_velocity = shift_nm / lambda0 * c
    if abs(vr_current) > 0.1:
        if vr_current > 0:
            shift_text = f'🔴 적색편이\nΔλ = +{shift_nm:.6f} nm\nv = +{shift_velocity:.1f} m/s'
            text_color = 'red'
        else:
            shift_text = f'🔵 청색편이\nΔλ = {shift_nm:.6f} nm\nv = {shift_velocity:.1f} m/s'
            text_color = 'blue'
    else:
        shift_text = f'⚪ 편이 없음\nΔλ ≈ 0 nm\nv ≈ 0 m/s'
        text_color = 'gray'
    
    ax3.text(0.02, 0.98, shift_text, transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=text_color, alpha=0.2))
    
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# Auto-refresh for animation
if use_animation and st.session_state.is_playing:
    time.sleep(0.05)  # 50ms refresh rate
    st.rerun()