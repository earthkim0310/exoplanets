# rv_streamlit_app.py
# 외계행성 RV(Doppler) 시뮬레이터 — 궤도/시선속도/스펙트럼 동시 시각화 (Streamlit)
# 작성 목적: 수업용 데모 (별-행성 질량중심 공전 + 청/적색편이 + 변수 조절)
# 실행: streamlit run rv_streamlit_app.py

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (강제 설정)
import platform
import os

# macOS에서 한글 폰트 강제 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
else:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'NanumGothic', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ---------- 상수/단위 ----------
G_SI = 6.67430e-11                # [m^3 kg^-1 s^-2]
M_sun = 1.98847e30                # [kg]
M_jup = 1.89813e27                # [kg]
AU = 1.495978707e11               # [m]
c = 299792458.0                   # [m/s]

# 천문단위계(편의): a[AU], M[M_sun], P[year]
# 케플러 제3법칙: P[yr] = sqrt( a^3 / (M_total) ) (Mp, Ms 단위 Msun)
SEC_PER_YEAR = 365.25 * 24 * 3600

# ---------- 유틸 ----------
def mjup_to_msun(mj):
    return mj * (M_jup / M_sun)

def kepler_solve_E(M, e, tol=1e-10, max_iter=100):
    """평균근점이각 M → 편심근점이각 E (뉴턴-랩슨)"""
    # 초기값
    M = np.mod(M, 2*np.pi)
    # 이심률이 높을 때 더 안정적인 초기값 사용
    if e < 0.8:
        E = M
    elif e < 0.99:
        E = np.pi
    else:
        E = M + e*np.sin(M)  # 고이심률에서 더 나은 초기값
    
    for _ in range(max_iter):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        
        # 분모가 0에 가까우면 안전한 스텝 사용 (배열 처리)
        small_fp = np.abs(fp) < 1e-12
        dE = np.where(small_fp, 0.1 * np.sign(f), -f / fp)
            
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

def true_anomaly_from_M(M, e):
    """평균근점이각 → 진근점이각 f"""
    E = kepler_solve_E(M, e)
    cosf = (np.cos(E) - e) / (1 - e*np.cos(E))
    sinf = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e*np.cos(E))
    f = np.arctan2(sinf, cosf)
    return f

def orbital_period_year(a_AU, Ms_sun, Mp_sun):
    Mtot = Ms_sun + Mp_sun
    return np.sqrt((a_AU**3) / Mtot)

def rv_semiamplitude_K(a_AU, Ms_sun, Mp_sun, e, inc_rad):
    """정확한 K: K = (2πG/P)^{1/3} * Mp sin i / (Mtot)^{2/3} * 1/sqrt(1-e^2)"""
    # P [s]
    P_year = orbital_period_year(a_AU, Ms_sun, Mp_sun)
    P = P_year * SEC_PER_YEAR
    Mtot_kg = (Ms_sun + Mp_sun) * M_sun
    Mp_kg = Mp_sun * M_sun
    K = ((2*np.pi*G_SI)/P)**(1/3) * (Mp_kg*np.sin(inc_rad)) / (Mtot_kg**(2/3)) * 1/np.sqrt(1 - e**2)
    return K  # [m/s]

def rv_time_series(times_s, a_AU, Ms_sun, Mp_sun, e, w_rad, inc_rad, t_peri_s, gamma_ms=0.0):
    """표준 케플러 해석에 따른 RV(t) — 관측자 시선 양의 방향: 멀어짐(적색, +)"""
    # 공전 주기
    P_year = orbital_period_year(a_AU, Ms_sun, Mp_sun)
    P_s = P_year * SEC_PER_YEAR
    n = 2*np.pi / P_s  # 평균운동

    # 평균근점이각
    M = n * (times_s - t_peri_s)
    f = true_anomaly_from_M(M, e)  # 진근점이각

    K = rv_semiamplitude_K(a_AU, Ms_sun, Mp_sun, e, inc_rad)
    # 표준식: v_r = K[cos(w+f) + e cos w] + gamma
    vr = K * (np.cos(w_rad + f) + e*np.cos(w_rad)) + gamma_ms
    return vr, f, P_s

def relativistic_doppler_lambda(lambda0, vr):
    """상대론적 도플러 이동 (흡수선 중심 파장) — vr>0: 적색이동, vr<0: 청색이동"""
    beta = vr / c
    factor = np.sqrt((1+beta)/(1-beta))
    return lambda0 * factor

def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)

def instrument_broadening(line, R, lambda_grid):
    """분광기 분해능 R ~ lambda/Δlambda => 가우시안 FWHM ~ lambda/R
       표준편차 sigma ≈ FWHM / (2*sqrt(2ln2))"""
    if R <= 0:
        return line
    
    lam0 = np.median(lambda_grid)
    fwhm = lam0 / R
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    
    # 간단한 컨볼루션 (경계 효과 최소화를 위해 same 모드)
    # 커널 길이: ±5σ
    dlam = np.mean(np.diff(lambda_grid))
    half_width = int(np.ceil(5*sigma/dlam))
    
    # 커널 크기가 너무 작으면 원본 반환
    if half_width < 1:
        return line
        
    kx = np.arange(-half_width, half_width+1) * dlam
    kernel = gaussian(kx, 0.0, sigma)
    kernel /= kernel.sum()
    
    # 경계 처리 개선
    return np.convolve(line, kernel, mode='same')

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RV 외계행성 시뮬레이터", layout="wide")

st.title("중심별 시선속도(RV) 외계행성 탐사 — Streamlit 시뮬레이터")
st.caption("별-행성 질량중심 공전, RV 곡선, 흡수선 도플러 이동(청/적색편이) 수업용 데모")

# 세션 상태 (재생)
if "playing" not in st.session_state:
    st.session_state.playing = False
if "t0" not in st.session_state:
    st.session_state.t0 = time.time()
if "phase0" not in st.session_state:
    st.session_state.phase0 = 0.0  # [0..1)

# ----- 사이드바 입력 -----
with st.sidebar:
    st.header("궤도/질량 파라미터")
    Ms = st.number_input("별 질량 Ms [M☉]", min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.2f")
    Mp_mode = st.radio("행성 질량 단위", ["Mjup(목성질량)", "Msun(태양질량)"], horizontal=True)
    if Mp_mode == "Mjup(목성질량)":
        Mp_j = st.slider("행성 질량 Mp [Mjup]", 0.1, 20.0, 1.0, 0.1)
        Mp = mjup_to_msun(Mp_j)
    else:
        Mp = st.number_input("행성 질량 Mp [M☉]", min_value=1e-6, max_value=0.1, value=0.003, step=0.001, format="%.6f")

    a_AU = st.slider("공전 반지름 a [AU] (행성 궤도 장반경)", 0.02, 5.0, 1.0, 0.01)
    e = st.slider("이심률 e", 0.0, 0.95, 0.0, 0.01)
    inc_deg = st.slider("궤도 경사각 i [deg] (0: 정면, 90: 에지온)", 0.0, 90.0, 60.0, 1.0)
    w_deg = st.slider("근일점 방위각 ω [deg]", 0.0, 360.0, 0.0, 1.0)
    gamma = st.slider("계통속도 γ [m/s] (RV 오프셋)", -50000.0, 50000.0, 0.0, 10.0)

    st.divider()
    st.header("시간/재생 제어")
    speed = st.slider("재생 속도 배율 (x)", 0.1, 20.0, 2.0, 0.1,
                      help="시간 흐름 배율 — 큰 값일수록 빨리 돎")
    use_play = st.toggle("▶ 재생 모드 사용", value=False, help="체크하면 Play/Pause 버튼 활성화")
    colp1, colp2 = st.columns(2)
    if use_play:
        if colp1.button("▶ Play / ❚❚ Pause", use_container_width=True):
            st.session_state.playing = not st.session_state.playing
            if st.session_state.playing:
                st.session_state.t0 = time.time()
            else:
                # 일시정지 시 현재 phase를 고정
                pass
        if colp2.button("↺ 위상 초기화", use_container_width=True):
            st.session_state.phase0 = 0.0
            st.session_state.t0 = time.time()

    st.divider()
    st.header("스펙트럼 설정")
    lambda0 = st.number_input("흡수선 중심 파장 λ₀ [nm]", min_value=300.0, max_value=900.0, value=656.28, step=0.01,
                              help="예: Hα = 656.28 nm")
    R = st.slider("분해능 R(=λ/Δλ)", 1000, 150000, 50000, 500)
    line_depth = st.slider("선 심도 (0=선 없음, 1=바닥)", 0.0, 1.0, 0.5, 0.01)
    instr_sigma_pix = st.slider("내재적 선폭(가우시안 σ) [픽셀상수]", 0.5, 5.0, 1.5, 0.1,
                                help="기본 선폭(계기 전 컨볼루션용). 작을수록 날카로운 선")

    st.divider()
    st.header("표시/보조")
    n_orbit_pts = st.slider("궤도 궤적 점 개수", 100, 2000, 800, 50)
    show_bary = st.toggle("질량중심 표시", value=True)
    exaggerate_bary = st.slider("별 흔들림 시각 과장 배율", 1.0, 50.0, 5.0, 1.0)
    noise_amp = st.slider("스펙트럼 잡음(상대) ±", 0.0, 0.05, 0.0, 0.005)

# 라디안 변환
inc = np.deg2rad(inc_deg)
w = np.deg2rad(w_deg)

# 공전 주기와 시간축 준비
P_year = orbital_period_year(a_AU, Ms, Mp)
P_s = P_year * SEC_PER_YEAR

# 현재 시점(위상) 결정
if use_play and st.session_state.playing:
    elapsed = (time.time() - st.session_state.t0) * speed
    phase = (st.session_state.phase0 + (elapsed / P_s)) % 1.0
    # 위상 표시
    st.write(f"현재 위상: {phase:.3f}")
else:
    # 수동: 위상 슬라이더
    phase = st.slider("현재 위상 φ (0~1, φ=0이 근일점 통과)", 0.0, 1.0, 0.1, 0.001)

t_now = phase * P_s
t_peri = 0.0  # φ=0이 근일점 통과 시각

# 현재 RV/진근점이각
vr_now, f_now, _ = rv_time_series(np.array([t_now]), a_AU, Ms, Mp, e, w, inc, t_peri, gamma_ms=gamma)
vr_now = vr_now[0]
f_now = f_now[0]

# RV 곡선(한 주기) 샘플
N_ts = 800
ts = np.linspace(0, P_s, N_ts)
vr_curve, f_curve, _ = rv_time_series(ts, a_AU, Ms, Mp, e, w, inc, t_peri, gamma_ms=gamma)

# 궤도 좌표(질량중심 기준) — 3D에서 z축이 시선, xy 평면을 하늘면으로 가정
# 타원 궤도 방정식(초점 기준): r = a(1-e^2) / (1 + e cos f)
def orbit_xy(a_AU, e, f_array, inc, w):
    a_m = a_AU * AU
    r = a_m * (1 - e**2) / (1 + e*np.cos(f_array))
    # 궤도면 좌표
    x_orb = r * np.cos(f_array)
    y_orb = r * np.sin(f_array)
    # ω 회전
    cw, sw = np.cos(w), np.sin(w)
    x1 = cw*x_orb - sw*y_orb
    y1 = sw*x_orb + cw*y_orb
    # 경사 i 적용: y축을 경사 → 관측 평면 투영 (간단한 투영: y' = y*cos i)
    x_proj = x1
    y_proj = y1 * np.cos(inc)
    return x_proj, y_proj, r

f_grid = np.linspace(0, 2*np.pi, n_orbit_pts)
# 질량중심 위치비: a_star = a * (Mp/Mtot), a_planet = a * (Ms/Mtot)
Mtot = Ms + Mp
a_star_AU = a_AU * (Mp / Mtot)
a_plan_AU = a_AU * (Ms / Mtot)

xS, yS, rS = orbit_xy(a_star_AU, e, f_grid, inc, w)
xP, yP, rP = orbit_xy(a_plan_AU, e, f_grid, inc, w)

# 현재 위치 (f_now에서의 r)
xS_now, yS_now, _ = orbit_xy(a_star_AU, e, np.array([f_now]), inc, w)
xP_now, yP_now, _ = orbit_xy(a_plan_AU, e, np.array([f_now + np.pi]), inc, w)  # 행성은 반대 위상
xS_now, yS_now = xS_now[0], yS_now[0]
xP_now, yP_now = xP_now[0], yP_now[0]

# 시각 과장(별 흔들림 강조)
xS_plot = xS * exaggerate_bary
yS_plot = yS * exaggerate_bary
xS_now_plot = xS_now * exaggerate_bary
yS_now_plot = yS_now * exaggerate_bary

# 스펙트럼 축
# 파장 범위: 도플러 이동을 여유 있게 보기 위해 ±(vr_max/c)*λ0에 여분 추가
vr_max = np.max(np.abs(vr_curve - np.mean(vr_curve)))
margin = max(0.0005*lambda0, 5.0)  # nm, 최소 여유폭
lam_span = lambda0 * (vr_max/c) * 6 + margin
lam_min = lambda0 - lam_span
lam_max = lambda0 + lam_span
lam = np.linspace(lam_min, lam_max, 2000)

# 흡수선 모델 (연속 = 1.0, 중심선은 1 - depth*Gaussian)
lambda_now = relativistic_doppler_lambda(lambda0, vr_now)
# 기본 선폭: 픽셀상수로 제어 → 파장축 실제 폭으로 변환
pix = np.arange(lam.size)
pix_scale_nm = (lam_max - lam_min) / lam.size
sigma_nm = instr_sigma_pix * pix_scale_nm

# 연속 스펙트럼 (무지개색 배경)
continuum = np.ones_like(lam)
# 흡수선 (검은색으로 표시하기 위해 1에서 빼기)
absorption_line = line_depth * gaussian(lam, lambda_now, sigma_nm)
line = continuum - absorption_line

# 분광기 컨볼루션
line_conv = instrument_broadening(line, R, lam)
# 잡음 추가(선택)
if noise_amp > 0:
    rng = np.random.default_rng(1234)
    line_conv = np.clip(line_conv + rng.uniform(-noise_amp, noise_amp, size=line_conv.size), 0, 1.2)

# ---------- 레이아웃 ----------
col1, col2, col3 = st.columns([1.1, 1.1, 1.2])

# (1) 궤도 플롯
with col1:
    st.subheader("질량중심 기준 궤도 투영")
    fig1, ax1 = plt.subplots(figsize=(4.8, 4.8))
    ax1.plot(xP/AU, yP/AU, lw=1.5, label="행성 궤도 (AU)")
    ax1.plot(xS_plot/AU, yS_plot/AU, lw=1.5, label=f"별 궤도×{int(exaggerate_bary)} (과장, AU)")
    ax1.scatter([xP_now/AU], [yP_now/AU], s=60, label="행성(현재)", zorder=5)
    ax1.scatter([xS_now_plot/AU], [yS_now_plot/AU], s=60, marker="*", label="별(현재, 과장)", zorder=6)
    if show_bary:
        ax1.scatter([0],[0], c="k", s=20, label="질량중심(=0)")
    ax1.set_xlabel("X [AU]")
    ax1.set_ylabel("Y [AU] (투영)")
    ax1.axis("equal")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(loc="best", fontsize=9)
    st.pyplot(fig1)

# (2) RV 곡선
with col2:
    st.subheader("시선속도 곡선 v_r(t)")
    fig2, ax2 = plt.subplots(figsize=(5.2, 4.0))
    t_days = ts / 86400.0
    ax2.plot(t_days, vr_curve, lw=1.5)
    # 현재 시각 표시
    ax2.axvline(t_now/86400.0, ls="--", alpha=0.6)
    # 0선
    ax2.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax2.set_xlabel("시간 [일] (한 주기)")
    ax2.set_ylabel("시선속도 v_r [m/s]\n(+ 적색편이: 멀어짐, − 청색편이: 다가옴)")
    ax2.grid(True, ls="--", alpha=0.4)
    txt = (f"P = {P_year:.3f} yr = {P_year*365.25:.1f} d\n"
           f"K ≈ {rv_semiamplitude_K(a_AU, Ms, Mp, e, inc):.1f} m/s\n"
           f"v_r(now) = {vr_now:.1f} m/s")
    ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, lw=0.5))
    st.pyplot(fig2)

# (3) 스펙트럼
with col3:
    st.subheader("흡수선 스펙트럼 (도플러 이동)")
    fig3, ax3 = plt.subplots(figsize=(5.6, 4.0))
    
    # 무지개색 배경 생성
    lam_norm = (lam - lam_min) / (lam_max - lam_min)  # 0~1 정규화
    
    # 연속 스펙트럼을 무지개색으로 표시 (더 명확하게)
    step = max(1, len(lam) // 50)  # 50개 구간으로 나누기
    for i in range(0, len(lam)-step, step):
        color = plt.cm.rainbow(lam_norm[i])
        ax3.plot(lam[i:i+step+1], line_conv[i:i+step+1], 
                color=color, lw=4, alpha=0.8)
    
    # 흡수선 강조 (검은색, 매우 굵게)
    ax3.plot(lam, line_conv, 'k-', lw=3, alpha=1.0, label="흡수선")
    
    # 배경을 더 밝게 하기 위해 연속 스펙트럼을 다시 그리기
    ax3.fill_between(lam, line_conv, 1.0, alpha=0.3, color='white')
    
    # 기준선들
    ax3.axvline(lambda0, ls="--", color='gray', alpha=0.7, label="정지 파장 λ₀")
    ax3.axvline(lambda_now, ls="--", color='red', alpha=0.9, lw=2, label="현재 중심 파장")
    
    ax3.set_xlabel("파장 [nm]")
    ax3.set_ylabel("상대광도 (정규화)")
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, ls="--", alpha=0.4)
    
    # 청/적 판별 텍스트
    shift_nm = lambda_now - lambda0
    if vr_now > 0:
        tag = "🔴 적색편이 (멀어짐, +v_r)"
        color_tag = "red"
    elif vr_now < 0:
        tag = "🔵 청색편이 (다가옴, −v_r)"
        color_tag = "blue"
    else:
        tag = "⚪ 편이 없음 (v_r=0)"
        color_tag = "gray"
    
    ax3.legend(loc="best", fontsize=9)
    ax3.text(0.02, 0.98,
             f"λ_now − λ₀ = {shift_nm:+.4f} nm\n{tag}",
             transform=ax3.transAxes, va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round", facecolor=color_tag, alpha=0.2, lw=1))
    st.pyplot(fig3)

# 재생 모드일 때 주기적 갱신
if use_play and st.session_state.playing:
    # 매 프레임 경과 후 즉시 재실행하여 애니메이션 효과
    time.sleep(0.1)  # 적절한 갱신 주기
    st.rerun()