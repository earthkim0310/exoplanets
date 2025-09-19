#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV (Doppler) Exoplanet Simulator — LOS projection (with Kepler lock + UI patches + Korean font fix)
=================================================
- Left: RV curve (sliding window)
- Middle: Spectrum schematic (rainbow + single vertical absorption line)
- Right: Orbit view (지구 시선 기준 투영)
"""

import matplotlib
matplotlib.use("TkAgg")

# [추가] 한글 폰트 설정
from matplotlib import rcParams, font_manager

def setup_korean_font():
    candidates = [
        "AppleGothic",         # macOS 기본
        "NanumGothic",         # 나눔고딕
        "Noto Sans CJK KR",    # 구글 노토
        "Noto Sans KR",        # 구글 노토 KR
        "Malgun Gothic"        # Windows 맑은 고딕
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            chosen = name
            break
    rcParams["axes.unicode_minus"] = False
    if chosen is None:
        print("[Warn] 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있어요.")
    else:
        print(f"[Info] Matplotlib font set to: {chosen}")

setup_korean_font()

import math
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass


# Physical constants
G     = 6.67430e-11
M_SUN = 1.98847e30
M_JUP = 1.89813e27
DAY   = 86400.0
AU    = 1.495978707e11
C     = 299_792_458.0

# Visual exaggeration for spectral shift (display only)
DOPPLER_SCALE = 1e4

@dataclass
class Params:
    M_star_solar: float = 1.0
    M_p_jup: float = 1.0
    inc_deg: float = 90.0
    period_days: float = 365.25
    ecc: float = 0.0
    omega_deg: float = 0.0
    t0_days: float = 0.0
    gamma_ms: float = 0.0
    base_lambda_nm: float = 656.28
    # 추가
    a_au: float = 1.0
    kepler_lock: bool = True

class RvModel:
    def __init__(self, p: Params):
        self.update_params(p)

    def update_params(self, p: Params):
        # sanitize
        Ms_solar = max(1e-12, float(p.M_star_solar))
        Mp_jup   = max(0.0,    float(p.M_p_jup))
        inc_deg  = float(p.inc_deg)
        period_d = max(1e-9,   float(p.period_days))
        ecc_in   = float(p.ecc)
        ecc_clamped = min(max(ecc_in, 0.0), 0.999999)

        # 저장
        self.p = Params(
            M_star_solar=Ms_solar, M_p_jup=Mp_jup, inc_deg=inc_deg,
            period_days=period_d, ecc=ecc_clamped, omega_deg=p.omega_deg,
            t0_days=p.t0_days, gamma_ms=p.gamma_ms, base_lambda_nm=p.base_lambda_nm,
            a_au=max(1e-6, float(getattr(p, "a_au", 1.0))),
            kepler_lock=bool(getattr(p, "kepler_lock", True))
        )

        # 단위/내부 상태
        self.Ms = Ms_solar * M_SUN
        self.Mp = Mp_jup   * M_JUP
        self.i  = math.radians(inc_deg)
        self.e  = ecc_clamped
        self.omega = math.radians(p.omega_deg)  # [deg]→[rad]
        self.t0 = p.t0_days * DAY
        self.gamma = p.gamma_ms
        self.lam0  = p.base_lambda_nm

        # --- Kepler lock ---
        if self.p.kepler_lock:
            # a → P
            self.a = self.p.a_au * AU
            self.P = 2 * math.pi * math.sqrt(self.a**3 / (G * (self.Ms + self.Mp)))
            self.p.period_days = self.P / DAY
        else:
            # P → a
            self.P = period_d * DAY
            self.a = (G * (self.Ms + self.Mp) * self.P**2 / (4*math.pi**2)) ** (1/3.0)
            self.p.a_au = self.a / AU

        # 질량비에 따른 별/행성 궤도 반지름(바리센터 기준)
        self.a_star   = self.a * self.Mp / (self.Ms + self.Mp)
        self.a_planet = self.a * self.Ms / (self.Ms + self.Mp)

        # 참고용 K(세미진폭) 스케일
        denom = max(1e-16, 1.0 - self.e**2)
        self.K = (((2 * math.pi * G) / self.P) ** (1/3.0)
                  * (self.Mp * math.sin(self.i))
                  / ((self.Ms + self.Mp) ** (2/3.0))
                  / math.sqrt(denom))

    def mean_anomaly(self, t):
        return -2*math.pi * ((t - self.t0) % self.P) / self.P

    def eccentric_anomaly(self, M):
        E = M if self.e < 0.8 else math.pi
        for _ in range(50):
            f  = E - self.e*math.sin(E) - M
            fp = 1 - self.e*math.cos(E)
            dE = -f/fp
            E += dE
            if abs(dE) < 1e-12:
                break
        return E

    def true_anomaly(self, E):
        factor = math.sqrt((1 + self.e) / (1 - self.e))
        return 2 * math.atan2(factor * math.sin(E/2), math.cos(E/2))

    def radii_visual(self, nu: float, bary_boost: float, preserve_ecc: bool):
        """
        가시적 질량비 과장을 위한 반지름(표시용).
        preserve_ecc=True이면 r(ν)=a(1-e^2)/(1+e cosν)를 반영.
        """
        mratio_true = self.Mp / self.Ms
        mratio_vis  = max(1e-16, mratio_true * max(1.0, float(bary_boost)))
        a_star_vis   = self.a * (mratio_vis / (1.0 + mratio_vis))
        a_planet_vis = self.a * (1.0 / (1.0 + mratio_vis))
        if preserve_ecc:
            fac = (1.0 - self.e**2) / (1.0 + self.e * math.cos(nu))
            a_star_vis   *= fac
            a_planet_vis *= fac
        return a_star_vis, a_planet_vis

    def star_planet_positions(self, t, scale_factor=2000.0, schematic=False, bary_boost: float = 1.0):
        """
        base 좌표(회전 전): x_LOS = r cosθ sin i, y_sky = r sinθ  with θ = ν + ω
        화면 표시용 회전: (x_plot, y_plot) = (y_sky, +x_LOS)  ← (위상 정렬 패치)
        """
        M  = self.mean_anomaly(t)
        E  = self.eccentric_anomaly(M)
        nu = self.true_anomaly(E)
        theta = nu + self.omega  # 위치에도 ω 반영

        if schematic:
            # 원 궤도 모식도(반지름 고정). ν의존성 제거
            r_star, r_planet = self.radii_visual(nu=0.0, bary_boost=bary_boost, preserve_ecc=False)
        else:
            # 타원일 때 r(ν) 반영
            r_star, r_planet = self.radii_visual(nu=nu, bary_boost=bary_boost, preserve_ecc=True)

        inc = self.i  # radians

        # base (LOS, sky) — θ 사용
        x_s = -r_star   * math.cos(theta) * math.sin(inc) * scale_factor
        y_s = -r_star   * math.sin(theta) * scale_factor
        x_p =  r_planet * math.cos(theta) * math.sin(inc) * scale_factor
        y_p =  r_planet * math.sin(theta) * scale_factor

        # rotate for plotting: (x_plot, y_plot) = (y_sky, +LOS)  ← 변경
        xs_plot, ys_plot = y_s,  x_s
        xp_plot, yp_plot = y_p,  x_p
        return (xs_plot, ys_plot), (xp_plot, yp_plot)

    def rv(self, t):
        """
        별의 시선속도 v_star_los [m/s] — 그림 위상과 동기화
        - 합/충(정렬)에서 0, 사분점에서 ±최대
        """
        # 궤도 위상
        M  = self.mean_anomaly(t)
        E  = self.eccentric_anomaly(M)
        nu = self.true_anomaly(E)
        theta = nu + self.omega  # 퍼리아스트론 회전 포함

        # 궤도 상수
        mu = G * (self.Ms + self.Mp)                    # 표준 중력파라미터
        h  = math.sqrt(mu * self.a * (1.0 - self.e**2)) # 각운동량 크기
        v_scale = mu / h                                # 속도 스케일 (mu/h)

        # 상대속도(perifocal) 방향을 LOS로 투영한 성분 ~ -sin(θ) * v_scale
        v_rel_los = -math.sin(theta) * v_scale

        # 별의 속도 = -(Mp/(Ms+Mp)) * v_rel * sin(i)  (LOS 투영)
        v_star_los = - (self.Mp / (self.Ms + self.Mp)) * v_rel_los * math.sin(self.i)

        return v_star_los + self.gamma

class App:
    def __init__(self, root):
        self.root = root
        root.title("RV Exoplanet Simulator (Doppler)")
        self.params = Params()
        self.model  = RvModel(self.params)

        self.t_sim   = 0.0
        self.dt      = 5.0 * DAY
        self.running = False

        self.t_list, self.rv_list = [], []
        self.entries = {}

        self.scale_factor    = 2000.0
        self.rv_window_days  = 400.0
        self.schematic_var   = tk.BooleanVar(value=True)
        self.bary_boost      = tk.DoubleVar(value=80.0)
        self.kepler_lock_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._init_plots()

    def _build_ui(self):
        main = ttk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        right= ttk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        controls = [
            ("M★ (Msun)",   "M_star_solar",   self.params.M_star_solar),
            ("Mp (M_jup)",  "M_p_jup",        self.params.M_p_jup),
            ("i (deg)",     "inc_deg",        self.params.inc_deg),
            ("P (days)",    "period_days",    self.params.period_days),
            ("a (AU)",      "a_au",           self.params.a_au),
            ("e",           "ecc",            self.params.ecc),
            ("ω (deg)",     "omega_deg",      self.params.omega_deg),
            ("t0 (days)",   "t0_days",        self.params.t0_days),
            ("γ (m/s)",     "gamma_ms",       self.params.gamma_ms),
            ("λ₀ (nm)",     "base_lambda_nm", self.params.base_lambda_nm),
        ]
        for label, key, default in controls:
            row = ttk.Frame(left); row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=default)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)
            self.entries[key] = var

        # Kepler lock 체크박스
        row = ttk.Frame(left); row.pack(fill=tk.X, pady=(2,6))
        ttk.Checkbutton(row, text="Kepler lock (a·질량→P 자동결정)",
                        variable=self.kepler_lock_var).pack(side=tk.LEFT)

        row = ttk.Frame(left); row.pack(fill=tk.X, pady=(6,2))
        ttk.Checkbutton(row, text="모식도(원 궤도)", variable=self.schematic_var,
                        command=self.reset).pack(side=tk.LEFT)
        ttk.Label(left, text="Barycenter 강조(×MassRatio)").pack(anchor='w')
        ttk.Scale(left, from_=1.0, to=200.0, variable=self.bary_boost,
                  orient='horizontal').pack(fill=tk.X)

        ttk.Button(left, text="적용(Apply)", command=self.apply_params).pack(fill=tk.X, pady=3)
        self.btn_toggle = ttk.Button(left, text="▶ 재생", command=self.toggle_run)
        self.btn_toggle.pack(fill=tk.X, pady=3)
        ttk.Button(left, text="한 스텝", command=self.step_once).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="초기화", command=self.reset).pack(fill=tk.X, pady=3)

        self.fig = Figure(figsize=(16,4), dpi=100)
        self.ax_rv    = self.fig.add_subplot(131)
        self.ax_spec  = self.fig.add_subplot(132)
        self.ax_orbit = self.fig.add_subplot(133)
        self.ax_orbit.set_aspect('equal')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_plots(self):
        # RV
        self.ax_rv.set_xlabel("Time (days)")
        self.ax_rv.set_ylabel("RV (m/s)")
        self.ax_rv.grid(True)
        self.ax_rv.xaxis.set_major_locator(MaxNLocator(nbins=6))
        (self.rv_line,) = self.ax_rv.plot([], [], 'r-')
        self.ax_rv.set_xlim(0, self.rv_window_days)

        # Spectrum
        lam0 = self.params.base_lambda_nm
        self.ax_spec.set_xlabel("Wavelength (nm)")
        self.ax_spec.set_ylabel("Intensity (schematic)")
        self.ax_spec.set_ylim(0, 1)
        self.ax_spec.set_xlim(lam0-1, lam0+1)
        self.ax_spec.get_yaxis().set_visible(False)
        # rainbow background
        xres = 800
        gradient = np.linspace(0, 1, xres)[None, :]
        self.spec_bg = self.ax_spec.imshow(
            np.vstack([gradient]*20), extent=[lam0-1, lam0+1, 0, 1],
            origin='lower', aspect='auto', cmap='turbo')
        self.spec_center = self.ax_spec.axvline(lam0, color='k', lw=1, ls='--')
        self.spec_line   = self.ax_spec.axvline(lam0, color='k', lw=3)

        # RV/λ 라벨
        self.spec_text = self.ax_spec.text(0.02, 0.95, "", transform=self.ax_spec.transAxes,
                                           va='top', ha='left')
        # 청/적 안내 텍스트
        self.ax_spec.text(0.02, -0.12, "청색편이 ←", transform=self.ax_spec.transAxes,
                          ha='left', va='top')
        self.ax_spec.text(0.98, -0.12, "→ 적색편이", transform=self.ax_spec.transAxes,
                          ha='right', va='top')

        # [추가] 눈금은 유지, 라벨은 시작·중간·끝만
        from matplotlib.ticker import MultipleLocator, FuncFormatter
        self.ax_spec.xaxis.set_major_locator(MultipleLocator(0.5))  # 눈금 간격 유지
        def _three_labels(x, pos):
            lam0c = self.model.lam0
            if abs(x - (lam0c - 1)) < 1e-9 or abs(x - lam0c) < 1e-9 or abs(x - (lam0c + 1)) < 1e-9:
                return f"{x:.2f}"
            return ""
        self.ax_spec.xaxis.set_major_formatter(FuncFormatter(_three_labels))

        # Orbit
        self._draw_orbit_paths()
        self.canvas.draw()

    def _draw_orbit_paths(self):
        self.ax_orbit.cla()
        self.ax_orbit.set_facecolor('white')
        # 회전된 표시 좌표에 맞게 라벨
        self.ax_orbit.set_xlabel("x (sky proj)")
        self.ax_orbit.set_ylabel("y (downward LOS)")
        self.ax_orbit.grid(True, zorder=0)
        self.ax_orbit.set_aspect('equal')

        # 샘플 각도: nu(진근점이각) 배열
        nu_arr = np.linspace(0, 2*math.pi, 400)
        theta_arr = nu_arr + self.model.omega  # 경로에도 ω 적용
        boost = float(self.bary_boost.get())
        i     = self.model.i

        # visual mass ratio (display only)
        mratio_true = self.model.Mp / self.model.Ms
        mratio_vis  = max(1e-16, mratio_true * max(1.0, boost))
        a_star_vis   = self.model.a * (mratio_vis / (1.0 + mratio_vis))
        a_planet_vis = self.model.a * (1.0 / (1.0 + mratio_vis))

        if self.schematic_var.get():
            r_s = a_star_vis   * self.scale_factor
            r_p = a_planet_vis * self.scale_factor
        else:
            fac = (1.0 - self.model.e**2) / (1.0 + self.model.e*np.cos(nu_arr))  # r(ν)
            r_s = a_star_vis   * fac * self.scale_factor
            r_p = a_planet_vis * fac * self.scale_factor

        # base: x_LOS, y_sky  — θ 사용
        xsb = -r_s * np.cos(theta_arr) * np.sin(i)
        ysb = -r_s * np.sin(theta_arr)
        xpb =  r_p * np.cos(theta_arr) * np.sin(i)
        ypb =  r_p * np.sin(theta_arr)

        # rotate for plotting: (x_plot, y_plot) = (y_sky, +LOS)  ← 변경
        xs, ys = ysb,  xsb
        xp, yp = ypb,  xpb

        (self.star_path,)   = self.ax_orbit.plot(xs, ys, color='orange',      lw=2.5, label='Star orbit')
        (self.planet_path,) = self.ax_orbit.plot(xp, yp, color='saddlebrown', lw=2.5, label='Planet orbit')
        self.ax_orbit.plot(0,0,'kx',markersize=10,label='Barycenter')
        self.ax_orbit.legend(loc='upper right')

        Rmax   = float(max(np.max(np.hypot(xp, yp)), np.max(np.hypot(xs, ys))))
        margin = 0.2 * Rmax

        # Earth 아래(−y) 배치 + 연결선
        self.earth_x = 0.0
        self.earth_y = -1.5 * Rmax
        self.ax_orbit.arrow(0, 0, 0, self.earth_y, head_width=0.08*Rmax, color='green', zorder=2)
        self.ax_orbit.scatter([self.earth_x], [self.earth_y], marker='o', s=120,
                              color='tab:blue', edgecolor='k', zorder=5)
        self.ax_orbit.text(self.earth_x, self.earth_y - 0.12*Rmax, 'Earth',
                           color='blue', ha='center', va='top', zorder=5)

        self.ax_orbit.set_xlim(-Rmax - margin,  Rmax + margin)
        self.ax_orbit.set_ylim(self.earth_y - margin,  Rmax + margin)

        # moving markers (현재 위치; star_planet_positions는 회전된 좌표 반환)
        (self.star_pt,)   = self.ax_orbit.plot([], [], 'o', color='gold',     markersize=7)
        (self.planet_pt,) = self.ax_orbit.plot([], [], 'o', color='tab:blue', markersize=5)

    def _recompute_orbit_paths(self):
        self._draw_orbit_paths()

    def apply_params(self):
        p = Params(
            M_star_solar = self.entries['M_star_solar'].get(),
            M_p_jup      = self.entries['M_p_jup'].get(),
            inc_deg      = self.entries['inc_deg'].get(),
            period_days  = self.entries['period_days'].get(),
            ecc          = self.entries['ecc'].get(),
            omega_deg    = self.entries['omega_deg'].get(),
            t0_days      = self.entries['t0_days'].get(),
            gamma_ms     = self.entries['gamma_ms'].get(),
            base_lambda_nm = self.entries['base_lambda_nm'].get(),
            a_au         = self.entries['a_au'].get(),
            kepler_lock  = self.kepler_lock_var.get(),
        )
        self.params = p
        self.model.update_params(p)

        # UI 동기화: lock ON이면 계산된 P를, OFF면 계산된 a를 반영
        if self.kepler_lock_var.get():
            self.entries['period_days'].set(self.model.p.period_days)
        self.entries['a_au'].set(self.model.p.a_au)

        self.reset()

    def toggle_run(self):
        self.running = not self.running
        self.btn_toggle.config(text="⏸ 정지" if self.running else "▶ 재생")
        if self.running:
            self._tick()

    def step_once(self):
        self.t_sim += self.dt
        self._update_plots()

    def reset(self):
        self.t_sim = 0.0
        self.t_list.clear(); self.rv_list.clear()
        self.rv_line.set_data([], [])
        lam0 = self.model.lam0
        self.spec_bg.set_extent([lam0-1, lam0+1, 0, 1])
        self.ax_spec.set_xlim(lam0-1, lam0+1)
        self.spec_center.set_xdata([lam0, lam0])
        self.spec_line.set_xdata([lam0, lam0])
        self.spec_text.set_text("")
        self._recompute_orbit_paths()
        self.ax_rv.set_xlim(0, self.rv_window_days)
        self.canvas.draw()

    def _tick(self):
        if not self.running:
            return
        self.t_sim += self.dt
        self._update_plots()
        self.root.after(100, self._tick)

    def _update_plots(self):
        # --- RV ---
        t_days = self.t_sim / DAY
        rv     = self.model.rv(self.t_sim)
        self.t_list.append(t_days); self.rv_list.append(rv)
        self.rv_line.set_data(self.t_list, self.rv_list)
        if t_days > self.rv_window_days:
            self.ax_rv.set_xlim(t_days - self.rv_window_days, t_days)
        self.ax_rv.relim(); self.ax_rv.autoscale_view(scaley=True, scalex=False)

        # --- Spectrum (single absorption line; shifted)
        lam0 = self.model.lam0
        lam_shift = lam0 * (1 + (rv/C) * DOPPLER_SCALE)  # 표준: rv>0(멀어짐)→적색편이
        self.spec_line.set_xdata([lam_shift, lam_shift])
        self.spec_text.set_text(f"RV = {rv: .1f} m/s\nλ = {lam_shift:.3f} nm  (λ0={lam0:.3f})")

        # --- Orbit (current markers using rotated plotting coords)
        (xs, ys), (xp, yp) = self.model.star_planet_positions(
            self.t_sim,
            self.scale_factor,
            self.schematic_var.get(),
            bary_boost=float(self.bary_boost.get())
        )
        self.star_pt.set_data([xs], [ys])
        self.planet_pt.set_data([xp], [yp])

        self.canvas.draw()


def main():
    root = tk.Tk()
    app  = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()