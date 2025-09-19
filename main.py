# main.py — PyScript app with animation
import asyncio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyscript import display
from pyodide.ffi import create_proxy
from js import document

from rv_model import RVModel, DAY, C

# Figures
rv_fig, rv_ax = plt.subplots(figsize=(6,3), dpi=120)
rv_ax.set_title("Radial Velocity (m/s)")
rv_ax.set_xlabel("Time (days)"); rv_ax.set_ylabel("RV (m/s)")
rv_ax.grid(True)

spec_fig, spec_ax = plt.subplots(figsize=(6,3), dpi=120)
spec_ax.set_title("Absorption line (Doppler shift)")
spec_ax.set_xlabel("Wavelength (nm)"); spec_ax.set_ylabel("Intensity (arb.)")
spec_ax.grid(True)

orbit_fig, orbit_ax = plt.subplots(figsize=(6,3), dpi=120)
orbit_ax.set_title("Orbit (sky plane view)")
orbit_ax.set_aspect('equal')
orbit_ax.grid(True)

display(rv_fig, target="rv_plot", append=False)
display(spec_fig, target="spectrum_plot", append=False)
display(orbit_fig, target="orbit_plot", append=False)

# Model and data
model = RVModel()
N = 500
t_grid = np.linspace(0, model.P, N)
rv_vals = np.array([model.rv(tt) for tt in t_grid])
(rv_line,) = rv_ax.plot(t_grid/DAY, rv_vals, lw=2, label="RV")
(rv_marker,) = rv_ax.plot([0],[rv_vals[0]], "o", ms=6, label="current")
rv_ax.legend(loc="upper right")

# spectrum (simple gaussian absorption centered at shifted lambda)
lam_span = 6.0  # ± range around lambda0
lam_grid = np.linspace(model.lam0 - lam_span, model.lam0 + lam_span, 600)
def spectrum(lam_center, depth=0.6, sigma=0.2):
    prof = 1.0 - depth * np.exp(-0.5*((lam_grid - lam_center)/sigma)**2)
    return prof
spec_line, = spec_ax.plot(lam_grid, spectrum(model.lam0))
(spec_marker,) = spec_ax.plot([model.lam0],[0.4],"v",ms=8)

# orbit
xs, ys, xp, yp = model.orbit_xy(t_grid, scale=1.0)
(orbit_star_path,)   = orbit_ax.plot(xs, ys, lw=2, label="Star path")
(orbit_planet_path,) = orbit_ax.plot(xp, yp, lw=2, label="Planet path")
(orbit_star_marker,) = orbit_ax.plot([xs[0]],[ys[0]], "o", ms=6, label="Star")
(orbit_planet_marker,) = orbit_ax.plot([xp[0]],[yp[0]], "o", ms=6, label="Planet")
orbit_ax.plot(0,0,"kx",ms=8,label="Barycenter")
orbit_ax.legend(loc="upper right")

def get(id_): return document.getElementById(id_)
def rf(id_): return float(get(id_).value)
def set_txt(id_, txt): get(id_).innerText = str(txt)

def update_labels(evt=None):
    set_txt("Ms_val", f"{rf('Ms'):.2f}")
    set_txt("Mp_val", f"{rf('Mp'):.2f}")
    set_txt("inc_val", f"{rf('inc'):.0f}")
    set_txt("ecc_val", f"{rf('ecc'):.2f}")
    set_txt("omega_val", f"{rf('omega'):.0f}")
    set_txt("period_val", f"{rf('period'):.1f}")
    set_txt("lambda0_val", f"{rf('lambda0'):.0f}")
    set_txt("gamma_val", f"{rf('gamma'):.0f}")
    set_txt("speed_val", f"{rf('speed'):.1f}")

for _id in ["Ms","Mp","inc","ecc","omega","period","lambda0","gamma","speed"]:
    get(_id).addEventListener("input", create_proxy(update_labels))

def apply_params(evt=None):
    global model, t_grid, rv_vals, xs, ys, xp, yp, lam_grid
    model.set_params(
        Ms_solar=rf("Ms"), Mp_jup=rf("Mp"), inc_deg=rf("inc"),
        period_days=rf("period"), ecc=rf("ecc"), omega_deg=rf("omega"),
        t0_days=0.0, gamma_ms=rf("gamma"), base_lambda_nm=rf("lambda0")
    )
    t_grid = np.linspace(0, model.P, N)
    rv_vals = np.array([model.rv(tt) for tt in t_grid])
    rv_line.set_data(t_grid/DAY, rv_vals)
    rv_ax.relim(); rv_ax.autoscale_view()

    xs, ys, xp, yp = model.orbit_xy(t_grid, scale=1.0)
    orbit_star_path.set_data(xs, ys)
    orbit_planet_path.set_data(xp, yp)
    orbit_ax.relim(); orbit_ax.autoscale_view()

    # spectrum x-axis recenters to lam0
    lam_span = 6.0
    lam0 = model.lam0
    lam_grid[:] = np.linspace(lam0 - lam_span, lam0 + lam_span, lam_grid.size)
    spec_line.set_data(lam_grid, spectrum(lam0))

    rv_fig.canvas.draw(); orbit_fig.canvas.draw(); spec_fig.canvas.draw()

get("btn_apply").addEventListener("click", create_proxy(apply_params))

# animation state
running = False
t_elapsed = 0.0  # seconds
frame_dt = 0.05  # app tick ~50ms

async def animate():
    global t_elapsed
    while running:
        speed = rf("speed")
        t_elapsed = (t_elapsed + frame_dt*speed*DAY) % model.P
        # find nearest index
        idx = int((t_elapsed / model.P) * (N-1))

        # RV marker
        rv_marker.set_data([t_elapsed/DAY], [rv_vals[idx]])

        # Spectrum: shift line by current RV
        v = rv_vals[idx]
        lam_shift = model.doppler_shift_nm(v)
        spec_line.set_data(lam_grid, spectrum(lam_shift))
        spec_marker.set_data([lam_shift], [0.4])

        # Orbit markers
        orbit_star_marker.set_data([xs[idx]], [ys[idx]])
        orbit_planet_marker.set_data([xp[idx]], [yp[idx]])

        rv_fig.canvas.draw()
        spec_fig.canvas.draw()
        orbit_fig.canvas.draw()
        await asyncio.sleep(frame_dt)

def play(evt=None):
    global running
    if not running:
        running = True
        asyncio.create_task(animate())

def pause(evt=None):
    global running
    running = False

get("btn_play").addEventListener("click", create_proxy(play))
get("btn_pause").addEventListener("click", create_proxy(pause))

# initial draw
update_labels(); apply_params()
