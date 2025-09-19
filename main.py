# main.py — PyScript glue: HTML inputs → matplotlib plots
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from pyodide.ffi import create_proxy
from js import document
from rv_model import RVModel, DAY

# Create two figures (RV + Orbit)
rv_fig, rv_ax = plt.subplots(figsize=(6,3), dpi=120)
rv_ax.set_title("Radial Velocity (m/s)")
rv_ax.set_xlabel("Time (days)"); rv_ax.set_ylabel("RV (m/s)")
rv_ax.grid(True)

orbit_fig, orbit_ax = plt.subplots(figsize=(6,3), dpi=120)
orbit_ax.set_title("Orbit (sky plane view)")
orbit_ax.set_aspect('equal')
orbit_ax.grid(True)

# Add canvases to DOM
from pyscript import display
display(rv_fig, target="rv_plot", append=False)
display(orbit_fig, target="orbit_plot", append=False)

# Initial model
model = RVModel()
t = np.linspace(0, model.P, 400)
(rv_line,) = rv_ax.plot(t/DAY, [model.rv(tt) for tt in t], lw=2)

xs, ys, xp, yp = model.orbit_xy(np.linspace(0, model.P, 400), scale=1.0)
(orbit_star_path,)   = orbit_ax.plot(xs, ys, lw=2, label="Star orbit")
(orbit_planet_path,) = orbit_ax.plot(xp, yp, lw=2, label="Planet orbit")
orbit_ax.plot(0,0,"kx",ms=8,label="Barycenter")
orbit_ax.legend(loc="upper right")

def read_float(id_):
    return float(document.getElementById(id_).value)

def set_text(id_, txt):
    document.getElementById(id_).innerText = str(txt)

def on_input_update(evt):
    # live value labels
    set_text("Ms_val", f"{read_float('Ms'):.2f}")
    set_text("Mp_val", f"{read_float('Mp'):.2f}")
    set_text("inc_val", f"{read_float('inc'):.0f}")
    set_text("ecc_val", f"{read_float('ecc'):.2f}")
    set_text("omega_val", f"{read_float('omega'):.0f}")
    set_text("period_val", f"{read_float('period'):.1f}")
    set_text("lambda0_val", f"{read_float('lambda0'):.0f}")
    set_text("gamma_val", f"{read_float('gamma'):.0f}")

for _id in ["Ms","Mp","inc","ecc","omega","period","lambda0","gamma"]:
    document.getElementById(_id).addEventListener("input", create_proxy(on_input_update))

def apply_params(evt=None):
    global model, t
    model = RVModel(
        Ms_solar = read_float("Ms"),
        Mp_jup   = read_float("Mp"),
        inc_deg  = read_float("inc"),
        period_days = read_float("period"),
        ecc = read_float("ecc"),
        omega_deg = read_float("omega"),
        gamma_ms  = read_float("gamma"),
        base_lambda_nm = read_float("lambda0"),
    )
    t = np.linspace(0, model.P, 400)
    rv_line.set_data(t/DAY, [model.rv(tt) for tt in t])
    rv_ax.relim(); rv_ax.autoscale_view()

    xs, ys, xp, yp = model.orbit_xy(np.linspace(0, model.P, 400), scale=1.0)
    orbit_star_path.set_data(xs, ys)
    orbit_planet_path.set_data(xp, yp)
    orbit_ax.relim(); orbit_ax.autoscale_view()
    orbit_fig.canvas.draw()
    rv_fig.canvas.draw()

document.getElementById("btn_apply").addEventListener("click", create_proxy(apply_params))
# initial label update
on_input_update(None)
apply_params(None)
