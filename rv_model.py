# rv_model.py — pure Python model (no GUI)
import math
import numpy as np

G     = 6.67430e-11
M_SUN = 1.98847e30
M_JUP = 1.89813e27
DAY   = 86400.0
AU    = 1.495978707e11
C     = 299_792_458.0

class RVModel:
    def __init__(self, Ms_solar=1.0, Mp_jup=1.0, inc_deg=60.0,
                 period_days=10.0, ecc=0.0, omega_deg=0.0,
                 t0_days=0.0, gamma_ms=0.0, base_lambda_nm=656.0):
        self.Ms = Ms_solar * M_SUN
        self.Mp = Mp_jup   * M_JUP
        self.i  = math.radians(inc_deg)
        self.P  = period_days * DAY
        self.a  = (G * (self.Ms + self.Mp) * self.P**2 / (4*math.pi**2)) ** (1/3.0)
        self.e  = max(0.0, min(0.999, ecc))
        self.omega = math.radians(omega_deg)
        self.t0 = t0_days * DAY
        self.gamma = gamma_ms
        self.lam0  = base_lambda_nm

    # Orbital helpers
    def mean_anomaly(self, t):
        n = 2.0*math.pi/self.P
        return (n*(t-self.t0)) % (2.0*math.pi)

    def eccentric_anomaly(self, M, tol=1e-10, itmax=50):
        e = self.e
        if e < 1e-12:
            return M
        E = M if e < 0.8 else math.pi
        for _ in range(itmax):
            f  = E - e*math.sin(E) - M
            fp = 1.0 - e*math.cos(E)
            dE = -f/fp
            E += dE
            if abs(dE) < tol:
                break
        return E

    def true_anomaly(self, E):
        e = self.e
        if e < 1e-12:
            return E
        return 2.0*math.atan2(math.sqrt(1+e)*math.sin(E/2.0),
                              math.sqrt(1-e)*math.cos(E/2.0))

    def rv(self, t):
        """Star line-of-sight velocity [m/s]"""
        M  = self.mean_anomaly(t)
        E  = self.eccentric_anomaly(M)
        nu = self.true_anomaly(E)
        theta = nu + self.omega

        mu = G * (self.Ms + self.Mp)
        h  = math.sqrt(mu * self.a * (1.0 - self.e**2))
        v_scale = mu / h

        v_rel_los = -math.sin(theta) * v_scale
        v_star_los = - (self.Mp / (self.Ms + self.Mp)) * v_rel_los * math.sin(self.i)
        return v_star_los + self.gamma

    def doppler_shift_nm(self, v):
        return self.lam0 * (1.0 + v / C)

    def orbit_xy(self, t_array, scale=1.0):
        """Return star/planet positions projected to sky plane for visualization."""
        mu = G * (self.Ms + self.Mp)
        a_s = self.a * (self.Mp / (self.Ms + self.Mp))
        a_p = self.a * (self.Ms / (self.Ms + self.Mp))
        xs, ys, xp, yp = [], [], [], []
        for t in t_array:
            M  = self.mean_anomaly(t)
            E  = self.eccentric_anomaly(M)
            nu = self.true_anomaly(E)
            theta = nu + self.omega
            fac = (1.0 - self.e**2) / (1.0 + self.e*math.cos(nu))
            r_s = a_s * fac * scale
            r_p = a_p * fac * scale
            xsb = -r_s * math.cos(theta) * math.sin(self.i)
            ysb = -r_s * math.sin(theta)
            xpb =  r_p * math.cos(theta) * math.sin(self.i)
            ypb =  r_p * math.sin(theta)
            xs.append(ysb); ys.append(xsb)  # rotate for plotting (y_sky, +LOS)
            xp.append(ypb); yp.append(xpb)
        return np.array(xs), np.array(ys), np.array(xp), np.array(yp)
