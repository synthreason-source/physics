"""
Interactive Tkinter version of the reduced-model NCD channel ion-acceleration
simulation, with sliders to customize all key parameters and a THIRD,
user-adjustable "heavy projectile" ion species (mass and charge state are
both sliders) alongside the H+ and D+ beams from the original figure.

Run locally (requires a display):
python3 ncd_ion_acceleration_tk.py

Requires: numpy, matplotlib, tkinter (standard library on most Python
installs; on some Linux distros install separately, e.g.
`sudo apt install python3-tk`).
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------
c = 2.998e8
e = 1.602e-19
m_e = 9.109e-31
amu = 1.6605e-27  # kg per atomic mass unit
mu0 = 4 * np.pi * 1e-7
eps0 = 8.854e-12

lam_L = 0.8e-6
omega_L = 2 * np.pi * c / lam_L
n_c = eps0 * m_e * omega_L**2 / e**2


class IonAccelSim:
    """Container for one simulation run given a set of parameters."""

    def __init__(self, params):
        self.p = params

    def laser_envelope(self, t, tau_L, t_peak):
        sigma = tau_L / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-(t - t_peak) ** 2 / (2 * sigma ** 2))

    def I_enclosed(self, r, t, R, I_peak, tau_L, t_peak):
        return I_peak * self.laser_envelope(t, tau_L, t_peak) * (1 - np.exp(-(r / R) ** 2))

    def B_theta(self, r, t, R, I_peak, tau_L, t_peak, r_min=1e-8):
        r_eff = np.maximum(r, r_min)
        return mu0 * self.I_enclosed(r_eff, t, R, I_peak, tau_L, t_peak) / (2 * np.pi * r_eff)

    def E_z_field(self, z, t, z_front, E0, sigma_z, tau_L, t_peak):
        return E0 * self.laser_envelope(t, tau_L, t_peak) * np.exp(-((z - z_front) ** 2) / (2 * sigma_z ** 2))

    def E_r_field(self, r, t, R, tau_L, t_peak, E_r_peak=2e11):
        return -E_r_peak * self.laser_envelope(t, tau_L, t_peak) * (r / R) * np.exp(-(r / (2 * R)) ** 2)

    def boris_push(self, x, y, z, ux, uy, uz, q, m, dt, t, z_front, p):
        r = np.sqrt(x ** 2 + y ** 2) + 1e-12
        cos_p, sin_p = x / r, y / r
        Bmag = self.B_theta(r, t, p["R_ch"], p["I_e_peak"], p["tau_L"], p["t_peak"])
        Bx, By, Bz = -Bmag * sin_p, Bmag * cos_p, 0.0

        Er = self.E_r_field(r, t, p["R_ch"], p["tau_L"], p["t_peak"])
        Ex, Ey = Er * cos_p, Er * sin_p
        Ez = self.E_z_field(z, t, z_front, p["E0_sheath"], p["sigma_z"], p["tau_L"], p["t_peak"])

        qmdt2 = q * dt / (2 * m)
        u_minus_x = ux + qmdt2 * Ex
        u_minus_y = uy + qmdt2 * Ey
        u_minus_z = uz + qmdt2 * Ez
        gamma_minus = np.sqrt(1 + (u_minus_x ** 2 + u_minus_y ** 2 + u_minus_z ** 2) / c ** 2)

        tx, ty, tz = qmdt2 * Bx / gamma_minus, qmdt2 * By / gamma_minus, qmdt2 * Bz / gamma_minus
        t2 = tx ** 2 + ty ** 2 + tz ** 2
        sx, sy, sz = 2 * tx / (1 + t2), 2 * ty / (1 + t2), 2 * tz / (1 + t2)

        u_prime_x = u_minus_x + (u_minus_y * tz - u_minus_z * ty)
        u_prime_y = u_minus_y + (u_minus_z * tx - u_minus_x * tz)
        u_prime_z = u_minus_z + (u_minus_x * ty - u_minus_y * tx)

        u_plus_x = u_minus_x + (u_prime_y * sz - u_prime_z * sy)
        u_plus_y = u_minus_y + (u_prime_z * sx - u_prime_x * sz)
        u_plus_z = u_minus_z + (u_prime_x * sy - u_prime_y * sx)

        ux_new = u_plus_x + qmdt2 * Ex
        uy_new = u_plus_y + qmdt2 * Ey
        uz_new = u_plus_z + qmdt2 * Ez

        gamma_new = np.sqrt(1 + (ux_new ** 2 + uy_new ** 2 + uz_new ** 2) / c ** 2)
        vx, vy, vz = ux_new / gamma_new, uy_new / gamma_new, uz_new / gamma_new
        x_new, y_new, z_new = x + vx * dt, y + vy * dt, z + vz * dt
        return x_new, y_new, z_new, ux_new, uy_new, uz_new, gamma_new

    def run(self):
        p = self.p
        rng = np.random.default_rng(42)
        N = p["N_particles"]
        dt = p["dt"]
        N_steps = p["N_steps"]

        species = {
            "H+": dict(q=e, m=1.0 * amu, color="crimson"),
            "D+": dict(q=e, m=2.0 * amu, color="goldenrod"),
            "Heavy": dict(q=p["heavy_q"], m=p["heavy_mass_kg"], color="mediumblue"),
        }

        particles = {}
        for name, sp in species.items():
            r0 = p["R_ch"] * (0.7 + 0.3 * rng.random(N))
            phi0 = 2 * np.pi * rng.random(N)
            z0 = rng.normal(0.0, 2e-6, N)
            x0, y0 = r0 * np.cos(phi0), r0 * np.sin(phi0)
            v_th = np.sqrt(2 * 2e3 * e / sp["m"])
            ux0 = rng.normal(0, v_th, N)
            uy0 = rng.normal(0, v_th, N)
            v_drift = p["v_drift_frac"] * c
            uz0 = rng.normal(v_drift, 0.3 * v_drift, N)
            particles[name] = dict(
                x=x0,
                y=y0,
                z=z0,
                ux=ux0,
                uy=uy0,
                uz=uz0,
                traj_r=[],
                traj_z=[],
                gamma=None,
                **sp
            )

        for step in range(N_steps):
            t = step * dt
            z_front = p["v_front0"] * t * (
                1 + 0.5 * self.laser_envelope(t, p["tau_L"], p["t_peak"])
            )
            for name, P in particles.items():
                P["x"], P["y"], P["z"], P["ux"], P["uy"], P["uz"], gamma = self.boris_push(
                    P["x"],
                    P["y"],
                    P["z"],
                    P["ux"],
                    P["uy"],
                    P["uz"],
                    P["q"],
                    P["m"],
                    dt,
                    t,
                    z_front,
                    p,
                )
                P["gamma"] = gamma
                if step % max(1, N_steps // 150) == 0:
                    r = np.sqrt(P["x"] ** 2 + P["y"] ** 2)
                    P["traj_r"].append(r.copy())
                    P["traj_z"].append(P["z"].copy())

        # peak B for reporting
        r_test = np.linspace(1e-8, 3 * p["R_ch"], 300)
        B_peak = np.max(
            self.B_theta(r_test, p["t_peak"], p["R_ch"], p["I_e_peak"], p["tau_L"], p["t_peak"])
        )

        a0 = 0.85 * np.sqrt(p["I_laser_Wcm2"] * (lam_L * 1e6) ** 2 / 1.37e18)

        return particles, B_peak, a0


class App:
    def __init__(self, root):
        self.root = root
        root.title("Laser-Driven NCD H/D/Heavy-Ion Channel Acceleration")

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        # ---------- Scrollable controls panel ----------
        controls_panel = ttk.Frame(main, padding=5)
        controls_panel.pack(side="left", fill="y")

        controls_canvas = tk.Canvas(
            controls_panel,
            width=320,
            highlightthickness=0,
            borderwidth=0,
        )
        controls_canvas.pack(side="left", fill="y")

        controls_scrollbar = ttk.Scrollbar(
            controls_panel,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls_scrollbar.pack(side="right", fill="y")

        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls = ttk.Frame(controls_canvas, padding=5)
        controls_window = controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )

        def update_scroll_region(event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        controls.bind("<Configure>", update_scroll_region)

        def resize_controls(event):
            controls_canvas.itemconfigure(controls_window, width=event.width)

        controls_canvas.bind("<Configure>", resize_controls)

        def scroll_controls(event):
            controls_canvas.yview_scroll(int(-event.delta / 120), "units")

        controls_canvas.bind_all("<MouseWheel>", scroll_controls)

        # ---------- Plot area ----------
        plot_frame = ttk.Frame(main)
        plot_frame.pack(side="right", fill="both", expand=True)

        self.sliders = {}

        def add_slider(label, key, frm, to, default, resolution=None, fmt="{:.2f}"):
            frame = ttk.Frame(controls)
            frame.pack(fill="x", pady=4)
            ttk.Label(frame, text=label).pack(anchor="w")
            var = tk.DoubleVar(value=default)
            res = resolution if resolution else (to - frm) / 200
            s = tk.Scale(
                frame,
                from_=frm,
                to=to,
                orient="horizontal",
                variable=var,
                resolution=res,
                length=260,
                showvalue=True,
            )
            s.pack(fill="x")
            self.sliders[key] = var

        def add_log_slider(label, key, log_from, log_to, log_default, unit_fn, resolution=0.05):
            """Slider that moves on a log10 scale; shows a decoded human-readable value below it."""
            frame = ttk.Frame(controls)
            frame.pack(fill="x", pady=4)
            ttk.Label(frame, text=label).pack(anchor="w")
            var = tk.DoubleVar(value=log_default)
            readout = tk.StringVar(value=unit_fn(10 ** log_default))

            def on_move(val):
                readout.set(unit_fn(10 ** float(val)))

            s = tk.Scale(
                frame,
                from_=log_from,
                to=log_to,
                orient="horizontal",
                variable=var,
                resolution=resolution,
                length=260,
                showvalue=False,
                command=on_move,
            )
            s.pack(fill="x")
            ttk.Label(frame, textvariable=readout, foreground="#0a5").pack(anchor="w")
            self.sliders[key] = var

        ttk.Label(controls, text="Laser / Plasma Drive", font=("", 11, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        add_slider("Laser intensity [x1e19 W/cm^2]", "I_laser", 1, 50, 5)
        add_slider("Peak electron current I_e [MA]", "I_e_peak_MA", 1, 100, 40)
        add_slider("Channel radius R_ch [um]", "R_ch_um", 1.0, 10.0, 4.0)
        add_slider("Pulse duration tau_L [fs]", "tau_L_fs", 10, 80, 30)

        ttk.Label(controls, text="Acceleration Fields", font=("", 11, "bold")).pack(
            anchor="w", pady=(10, 4)
        )
        add_slider("Sheath field E0 [x1e13 V/m]", "E0_sheath_e13", 0.5, 10.0, 3.0)
        add_slider("Front speed v_front [c]", "v_front_c", 0.05, 0.6, 0.3)
        add_slider("Ion forward drift [c]", "v_drift_c", 0.01, 0.2, 0.06)

        ttk.Label(
            controls,
            text="Heavy Projectile (custom, up to macroscopic!)",
            font=("", 9, "bold"),
        ).pack(anchor="w", pady=(10, 4))

        def mass_readout(mass_g):
            mass_kg = mass_g / 1000
            amu_equiv = mass_kg / amu
            if mass_g < 1e-18:
                return f"{mass_kg:.3e} kg (~{amu_equiv:.1f} amu, atomic scale)"
            elif mass_g < 1e-3:
                return f"{mass_kg:.3e} kg ({mass_g * 1e6:.2f} micrograms)"
            else:
                return f"{mass_kg:.3e} kg ({mass_g:.3g} grams!)"

        def charge_readout(Z):
            return f"{Z:.3g} elementary charges ({Z * e:.2e} C)"

        add_log_slider(
            "Projectile mass [log10 grams]",
            "heavy_mass_log10_g",
            -23.0,
            1.0,
            -22.0,
            mass_readout,
            resolution=0.1,
        )
        add_log_slider(
            "Projectile charge [log10 e]",
            "heavy_charge_log10_e",
            0.0,
            9.0,
            0.6,
            charge_readout,
            resolution=0.1,
        )

        ttk.Label(controls, text="Simulation Settings", font=("", 11, "bold")).pack(
            anchor="w", pady=(10, 4)
        )
        add_slider("Particles per species", "N_particles", 50, 500, 200, resolution=10)
        add_slider("Time steps (x1000)", "N_steps_k", 5, 40, 20, resolution=1)

        self.status = tk.StringVar(value="Adjust sliders, then click Run Simulation.")
        ttk.Label(controls, textvariable=self.status, wraplength=280, foreground="#333").pack(
            pady=(12, 6), anchor="w"
        )

        run_btn = ttk.Button(controls, text="Run Simulation", command=self.run_simulation)
        run_btn.pack(fill="x", pady=(6, 0))

        self.fig = plt.Figure(figsize=(10, 8))
        self.gs = GridSpec(2, 2, figure=self.fig, hspace=0.35, wspace=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.run_simulation()

    def get_params(self):
        s = {k: v.get() for k, v in self.sliders.items()}
        R_ch = s["R_ch_um"] * 1e-6
        tau_L = s["tau_L_fs"] * 1e-15
        params = dict(
            I_laser_Wcm2=s["I_laser"] * 1e19,
            I_e_peak=s["I_e_peak_MA"] * 1e6,
            R_ch=R_ch,
            tau_L=tau_L,
            t_peak=2 * tau_L,
            E0_sheath=s["E0_sheath_e13"] * 1e13,
            sigma_z=1.5 * R_ch,
            v_front0=s["v_front_c"] * c,
            v_drift_frac=s["v_drift_c"],
            heavy_mass_kg=(10 ** s["heavy_mass_log10_g"]) / 1000.0,
            heavy_q=(10 ** s["heavy_charge_log10_e"]) * e,
            N_particles=int(s["N_particles"]),
            N_steps=int(s["N_steps_k"] * 1000),
            dt=5e-18,
        )
        return params

    def run_simulation(self):
        self.status.set("Running...")
        self.root.update_idletasks()

        p = self.get_params()
        sim = IonAccelSim(p)
        particles, B_peak, a0 = sim.run()

        self.fig.clear()
        self.gs = GridSpec(2, 2, figure=self.fig, hspace=0.4, wspace=0.3)

        # ------------------------------------------------------------------
        # Chart 1: Laser Drive Envelope
        # ------------------------------------------------------------------
        ax1 = self.fig.add_subplot(self.gs[0, 0])
        tt = np.linspace(0, 4 * p["tau_L"], 400)
        ax1.plot(tt * 1e15, sim.laser_envelope(tt, p["tau_L"], p["t_peak"]), color="teal", lw=2, label="Envelope")
        ax1.set_xlabel("Time [fs]")
        ax1.set_ylabel("Normalized Amplitude")
        ax1.set_title(f"Laser Drive ($a_0$ = {a0:.2f})")
        ax1.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # Chart 2: Self-Generated Azimuthal Magnetic Field
        # ------------------------------------------------------------------
        ax2 = self.fig.add_subplot(self.gs[0, 1])
        r_um = np.linspace(0, 3 * p["R_ch"], 300) * 1e6
        Bvals = (
            sim.B_theta(r_um * 1e-6, p["t_peak"], p["R_ch"], p["I_e_peak"], p["tau_L"], p["t_peak"])
            / 1e6
        )
        ax2.plot(r_um, Bvals, color="green", lw=2, label="$B_\\theta$")
        ax2.axvline(p["R_ch"] * 1e6, color="k", ls=":", alpha=0.5, label="Channel Edge")
        ax2.set_xlabel("Radius $r$ [um]")
        ax2.set_ylabel("Magnetic Field [MT]")
        ax2.set_title(f"Self-Generated B-Field (Peak {B_peak / 1e6:.2f} MT)")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # Chart 3: Ion Focusing & Phase Space Transport
        # ------------------------------------------------------------------
        ax3 = self.fig.add_subplot(self.gs[1, 0])
        for name, P in particles.items():
            tr_r = np.array(P["traj_r"]) * 1e6
            tr_z = np.array(P["traj_z"]) * 1e6
            n_show = 30
            step = max(1, P["x"].shape[0] // n_show)
            for i in range(0, tr_r.shape[1], step):
                ax3.plot(tr_z[:, i], tr_r[:, i], color=P["color"], alpha=0.35, lw=0.8)
            ax3.plot([], [], color=P["color"], label=name)
        ax3.axhline(0, color="k", lw=1, alpha=0.4)
        ax3.set_ylim(0, p["R_ch"] * 1.4e6)
        ax3.set_xlabel("Longitudinal Position $z$ [um]")
        ax3.set_ylabel("Radial Position $r$ [um]")
        ax3.set_title("Ion Focusing & Transport")
        ax3.legend(fontsize=8, loc="upper right")
        ax3.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # Chart 4: Accelerated Ion Beam Kinetic Energy Spectrum
        # ------------------------------------------------------------------
        ax4 = self.fig.add_subplot(self.gs[1, 1])
        summary_lines = []

        def fmt_energy_J(E_J):
            E_eV = E_J / e
            if E_eV < 1e3:
                return f"{E_eV:.3g} eV"
            elif E_eV < 1e6:
                return f"{E_eV / 1e3:.3g} keV"
            elif E_eV < 1e9:
                return f"{E_eV / 1e6:.3g} MeV"
            elif E_eV < 1e12:
                return f"{E_eV / 1e9:.3g} GeV"
            else:
                return f"{E_J:.3g} J"

        for name, P in particles.items():
            KE_J = (P["gamma"] - 1) * P["m"] * c ** 2
            KE_MeV = KE_J / e / 1e6

            # Exclude Heavy species from the light-ion MeV histogram to prevent axis distortion
            if name != "Heavy" and KE_MeV.size > 0 and np.isfinite(KE_MeV).any() and KE_MeV.max() > 1e-6:
                ax4.hist(KE_MeV, bins=30, alpha=0.55, color=P["color"], label=name, density=True)

            mean_str = fmt_energy_J((P["gamma"] - 1).mean() * P["m"] * c ** 2)
            max_str = fmt_energy_J(KE_J.max())

            v_x = P["ux"] / P["gamma"]
            v_y = P["uy"] / P["gamma"]
            v_z = P["uz"] / P["gamma"]
            v_total = np.sqrt(v_x**2 + v_y**2 + v_z**2)

            v_avg_frac_c = v_total.mean() / c
            v_z_avg_frac_c = v_z.mean() / c

            summary_lines.append(
                f"{name} (m={P['m']:.3e} kg, q={P['q'] / e:.3g} e): "
                f"mean={mean_str}, max={max_str}, "
                f"v_total_avg={v_avg_frac_c:.3e} c, v_z_avg={v_z_avg_frac_c:.3e} c"
            )

        ax4.set_xlabel("Kinetic Energy [MeV] (Light species)")
        ax4.set_ylabel("Normalized Yield")
        ax4.set_title("Accelerated Ion Beam Spectrum")
        ax4.legend(fontsize=8, loc="upper right")
        ax4.grid(alpha=0.3)

        self.canvas.draw()
        self.status.set("Done.\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
