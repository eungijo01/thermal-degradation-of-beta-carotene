"""
Ea_literature_comparison.py
---------------------------
Compares the apparent activation energy (E_a) for beta-carotene thermal
degradation measured in this work against literature values, in two panels:

  (left)  Arrhenius-plane slope comparison. All lines are shifted to a common
          pivot at 200 C so that ONLY the slope (= -E_a/R) is compared; the
          vertical position is arbitrary and does NOT represent absolute rate.
          Thick segment = measured temperature range; thin = extrapolated slope.

  (right) Apparent E_a vs the midpoint of each study's measurement temperature
          range, with horizontal bars showing that range. Illustrates that the
          apparent E_a tends to fall as the measurement temperature rises
          (with matrix as a secondary factor).

This-work rate constants are re-derived from the raw UV-Vis spectra so the
figure is fully reproducible from primary data (FAIR).

Inputs (raw UV-Vis, tab-separated wavelength<TAB>absorbance after a
'>>>>>Begin Spectral Data<<<<<' header line):
    starlab-absorbance-carotene-0226-*min*.txt   (150 C series, trusted)
    starlab-absorbance-carotene-0326-*min*.txt   (300 C series, trusted)

Metric definitions (identical to peak_vs_auc / arrhenius_calculation notebooks):
    peak = mean absorbance over 450-550 nm
    AUC  = trapezoidal integral over 200-700 nm
    first-order fit on values normalised to t=0: A(t)/A(0) = exp(-k t), k in min^-1
    two-point Arrhenius between 150 and 300 C, k converted min^-1 -> s^-1.
"""

import glob
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

R = 8.314  # J / (mol K)

PALETTE = {
    "dark":  "#003d5c",   # 진한 남색
    "teal1": "#005d71",
    "teal2": "#007d67",
    "green": "#1a983f",
    "olive": "#92a800",
    "orange":"#ffa600",   # 하이라이트(=this work)
}

# Metric windows (nm)
PEAK_LO, PEAK_HI = 450, 550
AUC_LO, AUC_HI = 200, 700

# Trusted single series per temperature (date code -> filename glob)
TRUSTED = {150: "0226", 300: "0326-2"}




# 코드 상단, 상수 정의 부분에 추가
DATA_DIR = r"C:\Users\user\Desktop\biopigment\data"   # 실제 데이터 폴더로 수정

TRUSTED = {150: "0226", 300: "0326-2"}   # 폴더 이름 (노트북과 동일하게)

def rate_constants(folder_name):
    """폴더 안의 모든 파일을 시간별로 평균"""
    folder = os.path.join(DATA_DIR, folder_name)
    files = glob.glob(os.path.join(folder, "starlab-absorbance-carotene-*.txt"))
    if not files:
        raise FileNotFoundError(f"No files in folder: {folder}")

    by_time = {}
    for fp in files:
        minutes = parse_minutes(os.path.basename(fp))
        if minutes is None:
            continue
        wl, ab = load_spectrum(fp)
        pmask = (wl >= PEAK_LO) & (wl <= PEAK_HI)
        amask = (wl >= AUC_LO) & (wl <= AUC_HI)
        by_time.setdefault(minutes, []).append(
            (ab[pmask].mean(), np.trapezoid(ab[amask], wl[amask]))
        )
    times = np.array(sorted(by_time))
    peak = np.array([np.mean([v[0] for v in by_time[t]]) for t in times])
    auc  = np.array([np.mean([v[1] for v in by_time[t]]) for t in times])

    def fit_k(values):
        norm = values / values[0]
        popt, _ = curve_fit(first_order, times, norm, p0=[1.0, 0.01], maxfev=10000)
        return popt[1]

    return fit_k(peak), fit_k(auc)


# --------------------------------------------------------------------------- #
# 1. Load raw spectra and derive rate constants
# --------------------------------------------------------------------------- #
def load_spectrum(path):
    """Return (wavelength, absorbance) arrays from one StarLab .txt export."""
    wl, ab = [], []
    started = False
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if "Begin Spectral Data" in line:
                started = True
                continue
            if started:
                parts = line.replace("\r", "").split("\t")
                if len(parts) == 2:
                    try:
                        wl.append(float(parts[0]))
                        ab.append(float(parts[1]))
                    except ValueError:
                        pass
    arr = np.array(sorted(zip(wl, ab)))
    return arr[:, 0], arr[:, 1]


def parse_minutes(filename):
    m = re.search(r"(\d+)min", filename)
    return int(m.group(1)) if m else None


def first_order(t, A0, k):
    return A0 * np.exp(-k * t)


def two_point_Ea(T1_C, T2_C, k1_min, k2_min):
    """Two-point Arrhenius. k given in min^-1; converted to s^-1 (affects A only)."""
    T1, T2 = T1_C + 273.15, T2_C + 273.15
    k1, k2 = k1_min / 60.0, k2_min / 60.0            # min^-1 -> s^-1
    x = np.array([1.0 / T1, 1.0 / T2])
    y = np.log([k1, k2])
    slope = (y[1] - y[0]) / (x[1] - x[0])
    return -slope * R                                # E_a in J/mol


# Derive values from raw data
kp150, ka150 = rate_constants(TRUSTED[150])
kp300, ka300 = rate_constants(TRUSTED[300])
Ea_peak = two_point_Ea(150, 300, kp150, kp300) / 1000.0   # kJ/mol
Ea_auc = two_point_Ea(150, 300, ka150, ka300) / 1000.0    # kJ/mol

# Uncertainties on E_a were propagated from the rate-constant standard errors in
# the arrhenius_calculation notebook; quoted here for the figure labels.
sEa_peak, sEa_auc = 2.3, 1.6  # kJ/mol


# --------------------------------------------------------------------------- #
# 2. Literature values (all matrices noted; A omitted where not reported)
# --------------------------------------------------------------------------- #
# name, Ea(kJ/mol), T_min(C), T_max(C), colour, marker
literature = [
    ("Akonor 2023 (solar, flour)", 124.2, 25, 45,  PALETTE["dark"],  "^"),
    ("Akonor 2023 (drum, flour)",   85.8, 25, 45,  PALETTE["teal1"], "^"),
    ("Dutta 2006 (pumpkin puree)",  27.27, 70, 100, PALETTE["teal2"], "s"),
    ("Kumar 2024 (sunflower oil)",  56.65, 135, 220, PALETTE["green"], "D"),
]


# --------------------------------------------------------------------------- #
# 3. Figure
# --------------------------------------------------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- Panel 1: slope comparison about a common pivot ----------------------- #
T0 = 200 + 273.15                                    # arbitrary common pivot
T_full = np.linspace(25 + 273.15, 300 + 273.15, 100)
x_full = 1000.0 / T_full

ax1.axvline(1000.0 / T0, color="grey", ls=":", lw=0.9, alpha=0.7)
ax1.plot(1000.0 / T0, 0, "ko", ms=7, zorder=10)


def plot_slope(ax, Ea, Tmin, Tmax, colour, label, ls="-", lw=2.2):
    """Line of slope -Ea/R passing through (1/T0, 0); thin over full range,
    thick over the study's measured range."""
    y_full = -Ea * 1000.0 / R * (1.0 / T_full - 1.0 / T0)
    ax.plot(x_full, y_full, color=colour, lw=1.0, ls=ls, alpha=0.3)

    Tm = np.linspace(Tmin + 273.15, Tmax + 273.15, 50)
    ym = -Ea * 1000.0 / R * (1.0 / Tm - 1.0 / T0)
    ax.plot(1000.0 / Tm, ym, color=colour, lw=lw, ls=ls, label=label, zorder=5)




for name, Ea, Tmin, Tmax, colour, _ in literature:
    plot_slope(ax1, Ea, Tmin, Tmax, colour, f"{name}: {Ea:.0f}")
plot_slope(ax1, Ea_peak, 150, 300, PALETTE["orange"], f"This study peak: {Ea_peak:.1f}", lw=3)
plot_slope(ax1, Ea_auc,  150, 300, PALETTE["olive"],  f"This study AUC: {Ea_auc:.1f}", ls="--", lw=3)

#ax1.annotate("thick = measured range\nthin = extrapolated slope",
#             (1.75, 0.5), fontsize=8, color="0.35", va="top")
#ax1.annotate("common pivot at 200 C\n(vertical position arbitrary)",
#             (1000.0 / T0, 0), textcoords="offset points", xytext=(10, 12),
#             fontsize=8, color="0.3")
ax1.set_xlabel("1000 / T   (K$^{-1}$)",  fontsize=12)
ax1.set_ylabel(r"$\ln k - \ln k_{200\,^\circ\mathrm{C}}$",  fontsize=12)
ax1.set_title("(A) Slope (= $-E_a/R$) comparison: steeper = higher $E_a$", fontsize=13)
ax1.legend(fontsize=11, loc="lower left", title="$E_a$ (kJ/mol)", title_fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_xlim(1.7, 3.5)

# ---- Panel 2: Ea vs measurement temperature ------------------------------- #
for name, Ea, Tmin, Tmax, colour, marker in literature:
    Tmid = 0.5 * (Tmin + Tmax)
    ax2.plot([Tmin, Tmax], [Ea, Ea], color=colour, lw=1.3, alpha=0.4)
    ax2.plot(Tmid, Ea, marker, color=colour, ms=9, mec="k", label=name)


ax2.plot([150, 300], [Ea_peak, Ea_peak], color=PALETTE["orange"], lw=1.3, alpha=0.4)
ax2.errorbar(225, Ea_peak, #yerr=sEa_peak#, 
             fmt="o", color=PALETTE["orange"], ms=9, mec="k",
             capsize=5, label=f"This study peak ({Ea_peak:.1f}\u00b1{sEa_peak})")
ax2.errorbar(225, Ea_auc,  #yerr=sEa_auc, 
             fmt="o", color=PALETTE["olive"],  ms=9, mec="k",
             capsize=5, label=f"This study AUC ({Ea_auc:.1f}\u00b1{sEa_auc})")
ax2.set_xlabel("Measurement temperature range (\u00b0C)", fontsize=12)
ax2.set_ylabel("Apparent $E_a$ (kJ/mol)", fontsize=12)
ax2.set_title("(B) $E_a$ vs measurement temperature range: lower $E_a$ at higher T", fontsize=13)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("Ea comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"This work  peak: Ea = {Ea_peak:.1f} +/- {sEa_peak} kJ/mol")
print(f"This work  AUC : Ea = {Ea_auc:.1f} +/- {sEa_auc} kJ/mol")
