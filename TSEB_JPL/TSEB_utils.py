import numpy as np
import math
from datetime import datetime, UTC
import calendar

deg2rad = math.pi / 180.0
rad2deg = 180.0 / math.pi



def _to_jd_np(year, month, day):
    """Computes the Julian Day for NumPy arrays."""
    a = np.floor((14 - month) / 12)
    y = year + 4800 - a
    m = month + a * 12 - 3
    jdn = (
        day
        + np.floor((153 * m + 2) / 5)
        + y * 365
        + np.floor(y / 4.0)
        - np.floor(y / 100.0)
        + np.floor(y / 400.0)
        - 32045
    )
    return jdn

def _solar_time_np(year, month, day, hour, minute):
    """Computes solar time variables for NumPy arrays."""
    time_t = hour + minute / 60.0
    julian = _to_jd_np(year, month, day)
    julian_ = time_t / 24.0 + julian
    j_cen = (julian_ + 0.5 - 2451545.0) / 36525.0
    lon_sun = (
        j_cen * 0.0003032 + 36000.76983
    ) * j_cen % 360.0 + 280.46646 - 360.0
    an_sun = j_cen * -0.0001537 + 35999.05029 * j_cen + 357.52911
    ecc = (
        j_cen * 0.0000001267 + 0.000042037
    ) * j_cen * -1 + 0.016708634
    ob_ecl = (
        j_cen * -0.001813 + 0.00059
    ) * j_cen + 46.815
    ob_ecl = ob_ecl * j_cen * -1 + 21.448
    ob_ecl = ob_ecl / 60.0 + 26
    ob_ecl = ob_ecl / 60 + 23
    ob_corr = (
        np.cos(j_cen * -1934.136 + 125.04 * deg2rad)
        * 0.00256 + ob_ecl
    )
    var_y = (
        np.tan(ob_corr / 2.0 * deg2rad)
        * np.tan(ob_corr / 2.0 * deg2rad)
    )
    eq_t = (
        np.sin(lon_sun * 2.0 * deg2rad) * var_y
        - np.sin(an_sun * deg2rad) * ecc * 2.0
        + (
            np.sin(an_sun * deg2rad)
            * np.cos(lon_sun * 2.0 * deg2rad)
            * var_y * ecc * 4.0
        )
        - (
            np.sin(lon_sun * 4.0 * deg2rad)
            * var_y * var_y * 0.5
        )
        - (
            np.sin(an_sun * 2.0 * deg2rad)
            * ecc * ecc * 1.25
        )
    ) * 4.0 * rad2deg
    sun_eq = (
        np.sin(an_sun * deg2rad)
        * (
            j_cen * 0.000014 + 0.004817
            * j_cen * -1 + 1.914602
        )
        + (
            np.sin(an_sun * 2.0 * deg2rad)
            * (j_cen * -0.000101 + 0.019993)
        )
        + np.sin(an_sun * 3.0 * deg2rad) * 0.000289
    )
    sun_true = sun_eq + lon_sun
    sun_app = (
        np.sin(j_cen * -1934.136 + 125.04 * deg2rad)
        * -0.00478 - 0.00569 + sun_true
    )
    d = np.arcsin(np.sin(ob_corr * deg2rad) * np.sin(sun_app * deg2rad))
    return d, eq_t

def solar_noon_np(year, month, day, lon):
    """Computes solar noon time for NumPy arrays. lon in radians."""
    d_val, eq_t = _solar_time_np(year, month, day, 0, 0)
    t_noon = (lon * rad2deg * -4 - eq_t + 720) / 1440 * 24.0
    return t_noon

def solar_zenith_np(year, month, day, hour, minute, lon, lat):
    """Computes zenith angle for NumPy arrays. lon and lat in radians."""
    time_t = hour + minute / 60.0
    d, eq_t = _solar_time_np(year, month, day, hour, minute)
    ts_time = ((time_t / 24.0) * 1440.0 + eq_t + 4.0 * (lon * rad2deg)) % 1440.0
    ts_time = np.where(ts_time > 1440, ts_time - 1440, ts_time)
    w = ts_time / 4 + 180
    w = np.where(ts_time / 4 > 0, ts_time / 4 - 180, w)
    zs = np.arccos((np.sin(lat) * np.sin(d)) + (np.cos(lat) * np.cos(d) * np.cos(w * deg2rad)))
    return zs

def emissivity_np(t_air):
    """Apparent atmospheric emissivity for NumPy arrays. t_air in Kelvin."""
    e_atm = np.exp((t_air - 273.16)**2 * -0.0003523) * -0.2811 + 1
    return e_atm

def compute_Rn_c_np(albedo_c, t_air, t_c, t_s, e_atm, rs_c, f):
    """Compute Canopy Net Radiation for NumPy arrays."""
    kl = 0.95
    eps_s = 0.94
    eps_c = 0.99
    sb = 5.67e-8
    lc = t_c**4 * sb * eps_c
    ls = t_s**4 * sb * eps_s
    rle = t_air**4 * sb * e_atm
    rn_c = ((1 - albedo_c) * rs_c) + ((1 - np.exp(-kl * f)) * (rle + ls - 2 * lc))
    return rn_c

def compute_Rn_s_np(albedo_s, t_air, t_c, t_s, e_atm, rs_s, f):
    """Compute Soil Net Radiation for NumPy arrays."""
    kl = 0.95
    eps_s = 0.94
    eps_c = 0.99
    sb = 5.67e-8
    lc = t_c**4 * sb * eps_c
    ls = t_s**4 * sb * eps_s
    rle = t_air**4 * sb * e_atm
    rn_s = ((1 - albedo_s) * rs_s) + (np.exp(-kl * f) * rle) + ((1 - np.exp(-kl * f)) * lc) - ls
    return rn_s

def compute_G0_np(rn, rn_s, ef_s, water_mask, lon_deg, year, month, day, hour, minute):
    """Compute Soil Heat Flux (G0) for NumPy arrays. lon_deg in degrees."""
    w = (ef_s / 0.5)**8 + 1
    w = 1 / w
    c_g = (w * 0.35) + ((1 - w) * 0.31)
    t_g = (w * 100000.0) + ((1 - w) * 74000.0)
    time_np = hour + minute / 60.0
    t_noon_np = solar_noon_np(year, month, day, lon_deg * deg2rad)
    t_g0 = (t_noon_np * -1 + time_np) * 3600
    g0_temp = np.cos((t_g0 + 10800) * 2 * math.pi / t_g)
    g0 = rn_s * c_g * g0_temp
    g0 = np.where(water_mask, rn * 0.5, g0)
    return g0

def compute_u_attr_np(u, d0, z0m, z_u, fm):
    """Friction Velocity for NumPy arrays."""
    u_attr = 0.41 * u / (np.log((z_u - d0) / z0m) - fm)
    u_attr = np.where(u_attr == 0, 10, u_attr)
    u_attr = np.where(u_attr <= 0, 0.01, u_attr)
    return u_attr

def compute_r_ah_np(u_attr, d0, z0h, z_t, fh):
    """Aerodynamic Resistance to Heat Transport for NumPy arrays."""
    r_ah = (np.log((z_t - d0) / z0h) - fh) / (u_attr * 0.41)
    r_ah = np.where(r_ah == 0, 500, r_ah)
    r_ah = np.maximum(r_ah, 1)
    return r_ah

def compute_r_s_np(u_attr, t_s, t_c, hc, f, d0, z0m, leaf, leaf_s, fm_h):
    """Soil Aerodynamic Resistance to Heat Transport (r_s) for NumPy arrays."""
    c_a = 0.004
    c_b = 0.012
    c_c = 0.0025
    u_c = (np.log((hc - d0) / z0m) - fm_h) * u_attr / 0.41
    u_c = np.where(u_c <= 0, 0.1, u_c)
    u_s = np.exp((0.05 / hc - 1) * leaf) * u_c
    r_ss = 1 / (np.exp((1 - 0.05 / hc) * -leaf_s) * c_b + c_a)
    r_s1 = 1 / (np.abs(t_s - t_c)**(1/3) * c_c + u_s * c_b)
    r_s2 = 1 / (u_s * c_b + c_a)
    r_s = ((r_ss - 1) / (0.09 * (f - 0.01))) + 1
    r_s = np.where(f > 0.1, r_s1, r_s)
    r_s = np.where(np.abs(t_s - t_c) < 1, r_s2, r_s)
    r_s = np.where(f > 3, r_s2, r_s)
    return r_s

def compute_r_x_np(u_attr, hc, f, d0, z0m, xl, leaf_c, fm_h):
    """Canopy Boundary Layer Resistance (r_x) for NumPy arrays."""
    c = 175.0
    u_c = (np.log((hc - d0) / z0m) - fm_h) * u_attr / 0.41
    u_c = np.where(u_c <= 0, 0.1, u_c)
    u_d = np.exp(((d0 + z0m) / hc - 1) * leaf_c) * u_c
    u_d = np.where(u_d <= 0, 100, u_d)
    r_x = np.sqrt(xl / u_d) * c / f
    r_x = np.where(u_d == 100, 0.1, r_x)
    return r_x


def temp_separation_tc_np(h_c, fc, t_air, t0, r_ah, r_s, r_x, r_air, cp=1004.16):
    """Compute canopy temperature for NumPy arrays."""
    num_t_c_lin = (t_air / r_ah) + (t0 / (r_s * (1 - fc))) + \
                  (h_c * r_x / (r_air * cp) * ((1 / r_ah) + (1 / r_s) + (1 / r_x)))
    den_t_c_lin = (1 / r_ah) + (1 / r_s) + (fc / (r_s * (1 - fc)))
    t_c_lin = num_t_c_lin / den_t_c_lin
    Td = (t_c_lin * (1 + (r_s / r_ah))) - \
         (h_c * r_x / (r_air * cp) * (1 + (r_s / r_x) + (r_s / r_ah))) - \
         (t_air * r_s / r_ah)
    num_delta_t_c = (t0**4) - (fc * (t_c_lin**4)) - ((1 - fc) * (Td**4))
    den_delta_t_c = (4 * (1 - fc) * (Td**3) * (1 + (r_s / r_ah))) + (4 * fc * (t_c_lin**3))
    delta_t_c = num_delta_t_c / den_delta_t_c
    t_c = t_c_lin + delta_t_c
    t_c = np.where(fc < 0.10, t0, t_c)
    t_c = np.where(fc > 0.90, t0, t_c)
    t_c = np.maximum(t_c, t_air - 10.0)
    t_c = np.minimum(t_c, t_air + 50.0)
    return t_c

def temp_separation_ts_np(t_c, fc, t_air, t0):
    """Compute soil temperature for NumPy arrays."""
    delta = t0**4 - t_c**4 * fc
    delta = np.where(delta <= 0, 10, delta)
    condition_ts = t0**4 <= t_c**4 * fc
    val_if_true = (t0 - (fc * t_c)) / (1 - fc)
    val_if_false = (delta / (1 - fc))**0.25
    t_s = np.where(condition_ts, val_if_true, val_if_false)
    t_s = np.where(fc < 0.1, t0, t_s)
    t_s = np.where(fc > 0.9, t0, t_s)
    t_s = np.maximum(t_s, t_air - 10.0)
    t_s = np.minimum(t_s, t_air + 50.0)
    return t_s

def temp_separation_tac_np(t_c, t_s, fc, t_air, r_ah, r_s, r_x):
    """Compute air temperature at the canopy interface for NumPy arrays."""
    num = (t_air / r_ah) + (t_s / r_s) + (t_c / r_x)
    den = (1 / r_ah) + (1 / r_s) + (1 / r_x)
    t_ac = num / den
    return t_ac

def compute_stability_fh_np(h, t0, u_attr, r_air, z_t, d0, cp=1004.16):
    """Compute atmospheric stability correction for heat (fh) for NumPy arrays."""
    l_ob = (u_attr**3 * t0 * r_air * (cp / -0.41 / 9.806)) / h
    l_ob = np.where(l_ob >= 0, -99, l_ob)
    mh_arg = 16 * (d0 - z_t) / l_ob + 1
    mh = np.power(mh_arg, 0.25)
    mh = np.where(l_ob == -99, 0.0, mh)
    fh = 2 * np.log((mh**2 + 1) / 2)
    fh = np.where((l_ob <= -100) | (l_ob >= 100), 0, fh)
    return fh

def compute_stability_fm_np(h, t0, u_attr, r_air, z_u, d0, z0m, cp=1004.16):
    """Compute atmospheric stability correction for momentum (fm) for NumPy arrays."""
    l_ob = (u_attr**3 * t0 * r_air * (cp / -0.41 / 9.806)) / h
    l_ob = np.where(l_ob >= 0, -99, l_ob)
    mh_arg = 16 * (d0 - z_u) / l_ob + 1
    mh = np.power(mh_arg, 0.25)
    mh = np.where(l_ob == -99, 0.0, mh)
    fm_calc = 2.0 * np.log((1.0 + mh) / 2.0) + np.log((1.0 + (mh**2)) / 2.0) - 2.0 * np.arctan(mh) + (math.pi / 2)
    fm = np.where((l_ob <= -100) | (l_ob >= 100), 0, fm_calc)
    condition_fm_eq = fm == np.log((z_u - d0) / z0m)
    fm = np.where(condition_fm_eq, fm + 1.0, fm)
    return fm

def compute_stability_fm_h_np(h, t0, u_attr, r_air, hc, d0, z0m, cp=1004.16):
    """Compute atmospheric stability correction for momentum at canopy height (fm_h) for NumPy arrays."""
    l_ob = (u_attr**3 * t0 * r_air * (cp / -0.41 / 9.806)) / h
    l_ob = np.where(l_ob >= 0, -99, l_ob)
    mm_h_arg = 16 * (d0 - hc) / l_ob + 1
    mm_h = np.power(mm_h_arg, 0.25)
    mm_h = np.where(l_ob == -99, 0.0, mm_h)
    fm_h_calc = 2.0 * np.log((1.0 + mm_h) / 2.0) + np.log((1.0 + (mm_h**2)) / 2.0) - 2.0 * np.arctan(mm_h) + (math.pi / 2)
    fm_h = np.where((l_ob <= -100) | (l_ob >= 100), 0, fm_h_calc)
    condition_fm_h_eq = fm_h == np.log((hc - d0) / z0m)
    fm_h = np.where(condition_fm_h_eq, fm_h + 1.0, fm_h)
    return fm_h


if __name__ == "__main__":
    pass