#!/usr/bin/env python3
"""UDCT v7.09 -- implementation of the pre-registered GQUMOND example-A test.

Companion to:
  "UDCT v7.09 -- Pre-registration: Applying the v7.06 virial and Jeans protocol
   to Milgrom's published GQUMOND length-screening example, with a QUMOND
   control, in three ultra-faint dwarfs" (30 September 2026).

What this script is
-------------------
The registration (Section 2) freezes the implementation as "the released v7.06
script (SHA-256 prefix a64440e1350ae8bf) with only the constitutive law
replaced".  This file does exactly that:

  * the unmodified v7.06 script is loaded from the same folder and its SHA-256
    is checked against the registered prefix before anything runs;
  * objects, profiles, Galactic Taylor field, radial/angular grids, the
    integration-by-parts evaluation of V, and the zero-outer-pressure Jeans
    integral are all taken from that unmodified file;
  * only the constitutive law (the multiplier on grad(psi) and the Hessian
    tensor) is new, and it is analytic -- there is no action integral here.

Theory (registration Section 1; Milgrom 2023, arXiv:2305.01589, Eqs. 7, 13-16):
  P = f(u) Q(Z/f),  Z = (grad psi)^2/a0^2,  u = (grad psi)^2/(l0^2 psi,ij psi,ij),
  f(u) = u/(1+u),  Q'(Y^2) = nu0(Y) = (1 + 1/Y)^(1/2),  Q(0) = 0,
  lap(phi) = div(nu_eff grad psi) - (1/2) d_i d_j P_ij        (both terms kept)
with
  nu_eff = Q'(W) + (Q - W Q') f'(u) u / Z,                    W = Z/f
  P_ij   = -2 a0^2 (Q - W Q') f'(u) (u/s) psi,ij,             s = psi,ij psi,ij
The QUMOND control is f == 1 (nu_eff = Q'(Z), P_ij = 0).

Status of this file: the registered calculation has NOT been run by the author
of this file.  --selftest checks only analytic identities and that the copied
profile geometry matches the v7.06 script; it evaluates no registered quantity
(no V/V_N, no effective mass, no Jeans pressure).

Needs numpy and scipy.  Run --help for options.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

# --------------------------------------------------------------------------
# Load the frozen v7.06 script (hash-checked)
# --------------------------------------------------------------------------
V706_NAME = "UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce.py"
V706_SHA256_PREFIX = "a64440e1350ae8bf"        # registered in v7.09, Section 2


def _load_v706():
    default = Path(__file__).resolve().with_name(V706_NAME)
    path = Path(os.environ.get("UDCT_V706_SCRIPT", default))
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen v7.06 script not found: {path}\n"
            f"Place the unmodified '{V706_NAME}' next to this file "
            f"(or set UDCT_V706_SCRIPT).")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(V706_SHA256_PREFIX):
        raise RuntimeError(
            f"SHA-256 of {path.name} is {digest}; the registration requires "
            f"prefix {V706_SHA256_PREFIX}.  Refusing to run on a modified "
            f"v7.06 script.")
    spec = importlib.util.spec_from_file_location("udct_v706_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module         # required for dataclasses
    spec.loader.exec_module(module)
    return module, digest


V706, V706_SHA256 = _load_v706()
G, MSUN, PC, A0, ML = V706.G, V706.MSUN, V706.PC, V706.A0, V706.ML
DWARFS, PROFILES, Grid = V706.DWARFS, V706.PROFILES, V706.Grid

ELL0_GRID_PC = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)   # registered grid (Section 1)
TRUNCATE_RH = 5.0           # registration Section 3(b) follow-up check
OUTWARD_WARNING = 10.0      # registration Section 3(b): factor 10
RANGE_RH = (0.1, 10.0)      # registered range for force sign / M_eff

SMOKE = dict(radial=180, polar=16, azimuth=24,
             disk_radial=40, disk_azimuth=24, disk_height=8)


# --------------------------------------------------------------------------
# Q for nu0(Y) = sqrt(1 + 1/Y)  (closed forms; checked in --selftest)
# --------------------------------------------------------------------------
def q_func(w):
    """Q(W) = int_0^W nu0(sqrt(s)) ds.  Used only by --selftest."""
    y = np.sqrt(w)
    return 0.5 * (2 * y + 1) * np.sqrt(y * (y + 1)) - 0.5 * np.arcsinh(np.sqrt(y))


def q_prime(w):
    """Q'(W) = nu0(sqrt(W)) = sqrt(1 + 1/sqrt(W))."""
    return np.sqrt(1.0 + 1.0 / np.sqrt(w))


def q_minus_wqp(w):
    """D(W) = Q(W) - W Q'(W) = (1/2)[sqrt(Y(Y+1)) - asinh(sqrt(Y))], Y = sqrt(W)."""
    y = np.sqrt(w)
    return 0.5 * (np.sqrt(y * (y + 1.0)) - np.arcsinh(np.sqrt(y)))


def gq_kernel(g2, s, ell0_m):
    """Constitutive law at points with |grad psi|^2 = g2 and psi,ij psi,ij = s.

    Returns (nu_eff, wh) where P_ij = wh * psi,ij.  ell0_m=None is the QUMOND
    control (f == 1): nu_eff = Q'(Z), wh = 0.
    """
    z = g2 / A0 ** 2
    if ell0_m is None:
        return q_prime(z), np.zeros_like(z)
    u = g2 / (ell0_m ** 2 * s)
    w = z * (1.0 + 1.0 / u)                  # W = Z / f(u)
    common = q_minus_wqp(w) * u / (1.0 + u) ** 2     # (Q - W Q') f'(u) u
    nu = q_prime(w) + common / z
    wh = -2.0 * A0 ** 2 * common / s
    return nu, wh


# --------------------------------------------------------------------------
# Profile / geometry part: copied verbatim from V706.model_state
# (everything up to, but excluding, the constitutive law)
# --------------------------------------------------------------------------
def geometry(obj, profile, ratios, e, h, grid):
    mass = ML * obj.luminosity * MSUN
    re = obj.re_pc * PC
    if profile == "Plummer":
        b = re
        rh = b / np.sqrt(2 ** (2 / 3) - 1)
    elif profile == "Tapered cusp":
        _, a_re, rh_re, _, _ = V706.cusp_calibration()
        b, rh = a_re * re, rh_re * re
    else:
        raise ValueError(profile)
    r = np.asarray(ratios) * rh
    polar, wpolar = leggauss(grid.polar)
    az = 2 * np.pi * np.arange(grid.azimuth) / grid.azimuth
    n = np.stack((np.repeat(np.sqrt(1 - polar ** 2), grid.azimuth) *
                  np.tile(np.cos(az), grid.polar),
                  np.repeat(np.sqrt(1 - polar ** 2), grid.azimuth) *
                  np.tile(np.sin(az), grid.polar),
                  np.repeat(polar, grid.azimuth)), axis=1)
    weights = np.repeat(wpolar, grid.azimuth) / (2 * grid.azimuth)
    ne, nhn = n @ e, np.einsum("ki,ij,kj->k", n, h, n)
    hn = n @ h.T
    ehn, hn2 = hn @ e, np.einsum("ki,ki->k", hn, hn)

    if profile == "Plummer":
        gn = G * mass * r / (r * r + b * b) ** 1.5
        lr = G * mass * (b * b - 2 * r * r) / (r * r + b * b) ** 2.5
        lt = G * mass / (r * r + b * b) ** 1.5
        rho = (1 + (r / b) ** 2) ** -2.5
        dlogrho = -5 * r / (r * r + b * b)
    else:
        q, a_re, _, cum, total = V706.cusp_calibration()
        rq = r / re
        rho_q = (a_re / rq) / (1 + rq / a_re) ** 3 * np.exp(-(rq / 10) ** 4)
        rho_mass = mass * rho_q / (4 * np.pi * re ** 3 * total)
        contained = np.interp(rq, q, cum, left=0, right=1)
        gn = G * mass * contained / r ** 2
        lt = gn / r
        lr = 4 * np.pi * G * rho_mass - 2 * lt
        rho = (b / r) / (1 + r / b) ** 3 * np.exp(-(r / (10 * re)) ** 4)
        dlogrho = -1 / r - 3 / (r + b) - 4 * r ** 3 / (10 * re) ** 4

    # s = psi,ij psi,ij  (full Hessian: stellar part + Galactic tide)
    t2 = (lr[:, None] ** 2 + 2 * lt[:, None] ** 2 +
          2 * lt[:, None] * np.trace(h) +
          2 * (lr - lt)[:, None] * nhn + np.trace(h @ h))
    rr = r[:, None]
    # |grad psi|^2 with grad psi = gn n + E + r H n
    x2 = (gn[:, None] ** 2 + e @ e + 2 * gn[:, None] * ne +
          2 * rr * gn[:, None] * nhn + 2 * rr * ehn + rr * rr * hn2)
    radial_grad = gn[:, None] + ne + rr * nhn          # n . grad psi
    return dict(r=r, rh=rh, b=b, gn=gn, lr=lr, lt=lt, rho=rho, dlogrho=dlogrho,
                n=n, weights=weights, t2=t2, g2=x2, radial_grad=radial_grad,
                trh=np.trace(h), nhn=nhn)


def model_state(obj, profile, ratios, e, h, grid, ell0_pc):
    """Angle-averaged leading force, P_nn and tr(P): same keys as V706.model_state."""
    geo = geometry(obj, profile, ratios, e, h, grid)
    g2 = np.maximum(geo["g2"], 1e-200)
    s = np.maximum(geo["t2"], 1e-200)
    ell0_m = None if ell0_pc is None else ell0_pc * PC
    nu, wh = gq_kernel(g2, s, ell0_m)
    wts = geo["weights"]
    lr, lt = geo["lr"], geo["lt"]
    lead = (nu * geo["radial_grad"]) @ wts
    pnn = (wh * (lr[:, None] + geo["nhn"])) @ wts
    trp = (wh * (lr + 2 * lt + geo["trh"])[:, None]) @ wts
    return dict(r=geo["r"], rh=geo["rh"], b=geo["b"], gn=geo["gn"],
                rho=geo["rho"], dlogrho=geo["dlogrho"],
                lead=lead, pnn=pnn, trp=trp)


def mesh(profile, grid, truncate_rh=None):
    """Radial mesh in units of r_h.  Default: the v7.06 mesh (80 / 40 r_h)."""
    if truncate_rh is None:
        return V706.radial_mesh(profile, grid)
    return np.sort(np.unique(np.r_[np.geomspace(0.002, truncate_rh, grid.radial),
                                   1.0]))


def one(obj, profile, e, h, grid, ell0_pc, truncate_rh=None):
    ratios = mesh(profile, grid, truncate_rh)
    st = model_state(obj, profile, ratios, e, h, grid, ell0_pc)
    work, force = V706.work_from_state(st)
    x = st["r"] / st["rh"]
    ratio = force / st["gn"]
    sel = (x >= RANGE_RH[0]) & (x <= RANGE_RH[1])
    meff = st["r"] ** 2 * force / G / MSUN          # Msun, inward-positive force
    mstar_total = ML * obj.luminosity
    k = int(np.argmin(np.where(sel, meff, np.inf)))
    total_abs = abs(work["leading"]) + abs(work["derivative"]) + abs(work["boundary"])
    jr, _ = V706.aperture_jeans(st, force, obj.re_pc)
    return dict(
        profile=profile, ell0_pc=ell0_pc, truncate_rh=truncate_rh,
        virial=work["virial"], leading=work["leading"],
        derivative=work["derivative"], boundary=work["boundary"],
        direct=work["direct"],
        derivative_abs_fraction=float(abs(work["derivative"]) / total_abs),
        local_force_over_gstar_at_rh=float(np.interp(1.0, x, ratio)),
        force_negative_in_range=bool(np.any(ratio[sel] < 0)),
        min_force_over_gstar_in_range=float(np.min(ratio[sel])),
        meff_min_msun=float(meff[k]),
        meff_min_over_mstar_total=float(meff[k] / mstar_total),
        meff_min_at_r_over_rh=float(x[k]),
        meff_negative=bool(meff[k] < 0),
        warning_outward_gt10x_newtonian=bool(np.any(ratio[sel] < -OUTWARD_WARNING)),
        jeans=jr)


# --------------------------------------------------------------------------
# Orchestration and registered decision rules (Section 3)
# --------------------------------------------------------------------------
def make_grid(smoke, overrides):
    base = dict(SMOKE) if smoke else dict(
        radial=Grid.radial, polar=Grid.polar, azimuth=Grid.azimuth,
        disk_radial=Grid.disk_radial, disk_azimuth=Grid.disk_azimuth,
        disk_height=Grid.disk_height)
    for key, val in overrides.items():
        if val is not None:
            base[key] = val
    grid = Grid(**base)                    # 'action' nodes unused (analytic law)
    if min(base.values()) < 4:
        raise ValueError("Each quadrature dimension must be at least four")
    return grid


def run_grid(grid, names, do_control, do_gq, ell0_list):
    out = []
    for name in names:
        obj = DWARFS[name]
        e, h, mass = V706.galactic_field(obj, grid)
        for profile in PROFILES:
            rec = dict(object=name, profile=profile,
                       galaxy_integrated_msun=float(mass),
                       e_m_per_s2=e.tolist(), h_per_s2=h.tolist())
            if do_control:
                rec["control_qumond"] = one(obj, profile, e, h, grid, None)
            if do_gq:
                cases = {}
                for ell0 in ell0_list:
                    case = one(obj, profile, e, h, grid, ell0)
                    if case["warning_outward_gt10x_newtonian"]:
                        # Section 3(b): follow-up with the Taylor field truncated
                        case["truncated_5rh_followup"] = one(
                            obj, profile, e, h, grid, ell0, TRUNCATE_RH)
                    cases[str(ell0)] = case
                rec["gqumond"] = cases
            if do_control and "control_qumond" in rec and \
                    rec["control_qumond"]["warning_outward_gt10x_newtonian"]:
                rec["control_qumond"]["truncated_5rh_followup"] = one(
                    obj, profile, e, h, grid, None, TRUNCATE_RH)
            out.append(rec)
    return out


def classify(outputs, complete):
    """Registered outcomes A / B / C (Section 3).  Needs the full registered set."""
    if not complete:
        return dict(outcome="not classified (partial run)")
    ctrl = [rec["control_qumond"]["virial"] for rec in outputs]
    gq = [(rec["object"], rec["profile"], float(k), c["virial"])
          for rec in outputs for k, c in rec["gqumond"].items()]
    if any(v < 0 for v in ctrl):
        return dict(outcome="C",
                    text="control negative: protocol or inputs suspect; "
                         "no claim about GQUMOND")
    neg = [t for t in gq if t[3] < 0]
    if neg:
        lo, hi = min(t[2] for t in neg), max(t[2] for t in neg)
        return dict(outcome="A", ell0_range_pc=[lo, hi], n_negative=len(neg),
                    text="Milgrom's example A exhibits negative virial work in "
                         f"the stated models for l0 in [{lo}, {hi}] pc")
    return dict(outcome="B", text="the v7.06 obstruction does not appear in "
                "this GQUMOND example; attributed to the UDCT integral construction")


def sign_signature(outputs):
    sig = {}
    for rec in outputs:
        key = f'{rec["object"]}|{rec["profile"]}'
        if "control_qumond" in rec:
            c = rec["control_qumond"]
            sig[key + "|control"] = (c["virial"] > 0, c["meff_negative"])
        for k, c in rec.get("gqumond", {}).items():
            sig[key + f"|l0={k}"] = (c["virial"] > 0, c["meff_negative"])
    return sig


def run(args):
    names = tuple(DWARFS) if args.object == "all" else (args.object,)
    do_control = args.mode in ("control", "all")
    do_gq = args.mode in ("gqumond", "all")
    ell0_list = ELL0_GRID_PC if args.ell0 is None else (args.ell0,)
    complete = (args.object == "all" and args.mode == "all" and args.ell0 is None)
    overrides = dict(radial=args.radial, polar=args.polar, azimuth=args.azimuth,
                     disk_radial=args.disk_radial, disk_azimuth=args.disk_azimuth,
                     disk_height=args.disk_height)
    result = dict(
        status="UDCT v7.09 pre-registered test implementation; "
               "see the registration PDF for decision rules",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        v706_sha256=V706_SHA256, mode=args.mode, complete_registered_set=complete)

    grids = [("default" if not args.smoke else "smoke",
              make_grid(args.smoke, overrides))]
    if args.convergence:
        grids = [("default", make_grid(False, overrides)),
                 ("smoke", make_grid(True, overrides))]
    runs = {}
    for label, grid in grids:
        outputs = run_grid(grid, names, do_control, do_gq, ell0_list)
        runs[label] = dict(grid=grid.__dict__, outputs=outputs,
                           outcome=classify(outputs, complete) if do_control and do_gq
                           else dict(outcome="not classified (needs --mode all)"))
    result["runs"] = runs
    if args.convergence:
        a, b = (sign_signature(runs[k]["outputs"]) for k in ("default", "smoke"))
        disagree = sorted(k for k in a if a[k] != b[k])
        agree_outcome = runs["default"]["outcome"]["outcome"] == runs["smoke"]["outcome"]["outcome"]
        result["convergence"] = dict(
            signs_agree_on_default_and_smoke=(not disagree) and agree_outcome,
            disagreeing_cases=disagree,
            claim_allowed=(not disagree) and agree_outcome and complete)
    return result


# --------------------------------------------------------------------------
# --selftest: analytic identities + geometry regression; no registered quantity
# --------------------------------------------------------------------------
def selftest():
    from scipy.integrate import quad
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("UDCT v7.09 self-test (analytic identities; no registered quantity)")
    check("v7.06 script hash matches the registered prefix", True,
          f"({V706_SHA256[:16]})")

    # T1: closed form for Q and for D = Q - W Q'
    worst = 0.0
    for w in (1e-4, 1e-2, 0.3, 1.0, 25.0, 1e3, 1e6):
        num = quad(lambda s: np.sqrt(1 + 1 / np.sqrt(s)), 0, w, epsabs=0,
                   epsrel=1e-12, limit=400)[0]
        worst = max(worst, abs(q_func(w) / num - 1),
                    abs(q_minus_wqp(w) / (num - w * q_prime(w)) - 1))
    check("Q(W) and Q - W Q' closed forms vs numerical integral", worst < 1e-7,
          f"(max rel. err {worst:.2e})")

    # T2: nu_eff and P_ij coefficient vs finite differences of P(g2, s)
    def pfun(g2, s, ell0):
        u = g2 / (ell0 ** 2 * s)
        f = u / (1 + u)
        return f * q_func(g2 / A0 ** 2 / f)

    rng = np.random.default_rng(709)
    worst_nu = worst_wh = 0.0
    for _ in range(40):
        x = 10 ** rng.uniform(-2, 3)
        u = 10 ** rng.uniform(-3, 4)
        ell0 = 10 ** rng.uniform(15, 18)
        g2 = (x * A0) ** 2
        s = g2 / (ell0 ** 2 * u)
        nu, wh = gq_kernel(np.array(g2), np.array(s), ell0)
        dg, ds = 1e-5 * g2, 1e-5 * s
        pg = (pfun(g2 + dg, s, ell0) - pfun(g2 - dg, s, ell0)) / (2 * dg)
        ps = (pfun(g2, s + ds, ell0) - pfun(g2, s - ds, ell0)) / (2 * ds)
        worst_nu = max(worst_nu, abs(nu / (A0 ** 2 * pg) - 1))
        worst_wh = max(worst_wh, abs(wh / (2 * A0 ** 2 * ps) - 1))
    check("nu_eff = a0^2 dP/d(grad psi)^2 (finite difference)", worst_nu < 1e-5,
          f"(max rel. err {worst_nu:.2e})")
    check("P_ij coefficient = 2 a0^2 dP/ds (finite difference)", worst_wh < 1e-4,
          f"(max rel. err {worst_wh:.2e})")

    # T3: limits
    g2 = np.array((0.7 * A0) ** 2)
    nu_c, wh_c = gq_kernel(g2, np.array(1.0), None)
    s_big = g2 / (1e-9 ** 2 * 1e8)                  # u = 1e8: f -> 1
    nu_g, wh_g = gq_kernel(g2, s_big, 1e-9)
    check("f -> 1 (u >> 1): nu_eff -> QUMOND Q' and Hessian factor -> 0",
          abs(nu_g / nu_c - 1) < 1e-6 and abs(wh_g) * s_big / A0 ** 2 < 1e-6)
    check("QUMOND control: Hessian tensor identically zero", wh_c == 0)

    # T4: copied geometry vs explicit vector/tensor construction
    grid = Grid(radial=12, polar=4, azimuth=4)
    e = np.array([3e-11, -2e-11, 1e-11])
    hh = 3e-35 * rng.normal(size=(3, 3))
    h = 0.5 * (hh + hh.T)
    obj = DWARFS["Leo IV"]
    worst = 0.0
    for prof in PROFILES:
        geo = geometry(obj, prof, np.geomspace(0.01, 8, 12), e, h, grid)
        for i in (0, 5, 11):
            for k in (0, 7, 15):
                nvec = geo["n"][k]
                grad = geo["gn"][i] * nvec + e + geo["r"][i] * (h @ nvec)
                psi_ij = (geo["lr"][i] * np.outer(nvec, nvec) +
                          geo["lt"][i] * (np.eye(3) - np.outer(nvec, nvec)) + h)
                worst = max(worst,
                            abs(geo["g2"][i, k] / (grad @ grad) - 1),
                            abs(geo["t2"][i, k] / np.sum(psi_ij * psi_ij) - 1),
                            abs(geo["radial_grad"][i, k] / (nvec @ grad) - 1))
    check("copied geometry: |grad psi|^2, psi,ij psi,ij, n.grad psi vs explicit",
          worst < 1e-9, f"(max rel. err {worst:.2e})")

    # T5: profile pieces identical to the frozen v7.06 model_state
    worst = 0.0
    for prof in PROFILES:
        ratios = np.geomspace(0.05, 6, 15)
        g06 = Grid(radial=15, polar=4, azimuth=4, action=8)
        ref = V706.model_state(obj, prof, ratios, e, h, g06)
        mine = model_state(obj, prof, ratios, e, h, g06, 1.0)
        for key in ("r", "gn", "rho", "dlogrho"):
            worst = max(worst, float(np.max(np.abs(mine[key] / ref[key] - 1))))
        worst = max(worst, abs(mine["rh"] / ref["rh"] - 1))
    check("profile geometry identical to the frozen v7.06 model_state",
          worst < 1e-12, f"(max rel. diff {worst:.1e})")
    print("self-test:", "ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("control", "gqumond", "all"), default="all")
    p.add_argument("--object", choices=("all", *DWARFS), default="all")
    p.add_argument("--ell0", type=float, default=None,
                   help="single l0 in pc (partial run; no A/B/C classification)")
    p.add_argument("--smoke", action="store_true",
                   help="v7.06 smoke grid (180 x 16 x 24; disk 40 x 24 x 8)")
    p.add_argument("--convergence", action="store_true",
                   help="run default AND smoke grids and compare signs (registered rule)")
    p.add_argument("--selftest", action="store_true",
                   help="analytic identities only; evaluates no registered quantity")
    for name in ("radial", "polar", "azimuth", "disk-radial", "disk-azimuth",
                 "disk-height"):
        p.add_argument("--" + name, type=int, default=None)
    args = p.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    cli()
