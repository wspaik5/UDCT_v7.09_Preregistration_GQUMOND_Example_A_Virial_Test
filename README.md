# UDCT v7.09 — pre-registered GQUMOND example-A virial test (code)

Companion code for **UDCT v7.09, "Pre-registration: Applying the v7.06 virial and Jeans protocol to
Milgrom's published GQUMOND length-screening example, with a QUMOND control, in three ultra-faint
dwarfs"** (Won Shik Paik, 30 September 2026).

## Status: code released before any run

**No result of the registered calculation exists in this repository.** The script was written to
implement the registration exactly as frozen, and it is released *before* it has been run on the
registered cases, so that the implementation is fixed in advance of the outcome. Results (and the
exact commit that produced them) will be released afterwards, in a separate note.

What *has* been run is `--selftest`, which checks analytic identities and that the copied profile
geometry matches the v7.06 script. It evaluates **no registered quantity** (no `V/V_N`, no effective
mass, no Jeans pressure).

The question being tested: is the negative virial work found in v7.06 for the UDCT action a property
of Hessian-dependent GQUMOND theories in general, or specific to the UDCT integral construction
(see v7.08)? Milgrom's example A is one member of the class; **this test does not bear on UDCT
itself**, which is not the theory under test.

## What the script does

The registration (Section 2) freezes the implementation as *the released v7.06 script (SHA-256 prefix
`a64440e1350ae8bf`) with only the constitutive law replaced*. Accordingly:

- `UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce.py` (the **unmodified** v7.06 script) must sit in the
  same folder. The v7.09 script computes its SHA-256 and **refuses to run** unless it starts with
  `a64440e1350ae8bf`.
- Objects (Leo IV, Pegasus III, Hydrus I), both stellar profiles, the visible McMillan (2017) disks +
  point bulge as a local second-order Taylor field, the radial/angular grids, the
  integration-by-parts evaluation of `V`, and the zero-outer-pressure Jeans integral are all taken
  from that file.
- Only the constitutive law is new. It is analytic (no action integral), so v7.06's "action nodes"
  grid parameter does not apply.

### Theory under test (registration Section 1)

```
P = f(u) Q(Z/f),   Z = (∇ψ)²/a0²,   u = (∇ψ)² / (ℓ0² ψ,ij ψ,ij),   f(u) = u/(1+u)
Q′(Y²) = ν0(Y) = (1 + 1/Y)^(1/2),   Q(0) = 0,   a0 = 1.082401e-10 m/s²
∇²φ = (a0²/2)[ ∇i(∂P/∂ψ,i) − ∇i∇j(∂P/∂ψ,ij) ]      (both terms kept)
```

Written out (W = Z/f, s = ψ,ij ψ,ij), which is what the code evaluates:

```
ν_eff = Q′(W) + (Q − W Q′) f′(u) u / Z            multiplies ∇ψ
P_ij  = −2 a0² (Q − W Q′) f′(u) (u/s) ψ,ij        enters as −(1/2) ∂i∂j P_ij
Q − W Q′ = (1/2)[ √(Y(Y+1)) − asinh(√Y) ],  Y = √W
```

Control: QUMOND itself (`f ≡ 1`): `ν_eff = Q′(Z)`, `P_ij = 0`.

### Registered grid and quantities

| Registered item | In the code |
|---|---|
| ℓ0 grid 0.03, 0.1, 0.3, 1, 3, 10 pc | `ELL0_GRID_PC` (36 cases = 3 objects × 2 profiles × 6 ℓ0) |
| QUMOND control (6 cases) | `--mode control` |
| `V/V_N` (primary) | JSON key `virial` (with `leading`, `derivative`, `boundary`, and the `direct` cross-check from v7.06) |
| share of the Hessian-derivative term | `derivative_abs_fraction` |
| ⟨g_r⟩/g* at r_h; sign over 0.1–10 r_h | `local_force_over_gstar_at_rh`, `force_negative_in_range`, `min_force_over_gstar_in_range` |
| `M_eff(<r) = r²⟨g_r⟩/G`, its minimum, sign | `meff_min_msun`, `meff_min_over_mstar_total`, `meff_min_at_r_over_rh`, `meff_negative` |
| min σ_r² and projected LOS rms inside R_e | `jeans` block (v7.06 `aperture_jeans`) |
| warning if outward force > 10× stellar Newtonian | `warning_outward_gt10x_newtonian`; flagged cases get an automatic `truncated_5rh_followup` |
| sign claimed only if default and smoke grids agree | `--convergence` |

### Decision rules (registration Section 3, implemented in `classify`)

- **Outcome A — obstruction shared:** `V/V_N < 0` for at least one object/profile at some ℓ0, while the
  QUMOND control is positive.
- **Outcome B — construction-specific:** `V/V_N > 0` for all 36 cases.
- **Outcome C — control negative:** QUMOND itself gives `V/V_N < 0` anywhere; the protocol or inputs are
  suspect and no claim about GQUMOND is made.
- Jeans results are reported under every outcome but do not change A/B/C. A negative `M_eff` or outward
  mean force is an equilibrium diagnostic in a static model, not a prediction that stars are ejected.

## Install and run

Python ≥ 3.10, `numpy`, `scipy`.

```bash
python -m pip install "numpy>=1.24" "scipy>=1.10"

# 1. Analytic self-test (evaluates no registered quantity)
python UDCT_v7_09_Preregistration_GQUMOND_Example_A_Virial_Test_run.py --selftest

# 2. The registered calculation: all 42 cases on the default AND smoke grids,
#    with the registered convergence rule and A/B/C classification
python UDCT_v7_09_Preregistration_GQUMOND_Example_A_Virial_Test_run.py \
       --mode all --convergence > results_v7_09.json
```

Other switches: `--mode control|gqumond|all`, `--object 'Leo IV'`, `--ell0 1.0` (single value; partial
runs are not classified), `--smoke`, and the grid overrides (`--radial`, `--polar`, `--azimuth`,
`--disk-radial`, `--disk-azimuth`, `--disk-height`). Output is JSON on stdout and includes the SHA-256 of
this script and of the v7.06 script.

**Before running, record** the commit hash of this repository and the output of
`sha256sum UDCT_v7_09_Preregistration_GQUMOND_Example_A_Virial_Test_run.py` in the Zenodo record, so
that the code that produced the results is the code that was released.

## Implementation choices not fixed by the PDF

The registration does not specify the following; they are choices made in this implementation and are
declared here so that they are not mistaken for registered rules.

1. **"Taylor field truncated at 5 r_h"** (Section 3b) is implemented as an **outer integration radius of
   5 r_h** (radial mesh to 5 r_h, zero outer pressure there). Over 0.1–10 r_h, the force-sign and
   `M_eff` diagnostics then cover only 0.1–5 r_h in that follow-up.
2. **ν0** is exactly `(1 + 1/Y)^(1/2)` as registered. The v7.06 UDCT law `ν_c` additionally contains a
   high-`x` factor `1/(1 + (x/490)²)`; that factor is **not** used here.
3. **"Share of the Hessian-derivative term"** is reported as `|derivative| / (|leading| + |derivative| +
   |boundary|)`, using the decomposition returned by the v7.06 `work_from_state`.
4. **Range for `M_eff` and force sign** is 0.1–10 r_h on the v7.06 radial mesh (80 r_h for Plummer, 40 r_h
   for the tapered cusp).
5. **Jeans quantities** are those of the v7.06 `aperture_jeans` (minimum σ_r² over the mesh; formal
   aperture rms at R_e), unchanged.
6. **Convergence** is judged on the sign of `V/V_N` and the sign of `M_eff` for every case, and on the
   A/B/C outcome, between the default and smoke grids. `claim_allowed` is true only if all agree and the
   full registered set was run.

## Limits declared in the registration

Only one GQUMOND example and one `f`; the Galaxy is a local Taylor field (no global nonaxisymmetric
solve); spherical isotropic Jeans reduction with no anisotropy, binaries, tides or time dependence;
three objects only, no statistical claim; Milgrom's `Q` normalization differs in convention from the
campaign's `ν0`, and the registered `Q` is the campaign's choice. No LMC and no dark halo.

## Files

- `UDCT_v7_09_Preregistration_GQUMOND_Example_A_Virial_Test_2026-09-30.pdf` — the pre-registration (CC BY 4.0)
- `UDCT_v7_09_Preregistration_GQUMOND_Example_A_Virial_Test_run.py` — the implementation (MIT)
- `UDCT_v7_06_Hessian_Term_Sign_Audit_reproduce.py` — the **unmodified** v7.06 script, required (MIT;
  SHA-256 prefix `a64440e1350ae8bf`)
- `README.md` — this file (MIT)
- `LICENSE.txt` — MIT license for the scripts and this README

## Provenance and acknowledgement of assistance

This script and README were written with the assistance of a large language model (Anthropic Claude)
under the author's direction, as stated in the registration's Section 5. The self-test results above were
produced by running the script; the registered calculation has not been run at the time of release.
Independent verification by a physicist is welcome; please report errors via the repository issue
tracker.

## References

M. Milgrom, *Generalizations of quasilinear MOND*, Phys. Rev. D **108**, 084005 (2023),
arXiv:2305.01589.
P. J. McMillan, MNRAS **465**, 76 (2017), doi:10.1093/mnras/stw2759.
A. B. Pace, Local Volume Database (2025), doi:10.33232/001c.144859.

## License

The registration (PDF) is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). The scripts and this README are licensed under the MIT License (see `LICENSE.txt`).

Copyright (c) 2026 Won Shik Paik.
