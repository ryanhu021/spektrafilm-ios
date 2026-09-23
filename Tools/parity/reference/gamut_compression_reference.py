"""Dependency-free NumPy transcription of every colour-science transform the
reference `spektrafilm.utils.gamut_compression` module calls.

This exists so the Swift port has a source to transcribe from that (a) does not
import colour-science, and (b) is verified against colour-science. Running this
file diffs every function against colour-science 0.4.7 in the pinned oracle venv:

    /tmp/spektra-ref/spektrafilm/.venv/bin/python Tools/parity/reference/gamut_compression_reference.py

Measured agreement over 50k random XYZ samples:
    Oklab   forward / inverse   bit-identical
    JzAzBz  forward / inverse   bit-identical
    CAM16-UCS forward           5.7e-14
    CAM16-UCS inverse           3.7e-14   (finite rows only)

Spec: specs/gamut_compression.md in the port-spec set.
"""

from __future__ import annotations

import numpy as np

# ---------- Oklab ----------
M1 = np.array([
 [0.8189330101, 0.3618667424, -0.1288597137],
 [0.0329845436, 0.9293118715,  0.0361456387],
 [0.0482003018, 0.2643662691,  0.6338517070]])
M1i = np.linalg.inv(M1)
M2 = np.array([
 [0.2104542553,  0.7936177850, -0.0040720468],
 [1.9779984951, -2.4285922050,  0.4505937099],
 [0.0259040371,  0.7827717662, -0.8086757660]])
M2i = np.linalg.inv(M2)
def mv(M, v): return np.matmul(M, v[..., None]).squeeze(-1)
def cbrt_s(x): return np.sign(x)*np.abs(x)**(1.0/3.0)
def xyz_to_oklab(xyz): return mv(M2, cbrt_s(mv(M1, xyz)))
def oklab_to_xyz(lab):
    lms_p = mv(M2i, lab); return mv(M1i, np.sign(lms_p)*np.abs(lms_p)**3)

# ---------- JzAzBz ----------
JB = 1.15; JG = 0.66; JD = -0.56; JD0 = 1.6295499532821566e-11
PQ_m1 = 0.1593017578125; PQ_m2 = 1.7*2523/2**5
PQ_c1 = 0.8359375; PQ_c2 = 18.8515625; PQ_c3 = 18.6875
JA = np.array([
 [ 0.41478972, 0.579999, 0.0146480],
 [-0.2015100,  1.120649, 0.0531008],
 [-0.0166008,  0.264800, 0.6684799]])
JAi = np.linalg.inv(JA)
JB2 = np.array([
 [0.500000,  0.500000,  0.000000],
 [3.524000, -4.066708,  0.542708],
 [0.199076,  1.096799, -1.295875]])
JB2i = np.linalg.inv(JB2)
def spow(a, p): return np.sign(a)*np.abs(a)**p
def pq_inv(C):  # eotf_inverse_ST2084, L_p = 10000
    Yp = spow(C/10000.0, PQ_m1)
    return spow((PQ_c1 + PQ_c2*Yp)/(PQ_c3*Yp + 1.0), PQ_m2)
def pq_fwd(N):  # eotf_ST2084, L_p = 10000
    Vp = spow(N, 1.0/PQ_m2)
    n = np.maximum(0.0, Vp - PQ_c1)
    return 10000.0*spow(n/(PQ_c2 - PQ_c3*Vp), 1.0/PQ_m1)
def xyz_to_jzazbz(xyz):
    X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
    Xp = JB*X - (JB-1.0)*Z
    Yp = JG*Y - (JG-1.0)*X
    lms = mv(JA, np.stack([Xp, Yp, Z], -1))
    Iab = mv(JB2, pq_inv(lms))
    Iz = Iab[...,0]
    Jz = ((1.0+JD)*Iz)/(1.0+JD*Iz) - JD0
    return np.stack([Jz, Iab[...,1], Iab[...,2]], -1)
def jzazbz_to_xyz(jab):
    Jz, az, bz = jab[...,0], jab[...,1], jab[...,2]
    Iz = (Jz + JD0)/(1.0 + JD - JD*(Jz + JD0))
    lms = pq_fwd(mv(JB2i, np.stack([Iz, az, bz], -1)))
    Xp, Yp, Zp = [t for t in np.moveaxis(mv(JAi, lms), -1, 0)]
    X = (Xp + (JB-1.0)*Zp)/JB
    Y = (Yp + (JG-1.0)*X)/JG
    return np.stack([X, Y, Zp], -1)

# ---------- CAM16-UCS ----------
CAT16 = np.array([
 [ 0.401288, 0.650173, -0.051461],
 [-0.250268, 1.204414,  0.045854],
 [-0.002079, 0.048952,  0.953127]])
CAT16i = np.linalg.inv(CAT16)
UCS_c1 = 0.007; UCS_c2 = 0.0228
def cam16_vc(XYZ_w100, L_A, Y_b, F=1.0, c=0.69, N_c=1.0):
    RGB_w = mv(CAT16, XYZ_w100)
    D = np.clip(F*(1.0 - (1.0/3.6)*np.exp((-L_A - 42.0)/92.0)), 0.0, 1.0)
    Y_w = XYZ_w100[...,1]
    n = Y_b/Y_w
    k = 1.0/(5.0*L_A + 1.0); k4 = k**4
    F_L = 0.2*k4*(5.0*L_A) + 0.1*(1.0-k4)**2*spow(5.0*L_A, 1.0/3.0)
    N_bb = N_cb = 0.725*spow(1.0/n, 0.2)
    z = 1.48 + np.sqrt(n)
    D_RGB = D*Y_w/RGB_w + 1.0 - D
    RGB_aw = pa_fwd(D_RGB*RGB_w, F_L)
    A_w = (2.0*RGB_aw[...,0] + RGB_aw[...,1] + RGB_aw[...,2]/20.0 - 0.305)*N_bb
    return dict(D_RGB=D_RGB, n=n, F_L=F_L, N_bb=N_bb, N_cb=N_cb, z=z, A_w=A_w, c=c, N_c=N_c)
def pa_fwd(RGB, F_L):
    t = spow(F_L*np.abs(RGB)/100.0, 0.42)
    return (400.0*np.sign(RGB)*t)/(27.13 + t) + 0.1
def pa_inv(RGB, F_L):
    a = np.abs(RGB - 0.1)
    return np.sign(RGB - 0.1)*(100.0/F_L)*spow((27.13*a)/(400.0 - a), 1.0/0.42)
def xyz_to_cam16ucs(xyz_unit, XYZ_w_unit, L_A, Y_b):
    XYZ = xyz_unit*100.0; XYZ_w = XYZ_w_unit*100.0
    v = cam16_vc(XYZ_w, L_A, Y_b)
    RGB_a = pa_fwd(v["D_RGB"]*mv(CAT16, XYZ), v["F_L"])
    R, G, B = RGB_a[...,0], RGB_a[...,1], RGB_a[...,2]
    a = R - 12.0*G/11.0 + B/11.0
    b = (R + G - 2.0*B)/9.0
    h = np.degrees(np.arctan2(b, a)) % 360.0
    e_t = 0.25*(np.cos(2.0 + h*np.pi/180.0) + 3.8)
    A = (2.0*R + G + B/20.0 - 0.305)*v["N_bb"]
    J = 100.0*spow(A/v["A_w"], v["c"]*v["z"])
    t = ((50000.0/13.0)*v["N_c"]*v["N_cb"])*(e_t*spow(a*a + b*b, 0.5))/(R + G + 21.0*B/20.0)
    C = spow(t, 0.9)*spow(J/100.0, 0.5)*spow(1.64 - 0.29**v["n"], 0.73)
    M = C*spow(v["F_L"], 0.25)
    J_p = ((1.0 + 100.0*UCS_c1)*J)/(1.0 + UCS_c1*J)
    M_p = (1.0/UCS_c2)*np.log1p(UCS_c2*M)
    return np.stack([J_p, M_p*np.cos(np.radians(h)), M_p*np.sin(np.radians(h))], -1)
def cam16ucs_to_xyz(jab, XYZ_w_unit, L_A, Y_b):
    XYZ_w = XYZ_w_unit*100.0
    v = cam16_vc(XYZ_w, L_A, Y_b)
    J_p, a_p, b_p = jab[...,0], jab[...,1], jab[...,2]
    J = -J_p/(UCS_c1*J_p - 1.0 - 100.0*UCS_c1)
    M_p = np.hypot(a_p, b_p)
    h = np.degrees(np.arctan2(b_p, a_p)) % 360.0
    M = np.expm1(M_p*UCS_c2)/UCS_c2
    C = M/spow(v["F_L"], 0.25)
    J_pr = np.maximum(J, 2.2204460492503131e-16)
    t = spow(C/(np.sqrt(J_pr/100.0)*spow(1.64 - 0.29**v["n"], 0.73)), 1.0/0.9)
    e_t = 0.25*(np.cos(2.0 + h*np.pi/180.0) + 3.8)
    A = v["A_w"]*spow(J/100.0, 1.0/(v["c"]*v["z"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        P_1 = np.nan_to_num((50000.0/13.0)*v["N_c"]*v["N_cb"]*e_t/t, nan=0, posinf=0, neginf=0)
    P_2 = A/v["N_bb"] + 0.305
    P_3 = 21.0/20.0
    hr = np.radians(h); sin_hr = np.sin(hr); cos_hr = np.cos(hr)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.nan_to_num(cos_hr/sin_hr, nan=0, posinf=0, neginf=0)
        sc = np.nan_to_num(sin_hr/cos_hr, nan=0, posinf=0, neginf=0)
        P_4 = np.nan_to_num(P_1/sin_hr, nan=0, posinf=0, neginf=0)
        P_5 = np.nan_to_num(P_1/cos_hr, nan=0, posinf=0, neginf=0)
    nn = P_2*(2.0 + P_3)*(460.0/1403.0)
    use_b = np.abs(sin_hr) >= np.abs(cos_hr)
    bb = nn/(P_4 + (2.0+P_3)*(220.0/1403.0)*cs - (27.0/1403.0) + P_3*(6300.0/1403.0))
    aa1 = bb*cs
    aa = nn/(P_5 + (2.0+P_3)*(220.0/1403.0) - ((27.0/1403.0) - P_3*(6300.0/1403.0))*sc)
    bb2 = aa*sc
    a = np.where(use_b, aa1, aa); b = np.where(use_b, bb, bb2)
    zero = (t == 0); a = np.where(zero, 0.0, a); b = np.where(zero, 0.0, b)
    Mx = np.array([[460.0, 451.0, 288.0], [460.0, -891.0, -261.0], [460.0, -220.0, -6300.0]])/1403.0
    RGB_a = mv(Mx, np.stack([P_2, a, b], -1))
    RGB = pa_inv(RGB_a, v["F_L"])/v["D_RGB"]
    return mv(CAT16i, RGB)/100.0

# --------------------------------------------------------------------------
# Verification against colour-science. Requires the pinned oracle venv.
# --------------------------------------------------------------------------


def _verify() -> int:
    import colour

    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.001, 1.6, size=(50000, 3))
    XYZ_w = np.array([0.9504559270516716, 1.0, 1.0890577507598784])

    failures = 0
    def check(label, mine, ref, tol):
        nonlocal failures
        err = float(np.nanmax(np.abs(mine - ref)))
        ok = err <= tol
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label:26s} max|diff| = {err:.3e}  (tol {tol:.0e})")

    check("Oklab forward", xyz_to_oklab(xyz), np.asarray(colour.XYZ_to_Oklab(xyz)), 0.0)
    lab = np.asarray(colour.XYZ_to_Oklab(xyz))
    check("Oklab inverse", oklab_to_xyz(lab), np.asarray(colour.Oklab_to_XYZ(lab)), 0.0)

    check("JzAzBz forward", xyz_to_jzazbz(xyz * 100), np.asarray(colour.XYZ_to_Jzazbz(xyz * 100)), 0.0)
    jab = np.asarray(colour.XYZ_to_Jzazbz(xyz * 100))
    check("JzAzBz inverse", jzazbz_to_xyz(jab), np.asarray(colour.Jzazbz_to_XYZ(jab)), 0.0)

    ref = np.asarray(colour.XYZ_to_CAM16UCS(xyz, XYZ_w=XYZ_w, L_A=64.0, Y_b=20.0))
    check("CAM16-UCS forward", xyz_to_cam16ucs(xyz, XYZ_w, 64.0, 20.0), ref, 1e-12)

    # CAM16's forward emits NaN for negative-luminance XYZ; the reference pipeline
    # guarantees Y > 0, so the inverse is only compared on the finite rows.
    finite = np.all(np.isfinite(ref), axis=-1)
    check(
        "CAM16-UCS inverse",
        cam16ucs_to_xyz(ref[finite], XYZ_w, 64.0, 20.0),
        np.asarray(colour.CAM16UCS_to_XYZ(ref[finite], XYZ_w=XYZ_w, L_A=64.0, Y_b=20.0)),
        1e-12,
    )
    print(f"\n{'all transforms match' if not failures else f'{failures} transform(s) drifted'}")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(_verify())
