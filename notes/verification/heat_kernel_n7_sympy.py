"""Sympy derivation + verification of the closed-form n=7 heat kernel.

Extends the n=5 closed form one odd dimension higher. The H^7 kernel is
one corrected operator-chain recursion step from the (verified) H^5
closed form,

    k^7 = -exp(-5t) / (2π sinh r) · ∂_r k^5 ,

and equivalently the Grigor'yan operator chain at m=3. We:

  (A) derive k^7 from k^5 and extract the polynomial numerator P_7 with
          k^7_t(r) = (4πt)^{-7/2} · exp(-9t - r²/4t) · P_7 / sinh^7 r ;
  (B) confirm it equals the operator-chain form (m=3) — numerically at
      sample points (symbolic simplify of the triple operator chain is
      intractable; numerical agreement to ~1e-10 across a grid is the
      practical check);
  (C) confirm it satisfies the H^7 radial heat equation
          ∂_t k − ∂_r² k − 6·coth(r)·∂_r k = 0  — numerically;
  (D) report the r → 0 limit numerically.

Run:  uv run python -u notes/verification/heat_kernel_n7_sympy.py
"""

from __future__ import annotations

import sympy as sp


def main():
    t, r = sp.symbols("t r", positive=True, real=True)
    print("n=7 heat kernel: derivation + verification", flush=True)

    # ---- verified H^5 closed form ----
    k5 = (
        (4 * sp.pi * t) ** sp.Rational(-5, 2)
        * sp.exp(-4 * t - r ** 2 / (4 * t))
        * (r ** 2 * sp.sinh(r) + 2 * t * (r * sp.cosh(r) - sp.sinh(r)))
        / sp.sinh(r) ** 3
    )

    # ---- (A) k^7 by one corrected recursion step from k^5 ----
    k7 = -sp.exp(-5 * t) / (2 * sp.pi * sp.sinh(r)) * sp.diff(k5, r)

    # extract numerator over sinh^7 r:  P_7 = k7 · sinh^7 r / base
    base = (4 * sp.pi * t) ** sp.Rational(-7, 2) * sp.exp(-9 * t - r ** 2 / (4 * t))
    P7 = sp.simplify(sp.expand(sp.cancel(k7 * sp.sinh(r) ** 7 / base)))
    print("\n(A) numerator P_7 such that k^7 = (4πt)^{-7/2} e^{-9t-r²/4t} P_7 / sinh^7 r:",
          flush=True)
    print("  P_7 =", P7, flush=True)
    print("  collect in t:", sp.collect(sp.expand(P7), t), flush=True)

    # closed form rebuilt from the extracted numerator (what we implement)
    k7_closed = base * P7 / sp.sinh(r) ** 7

    # ---- (B) numerical agreement with the operator chain (m=3) ----
    def operator_chain(m):
        f = sp.exp(-(m ** 2) * t - r ** 2 / (4 * t))
        for _ in range(m):
            f = sp.diff(f, r) / sp.sinh(r)
        return sp.Integer(-1) ** m / (2 * sp.pi) ** m * (4 * sp.pi * t) ** sp.Rational(-1, 2) * f

    k7_op = operator_chain(3)
    fk_closed = sp.lambdify((t, r), k7_closed, "mpmath")
    fk_recur = sp.lambdify((t, r), k7, "mpmath")
    fk_op = sp.lambdify((t, r), k7_op, "mpmath")
    print("\n(B) numerical check: extracted closed form vs recursion vs operator chain",
          flush=True)
    pts = [(0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 0.3), (0.7, 3.0)]
    maxrel = 0.0
    for tv, rv in pts:
        a, b, c = float(fk_closed(tv, rv)), float(fk_recur(tv, rv)), float(fk_op(tv, rv))
        rel = max(abs(a - b), abs(a - c)) / abs(a)
        maxrel = max(maxrel, rel)
        print(f"  (t={tv}, r={rv}): closed={a:.10e} recur={b:.10e} op={c:.10e} rel={rel:.2e}",
              flush=True)
    print(f"  max relative disagreement: {maxrel:.2e}", "OK" if maxrel < 1e-9 else "CHECK",
          flush=True)

    # ---- (C) numerical H^7 heat-equation residual ----
    resid = (sp.diff(k7_closed, t) - sp.diff(k7_closed, r, 2)
             - 6 * sp.cosh(r) / sp.sinh(r) * sp.diff(k7_closed, r))
    fr = sp.lambdify((t, r), resid, "mpmath")
    print("\n(C) H^7 heat-equation residual ∂_t k − ∂_r²k − 6 coth(r) ∂_r k:", flush=True)
    maxres = 0.0
    for tv, rv in pts:
        val = abs(float(fr(tv, rv)))
        scale = abs(float(fk_closed(tv, rv))) / max(tv, 1.0)
        maxres = max(maxres, val / max(scale, 1e-300))
        print(f"  (t={tv}, r={rv}): |residual|={val:.3e}  (rel {val/max(scale,1e-300):.2e})",
              flush=True)
    print(f"  max relative residual: {maxres:.2e}", "OK" if maxres < 1e-8 else "CHECK",
          flush=True)

    # ---- (D) r → 0 limit: bracket B/sinh^5 r → 1 + 2t + 16t²/15 ----
    # (Evaluated at high precision: at tiny r the bracket suffers
    # catastrophic cancellation — B ~ O(r^5) from O(r^3) terms — so float
    # probes are unreliable; mpmath at high dps confirms the analytic limit.)
    import mpmath as mp
    mp.mp.dps = 50
    bracket = (r ** 3 * sp.sinh(r) ** 2 + 6 * r ** 2 * t * sp.sinh(r) * sp.cosh(r)
               + (8 * t ** 2 - 6 * t) * r * sp.sinh(r) ** 2
               + 12 * t ** 2 * (r - sp.sinh(r) * sp.cosh(r))) / sp.sinh(r) ** 5
    fb = sp.lambdify((t, r), bracket, "mpmath")
    print("\n(D) r → 0 bracket limit vs analytic 1 + 2t + 16t²/15:", flush=True)
    for tv in (0.25, 1.0, 2.0):
        num = fb(mp.mpf(tv), mp.mpf("1e-6"))
        ana = 1 + 2 * mp.mpf(tv) + mp.mpf(16) / 15 * mp.mpf(tv) ** 2
        print(f"  t={tv}: bracket(r=1e-6)={mp.nstr(num, 16)}  analytic={mp.nstr(ana, 16)}",
              flush=True)
    print("  => k^7_t(0) = (4πt)^{-7/2} e^{-9t} (1 + 2t + 16t²/15)", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
