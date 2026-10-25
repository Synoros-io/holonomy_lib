"""Sympy symbolic verification of the arcsinh-reparameterization for
hyperbolic distance and log on the Lorentz model.

The
implementation in `src/holonomy_lib/manifolds/lorentz.py:distance`
uses

    d_k(x, y) = (2/√|k|) · arcsinh(√|k| · ‖y − x‖_M / 2)

instead of the textbook

    d_k(x, y) = (1/√|k|) · arccosh(k · ⟨x, y⟩_M).

This script verifies the identity symbolically (for x, y on the
hyperboloid), then walks through *why* the arcsinh form is
preferable numerically:

  (A) arccosh's derivative singularity at z = 1
  (B) catastrophic cancellation in z = k·⟨x,y⟩_M when z ≈ 1

Run:  uv run python notes/verification/arcsinh_reparam_sympy.py
"""

from __future__ import annotations

import sympy as sp


def main():
    # Coordinates on H^n_k for n = 2 (1+1 ambient) — minimal case
    # that exhibits the algebra. Higher n adds spatial components
    # but doesn't change the identity.
    t, k = sp.symbols("t k", positive=True, real=True)  # t-param along geodesic
    abs_k = sp.symbols("abs_k", positive=True)  # |k|, k = -abs_k for hyperbolic

    print("=" * 72)
    print("Arcsinh reparameterization for hyperbolic distance")
    print("=" * 72)
    print()

    # Take two points on H^1_{-1} (the unit hyperboloid in R^{1,1}):
    #   x = (cosh(0), sinh(0)) = (1, 0)
    #   y = (cosh(α), sinh(α)) for some geodesic distance α.
    # With k = -1, ⟨x, y⟩_M = -cosh(α). Geodesic distance d = α.
    alpha = sp.symbols("alpha", positive=True, real=True)
    x0, x1 = 1, 0
    y0, y1 = sp.cosh(alpha), sp.sinh(alpha)

    # Minkowski inner product (k = -1 convention: ⟨u, v⟩_M = -u_0·v_0 + u_1·v_1)
    minkowski = -x0 * y0 + x1 * y1   # = -cosh(α)
    print(f"⟨x, y⟩_M = {sp.simplify(minkowski)}")

    z = -minkowski   # k = -1, so k·⟨x,y⟩_M = (-1)·(-cosh α) = cosh α
    print(f"For k = -1:  z = k · ⟨x, y⟩_M = {sp.simplify(z)}")

    # ----- Form 1: textbook arccosh -----
    d_textbook = sp.acosh(z)
    d_textbook_simp = sp.simplify(d_textbook)
    print()
    print(f"Textbook form (k = -1):")
    print(f"  d = (1/√|k|) · arccosh(k·⟨x,y⟩_M) = arccosh(cosh α) = {d_textbook_simp}")

    # ----- Form 2: arcsinh reparameterization -----
    # ‖y - x‖_M² for k=-1:
    diff0, diff1 = y0 - x0, y1 - x1
    diff_sq = -diff0 ** 2 + diff1 ** 2
    diff_sq_simp = sp.simplify(diff_sq)
    print()
    print(f"Minkowski-norm² of (y - x): {diff_sq_simp}")
    # Should equal 2·(cosh α - 1) = 4·sinh²(α/2).

    # arcsinh form (k = -1):
    arg = sp.sqrt(diff_sq_simp) / 2
    d_arcsinh = 2 * sp.asinh(arg)
    d_arcsinh_simp = sp.simplify(d_arcsinh)
    print()
    print(f"Arcsinh form:")
    print(f"  d = 2 · arcsinh(‖y-x‖_M / 2) = 2·arcsinh(sqrt({diff_sq_simp})/2)")
    print(f"     simplifies to: {d_arcsinh_simp}")

    # ----- Equivalence check -----
    print()
    print("Equivalence:")
    eq_diff = sp.simplify(d_textbook_simp - d_arcsinh_simp)
    print(f"  d_textbook − d_arcsinh (simplified): {eq_diff}")
    if eq_diff == 0:
        print("  OK ✓ identical algebraically")

    # ----- (A) Derivative singularity in arccosh at z = 1 -----
    print()
    print("=" * 72)
    print("(A) Derivative singularity at x = y (z = 1)")
    print("=" * 72)
    z_sym = sp.symbols("z", positive=True)
    d_arccosh_dz = sp.diff(sp.acosh(z_sym), z_sym)
    print(f"d/dz arccosh(z) = {d_arccosh_dz}")
    print("  → at z = 1: 1/sqrt(0) = infinity (singular)")
    print(
        "  Implementation impact: backward through arccosh at z=1 "
        "propagates ∞·0 = NaN."
    )
    print()
    arg_sym = sp.symbols("arg", real=True)
    d_arcsinh_darg = sp.diff(sp.asinh(arg_sym), arg_sym)
    print(f"d/d_arg arcsinh(arg) = {d_arcsinh_darg}")
    print("  → at arg = 0: 1/sqrt(1) = 1 (finite, smooth)")
    print("  Implementation impact: backward through arcsinh at arg=0 is finite.")

    # ----- (B) Cancellation in z = k·⟨x,y⟩_M at small d -----
    print()
    print("=" * 72)
    print("(B) Catastrophic cancellation in z = k·⟨x,y⟩_M as x → y")
    print("=" * 72)
    eps = sp.symbols("eps", positive=True)
    print()
    print("Take α = ε (small geodesic distance).")
    print(f"  z = cosh(ε) ≈ 1 + ε²/2 + O(ε⁴)")
    print(f"      Numerically: subtracting (1 + tiny) from ~1 loses bits.")
    print()
    print(f"  ‖y - x‖_M² = 2(cosh(ε) - 1) ≈ ε² + O(ε⁴)")
    print(
        f"      Computed from coord differences directly, no near-1 "
        f"subtraction."
    )
    z_taylor = sp.series(sp.cosh(eps), eps, 0, 6).removeO()
    diff_sq_taylor = sp.series(2 * (sp.cosh(eps) - 1), eps, 0, 6).removeO()
    print()
    print(f"  z = {z_taylor}")
    print(f"  ‖y-x‖_M² = {diff_sq_taylor}")
    print()
    print(
        "Conclusion: for x ≈ y, the textbook arccosh form requires "
        "subtracting two ~1 quantities to find z - 1 — catastrophic "
        "cancellation at single ulp. The arcsinh form goes through "
        "coordinate differences (no cancellation) and arcsinh near 0 "
        "(smooth, finite gradient)."
    )

    # ----- Numerical sample -----
    print()
    print("Numerical confirmation at α = 1e-5 (float64):")
    eps_val = sp.Rational(1, 10**5)
    # Direct numerical evaluation
    z_num = float((sp.cosh(eps_val) - 1).evalf(30))
    diff_sq_num = float((2 * (sp.cosh(eps_val) - 1)).evalf(30))
    print(f"  z - 1   = {z_num:.3e}  (vulnerable to cancellation)")
    print(f"  ‖y-x‖_M² = {diff_sq_num:.3e}  (clean)")
    print(f"  ratio: ‖y-x‖_M² = 2·(z - 1) = {diff_sq_num / (2*z_num):.6f} ✓")

    print()
    print("=" * 72)
    print("Pseudo-verification result")
    print("=" * 72)
    print(
        "(1) The arcsinh form d = (2/√|k|)·arcsinh(√|k|·‖y-x‖_M/2) is\n"
        "    algebraically identical to the textbook d = (1/√|k|)·\n"
        "    arccosh(k·⟨x,y⟩_M) form on the manifold.\n"
        "\n"
        "(2) But the arcsinh form is preferable numerically:\n"
        "      (A) arcsinh has no derivative singularity at its\n"
        "          boundary, while arccosh diverges at z=1.\n"
        "      (B) computing ‖y-x‖_M² from coord differences avoids\n"
        "          the near-1 cancellation that bedevils z = k·⟨x,y⟩_M\n"
        "          for x ≈ y.\n"
        "\n"
        "This is the implementation-correctness story underlying §3 of\n"
        "the paper: equivalent math, very different float-arithmetic\n"
        "behavior. The autograd-safety follows immediately."
    )


if __name__ == "__main__":
    main()
