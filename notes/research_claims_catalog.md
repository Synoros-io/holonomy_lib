# Research claims catalog

Research claims first collected for v0.5.0, reviewed against the 0.8.0
implementation on 2026-10-25. The status fields below supersede the earlier
blanket “paper-ready” labels. Historical strengthening and validation records
remain evidence about their recorded implementations, not release-wide
correctness or performance guarantees.

Each entry separates the mathematical statement, current implementation,
evidence, and remaining work. Finding IDs link to the 0.8.0 changelog; the
review reports themselves are local session artifacts.

## C1. Heat-kernel recursion and the spectral-shift factor

**Claim:** The published dimension-raising recursion is
`k^{n+2}(t, r) = -exp(-n*t) / (2*pi*sinh(r)) * partial_r k^n(t, r)`.
Grigor'yan–Noguchi (1998) state the correct factor. Omitting `exp(-n*t)`
was our implementation error, not an error in that published formula and
not a limitation imposed by autograd. Differentiating in radius does not
supply a multiplicative time-dependent factor omitted from the program.

**Current evidence:**

- [Implementation](../src/holonomy_lib/hyperbolic/heat_kernel.py).
- [Symbolic verification](verification/heat_kernel_recursion_sympy.py) and
  [historical derivation](strengthening/C1_C2_heat_kernel_strengthened.md).
- [B04](../CHANGELOG.md#v080-b04), [B04b](../CHANGELOG.md#v080-b04b),
  [B05](../CHANGELOG.md#v080-b05), and [B06](../CHANGELOG.md#v080-b06)
  compare repaired quadrature, batch behavior, and small-radius values
  against independent references. The H2 float32 origin value changes
  from `1.6393e9` to `0.0575358`.

**Status (0.8.0): formula verified; implementation repaired and checked on
stated cases.** Earlier residual and mass checks did not cover the failures
found in B04–B06. The current paths preserve time derivatives, distinguish
batch members, and run under `torch.no_grad()`. Those checks do not establish
accuracy at arbitrary dimensions, extreme times, or unrepresentable radii.

**Remaining work:** Any broader accuracy or performance claim needs new
measurements for the specified regime. The regular H2 quadrature endpoint
is `d + sqrt(4*tail_budget*t)`; small-radius descent uses
`sqrt(d**2 + 4*tail_budget*t)`. The tail budget is not a uniform relative-error
bound.

## C2. Closed-form H5 heat kernel

**Claim:** Applying the dimension-raising operator to H3 gives

```text
k^5_t(r) = (4*pi*t)^(-5/2) * exp(-4*t - r^2/(4*t))
           * [r^2*sinh(r) + 2*t*(r*cosh(r) - sinh(r))] / sinh(r)^3.
```

This is one H3-to-H5 recursion step; the equivalent Gaussian operator-chain
representation has two radial operators. The displayed expression is an
algebraic identity, not a recommendation to evaluate the cancellations
literally near zero.

**Current evidence:**

- [Implementation](../src/holonomy_lib/hyperbolic/heat_kernel.py), including
  `_heat_kernel_unit_n5`, `_heat_kernel_unit_n7`, and the odd radial series.
- [Symbolic H5 verification](verification/heat_kernel_n5_sympy.py).
- [B06](../CHANGELOG.md#v080-b06): H7 at `(t=1, d=1e-4)` changes from
  `2.50596e-7` to `7.13434e-8` in float64; independent high-precision
  comparisons also cover sampled small-radius H5/H7 values.

**Status (0.8.0): closed form retained; stable evaluation repaired.**
Factored hyperbolic functions and a small-radius series replace unstable
cancellations. H7 also has a closed form. Higher dimensions use the
spectral-shifted recurrence, with a series/descent treatment near zero.

**Remaining work:** The earlier
[speed and precision comparisons](strengthening/C1_C2_heat_kernel_strengthened.md)
are historical. They do not justify a general “three orders more precise”
or current speed claim.

## C3. Arcsinh reparameterization for hyperbolic distance and logarithms

**Claim:** On valid hyperboloid points, the distance identity
`d_k(x,y) = 2/sqrt(abs(k)) * asinh(sqrt(abs(k))*norm_M(y-x)/2)`
avoids subtracting a quantity near one before applying `acosh`. `asinh` is
smooth on the real line, but this alone cannot prevent cancellation in an
ambient Minkowski norm or recover information lost in floating-point inputs.
Distance is not differentiable at coincidence; the implementation chooses
a zero subgradient there.

**Current evidence:**

- [Lorentz implementation](../src/holonomy_lib/manifolds/lorentz.py),
  specifically `distance`, `log`, and `log_0`.
- [B13](../CHANGELOG.md#v080-b13): float32 radius-20 distance changes from
  `0` to `20`; the origin logarithm changes from `[242582592, 0]` to `[20, 0]`.
- [Historical symbolic identity](verification/arcsinh_reparam_sympy.py),
  [near-origin worked example](strengthening/C3_arcsinh_worked_example_results.md),
  and [earlier analysis](strengthening/C3_arcsinh_strengthened.md).

**Status (0.8.0): earlier implementation claim superseded.** Distance now
uses a spatial radial decomposition rather than subtracting temporal and
spatial squares of ambient differences. Logarithm tangents are reconstructed
using their tangency constraint. The old assertion “finite at every input”
was false: the identity does not guarantee representable coordinates,
adequate angular precision, or finite values outside the supported domain.

**Remaining work:** State coordinate range and precision when making
stability claims; the historical near-origin comparison does not certify
large-radius or arbitrary-input behavior.

## C4. Learnable curvature crossing zero

**Claim:** Tensor-valued κ can select the current spherical, Euclidean, or
hyperbolic chart during evaluation instead of fixing the branch at
construction. Near zero, the implementation uses a signed expansion in
`q = kappa * alpha**2` for `abs(q) <= sqrt(eps(dtype))`; inactive branch
arguments are kept finite. Clamping `abs(kappa)` to `finfo.tiny` alone did
not implement the zero-curvature derivative correctly.

**Current evidence:**

- [Stereographic implementation](../src/holonomy_lib/manifolds/stereographic.py)
  and [generalized sine](../src/holonomy_lib/manifolds/comparison.py).
- [B10](../CHANGELOG.md#v080-b10): a manifold constructed at tensor κ=0 and
  changed to -1 now maps `[0.5, 0]` to `[0.462117, 0]`.
- [B10/B11 derivatives](../CHANGELOG.md#v080-curvature-derivatives): the
  first-coordinate exp/log derivatives at κ=0 and input `[0.5, 0]` are
  `+1/24` and `-1/24`; the generalized sine derivative is `-1/6` at `t=1`
  (in general, `partial_kappa sn_0(t) = -t**3/6`). A reported training trajectory
  `+0.5 → -0.000964363664 → -0.5` had 64 finite gradients.

**Status (0.8.0): implementation corrected; derivatives at zero curvature and a training
crossing checked numerically.** Previous claims overstated the shipped
behavior: construction at zero could freeze the Euclidean branch, and
inactive branches could poison gradients. Crossing zero now works in the
checked domain; this is not unrestricted optimization across chart poles
or saturation limits.

**Remaining work:** The
[old strengthening results](strengthening/C4_kappa_crossing_strengthened.md)
and [symbolic continuation](verification/kappa_crossing_sympy.py) are
historical evidence. Earlier rankings of dispatch against Taylor evaluation
are not current: the repaired implementation uses Taylor evaluation itself.

## C5. Per-point curvature

**Claim:** `holonomy_lib.experimental.HeterogeneousKappaManifold` supplies
constant-curvature origin maps with a curvature tensor per point. Pairwise
`dissimilarity` uses a combined endpoint curvature. This does not integrate
a curvature field or define geodesics on a variable-curvature manifold.
Curvatures and coordinates must satisfy the selected chart domains.

**Current evidence:**

- [Experimental implementation](../src/holonomy_lib/experimental/heterogeneous_kappa.py).
- [B08/B17](../CHANGELOG.md#v080-heterogeneous): both endpoint domains are
  validated, and the negative harmonic-mean bug is repaired.
- [Historical synthetic task](strengthening/C5_C6_synthetic_task_results.md)
  and [analysis](strengthening/C5_C6_heterogeneous_kappa_strengthened.md).

**Status (0.8.0): experimental, in progress.** The pair values are not a
metric: distinct points can compare as zero and triangle inequalities can
fail. The API was renamed from `distance` to `dissimilarity` to expose that
limitation. Differentiable coordinate maps do not establish a metric.

**Remaining work:** Historical synthetic-task comparisons describe an older
implementation. They do not establish a variable-curvature geometry or
superiority over discrete gating; neither is a current library claim.

## C6. Pair-curvature combiner abstraction

**Claim:** Arithmetic mean, signed harmonic mean, and a callable override
choose the effective curvature used by the experimental pair comparison.
This is a modeling choice, not a metric construction established by the
constant-curvature references.

**Current evidence:**

- [Combiner implementation](../src/holonomy_lib/experimental/heterogeneous_kappa.py).
- [B17](../CHANGELOG.md#v080-heterogeneous): harmonic mean of `(-1, -1)`
  changes from `2.2250738585072014e-308` to `-1`.
- Homogeneous agreement with the ordinary stereographic distance excludes
  `(0, 0)` for the harmonic combiner: a zero sum raises `ValueError`.
  One zero operand and one nonzero operand give zero.

**Status (0.8.0): experimental modeling choice; metric interpretation
unresolved.** The old positive-clamp implementation destroyed signs.
Correcting it does not restore the metric axioms of the pair construction.

**Remaining work:** Earlier
[combiner rankings](strengthening/C5_C6_heterogeneous_kappa_strengthened.md)
used the faulty implementation. Any empirical ranking needs new evidence
and an explicit task; no combiner is claimed to recover a curvature field.

## C7. Complementary numerical validation

**Claim:** Differential-equation residuals and normalization check different
properties. A constant rescale of a heat kernel can preserve the homogeneous
PDE while changing its mass; a mass-preserving perturbation can violate the
PDE. Neither check alone is sufficient.

**Current evidence:**

- [Heat-kernel validation](validation/heat_kernel_validation.py).
- [Historical methodology analysis](strengthening/C7_validation_methodology_strengthened.md).
- [Graph heat-kernel validation](validation/graph_heat_kernel_validation.py).
- [B04–B06](../CHANGELOG.md#v080-b04): numerical failures missed by the
  earlier validation, despite correct formulas and favorable sampled residuals.

**Status (0.8.0): useful complementary checks; insufficient as a correctness
certificate.** Pair them with independent reference values, distinct-member
batch correspondence, and derivatives in each supported argument. Agreement
on sampled inputs is evidence for those inputs, not every implementation path.

**Remaining work:** No blanket paper-readiness or release-certification claim.
The 0.8.0 review concerns a usable, numerically checked library; future
research claims require an explicit question and evidence for its scope.
