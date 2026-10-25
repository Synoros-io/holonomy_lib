# holonomy_lib contents

Public API inventory for 0.8.0: signatures, one-line summaries, and citations.
Exports follow each subpackage’s `__all__`, including exported modules and public
class methods/properties. Signatures are read with `inspect.signature`; defaults
below are resolved values. Names in a section are relative to its module.

Dense numerical APIs use a leading batch dimension `B`; single-instance sparse
structures and backend exceptions are noted per section. Supported dtypes depend
on the underlying operation. See [README.md](README.md) for a worked example and
[CONVENTIONS.md](CONVENTIONS.md) for numerical and development standards.

## `holonomy_lib.manifolds`

Points carry a leading batch dimension; constructors choose device and dtype. Fixed-rank points are `(U, S, Vh)`, with `Vh` the conjugate transpose of the right singular vectors. Every manifold in this subpackage exposes `egrad_to_rgrad`; projection alone is not gradient conversion for a non-Euclidean metric. SPD spectral functions and differentiated fixed-rank retractions support first-order derivatives; individual SVD factors are nonunique at repeated singular values.

| Signature | Summary | Citation |
|---|---|---|
| `model_anisotropic_flux(kappas: 'torch.Tensor', r: 'torch.Tensor') -> 'torch.Tensor'` | Model flux `product_i sn_{kappa_i}(r)` from per-direction constant-curvature factors. | Petersen, P. (2016). Riemannian Geometry, 3rd ed. Springer, §7.1; the `sn_κ` volume element, taken here per principal direction (product over the anisotropic frame). |
| `model_ball_volume(kappa: 'torch.Tensor', N: 'torch.Tensor', r: 'torch.Tensor', n_quad: 'int' = 64) -> 'torch.Tensor'` | Volume of the geodesic ball of radius `r` in the space form `M_κ^N`: `V = ω_{N-1} · ∫_0^r sn_κ(t)^{N-1} dt`. | Sturm (2006), Acta Math 196, model-space volume; Petersen (2016) §7.1. |
| `model_sphere_area(kappa: 'torch.Tensor', N: 'torch.Tensor', r: 'torch.Tensor') -> 'torch.Tensor'` | Surface area of the geodesic sphere of radius `r` in the space form `M_κ^N`: `S = ω_{N-1} · sn_κ(r)^{N-1}`. | Dai-Wei (2019), §1.2 eq. (1.2.5). |
| `sn_kappa(kappa: 'torch.Tensor', t: 'torch.Tensor') -> 'torch.Tensor'` | Generalized sine, differentiable in signed curvature through zero. | Petersen (2016), Riemannian Geometry, §7.1, model-space Jacobi fields. |

### `FixedRankManifold`

| Signature | Summary | Citation |
|---|---|---|
| `FixedRankManifold(m: 'int', n: 'int', r: 'int', device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64, retraction_mode: "Literal['auto', 'exact', 'randomized']" = 'auto')` | Fixed-rank (m, n) matrix manifold of rank r, batched. | Vandereycken (2013), §2. |
| `FixedRankManifold.dense(point: 'FixedRankPoint') -> 'torch.Tensor'` | Reconstruct ambient dense matrices: M = U @ diag(S) @ Vh. | Vandereycken (2013), Low-Rank Matrix Completion by Riemannian Optimization, §2, embedded matrix representation and metric. |
| `FixedRankManifold.dim (property) -> 'int'` | Manifold dimension, r·(m + n − r). | Vandereycken (2013), §2. |
| `FixedRankManifold.egrad_to_rgrad(point: 'FixedRankPoint', egrad: 'torch.Tensor') -> 'torch.Tensor'` | Project an ambient gradient under the embedded Frobenius metric. | Absil, Mahony, Sepulchre (2008), §3.6; Vandereycken (2013), eq. 2.5. |
| `FixedRankManifold.inner(point: 'FixedRankPoint', tangent_a: 'torch.Tensor', tangent_b: 'torch.Tensor') -> 'torch.Tensor'` | Riemannian inner product of two tangents at point. | Vandereycken (2013), §2.2; induced metric. |
| `FixedRankManifold.norm(point: 'FixedRankPoint', tangent: 'torch.Tensor') -> 'torch.Tensor'` | Riemannian norm sqrt(<v,v>) of a tangent vector. (B,)-shaped output. | Vandereycken (2013), Low-Rank Matrix Completion by Riemannian Optimization, §2, embedded matrix representation and metric. |
| `FixedRankManifold.projection(point: 'FixedRankPoint', Z: 'torch.Tensor') -> 'torch.Tensor'` | Project ambient direction Z onto the tangent space at point. | Vandereycken (2013), eq. 2.5. |
| `FixedRankManifold.random_point(batch_size: 'int' = 1, generator: 'torch.Generator \| None' = None) -> 'FixedRankPoint'` | Sample B random points on the manifold. | Mezzadri, F. (2007). How to generate random matrices from the classical compact groups. Notices of the AMS, 54(5):592–604. |
| `FixedRankManifold.retraction(point: 'FixedRankPoint', tangent: 'torch.Tensor') -> 'FixedRankPoint'` | Move along tangent and project back onto the manifold via SVD. | Absil, Mahony, Sepulchre (2008), §4.1; retractions. Vandereycken (2013), eq. 2.7. Halko, Martinsson, Tropp (2011); randomized SVD. |

### `KappaStereographicManifold`

| Signature | Summary | Citation |
|---|---|---|
| `KappaStereographicManifold(n: 'int', kappa: 'float \| torch.Tensor' = -1.0, device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64)` | Constant-curvature κ-stereographic manifold. | Bachmann, G., Bécigneul, G., Ganea, O.-E. (2020). Constant Curvature Graph Convolutional Networks. ICML. (the κ-stereographic formulation) Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. Ungar, A. A. (2008). A Gyrovector Space Approach to Hyperbolic Geometry. (gyro-addition / κ-trig algebra) Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). Hyperbolic Neural Networks. NeurIPS. (Poincaré ball special case) |
| `KappaStereographicManifold.ambient_dim (property) -> 'int'` | Ambient dimension `n`. | Bachmann, G., Bécigneul, G., Ganea, O.-E. (2020). Constant Curvature Graph Convolutional Networks. ICML. (the κ-stereographic formulation) Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. Ungar, A. A. (2008). A Gyrovector Space Approach to Hyperbolic Geometry. (gyro-addition / κ-trig algebra) Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). Hyperbolic Neural Networks. NeurIPS. (Poincaré ball special case) |
| `KappaStereographicManifold.dim (property) -> 'int'` | Intrinsic manifold dimension `n`. | Bachmann, G., Bécigneul, G., Ganea, O.-E. (2020). Constant Curvature Graph Convolutional Networks. ICML. (the κ-stereographic formulation) Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. Ungar, A. A. (2008). A Gyrovector Space Approach to Hyperbolic Geometry. (gyro-addition / κ-trig algebra) Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). Hyperbolic Neural Networks. NeurIPS. (Poincaré ball special case) |
| `KappaStereographicManifold.distance(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Geodesic distance `d_κ(x, y) = (2/√\|κ\|) · tan_κ⁻¹(√\|κ\| · ‖(-x) ⊕_κ y‖)`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.egrad_to_rgrad(point: 'torch.Tensor', egrad: 'torch.Tensor') -> 'torch.Tensor'` | Divide an ambient gradient by the squared conformal factor. | Absil, Mahony, Sepulchre (2008), §3.6; Bachmann et al. (2020), §2. |
| `KappaStereographicManifold.exp(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Exponential map at a general base point: | Bachmann et al. (2020), eq. 5. |
| `KappaStereographicManifold.exp_0(v: 'torch.Tensor') -> 'torch.Tensor'` | Exponential map at the origin: `exp_0(v) = tan_κ(√\|κ\|·‖v‖) · v / (√\|κ\|·‖v‖)`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.inner(x: 'torch.Tensor', u: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Riemannian inner product `λ_κ(x)² · ⟨u, v⟩_Eucl`. | Bachmann et al. (2020), §3. |
| `KappaStereographicManifold.is_on_manifold(x: 'torch.Tensor', atol: 'float' = 1e-09) -> 'torch.Tensor'` | Check finite coordinates and the hyperbolic open-ball boundary. | Bachmann et al. (2020), §2, stereographic domain. |
| `KappaStereographicManifold.log(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Logarithmic map at a general base point: | Bachmann et al. (2020), eq. 5. |
| `KappaStereographicManifold.log_0(y: 'torch.Tensor') -> 'torch.Tensor'` | Logarithmic map at the origin: `log_0(y) = (1/√\|κ\|) · tan_κ⁻¹(√\|κ\|·‖y‖) · y / ‖y‖`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.mobius_add(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Möbius (gyro-) addition `x ⊕_κ y` on the κ-stereographic manifold. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.norm(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Riemannian norm `λ_κ(x) · ‖v‖_Eucl`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.origin(batch_size: 'int' = 1) -> 'torch.Tensor'` | The origin `(0, …, 0) ∈ R^n`; common base point for tangent-space embeddings. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.parallel_transport(x: 'torch.Tensor', y: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Parallel transport on the κ-stereographic manifold: | Bachmann et al. (2020), §2, κ-stereographic geometry; Ungar (2008), A Gyrovector Space Approach to Hyperbolic Geometry, gyration. The linear formula agrees with Geoopt stereographic.math. |
| `KappaStereographicManifold.projection(x: 'torch.Tensor', w: 'torch.Tensor') -> 'torch.Tensor'` | Tangent-space projection. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |
| `KappaStereographicManifold.random_point(batch_size: 'int' = 1, generator: 'torch.Generator \| None' = None) -> 'torch.Tensor'` | Draw Gaussian origin tangents and map them with the current κ. | Bachmann et al. (2020), §2, origin exponential. |
| `KappaStereographicManifold.retraction(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Retraction = exponential map (second-order, the geodesic itself satisfies the retraction axioms on constant-curvature spaces). | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, stereographic metric and maps. |

### `LorentzManifold`

| Signature | Summary | Citation |
|---|---|---|
| `LorentzManifold(n: 'int', k: 'float' = -1.0, device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64)` | Lorentz (hyperboloid) model of hyperbolic space H^n_k. | Nickel, M., Kiela, D. (2018). Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. ICML. Chen, W., et al. (2022). Fully hyperbolic neural networks. ACL. Lee, J. M. (2018). Introduction to Riemannian Manifolds, 2nd ed., Springer, §5–§6 (model spaces, hyperbolic geometry). Cannon, J. W., Floyd, W. J., Kenyon, R., Parry, W. R. (1997). Hyperbolic geometry. In Flavors of Geometry, MSRI Publications 31. Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds. Journal of Mathematical Imaging and Vision 25(1):127–154. (parallel transport in terms of log/distance, eq. 4). |
| `LorentzManifold.ambient_dim (property) -> 'int'` | Ambient dimension `n + 1`. | Nickel, M., Kiela, D. (2018). Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. ICML. Chen, W., et al. (2022). Fully hyperbolic neural networks. ACL. Lee, J. M. (2018). Introduction to Riemannian Manifolds, 2nd ed., Springer, §5–§6 (model spaces, hyperbolic geometry). Cannon, J. W., Floyd, W. J., Kenyon, R., Parry, W. R. (1997). Hyperbolic geometry. In Flavors of Geometry, MSRI Publications 31. Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds. Journal of Mathematical Imaging and Vision 25(1):127–154. (parallel transport in terms of log/distance, eq. 4). |
| `LorentzManifold.dim (property) -> 'int'` | Intrinsic manifold dimension `n`. | Lee (2018), §3. |
| `LorentzManifold.distance(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Geodesic distance from a spatial radial decomposition. | Nickel and Kiela (2018), eq. 7, Lorentz distance; Olver (2010), NIST Handbook §4.37, hyperbolic addition and half-angle identities. |
| `LorentzManifold.egrad_to_rgrad(point: 'torch.Tensor', egrad: 'torch.Tensor') -> 'torch.Tensor'` | Apply the Minkowski metric inverse, then project to the tangent. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3, Riemannian optimization. Absil, Mahony, Sepulchre (2008), §3.6. |
| `LorentzManifold.exp(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Exponential map exp_x(v) = cosh(α)·x + sinh(α)·v/α, α = √\|k\| · ‖v‖_x. | Nickel & Kiela (2018), eq. 8; Lee (2018), §5. |
| `LorentzManifold.exp_0(v_spatial: 'torch.Tensor') -> 'torch.Tensor'` | Exponential map at the origin from an (n,)-dimensional Euclidean coordinate. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.inner(x: 'torch.Tensor', u: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Riemannian inner product ⟨u, v⟩_x = ⟨u, v⟩_M on T_x H^n_k. | Lee (2018), §5. |
| `LorentzManifold.is_on_manifold(x: 'torch.Tensor', atol: 'float' = 1e-09) -> 'torch.Tensor'` | Per-batch test for membership in H^n_k. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.log(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Logarithmic map log_x(y) = (α / sinh α) · u, u = y − k·⟨x,y⟩_M · x, α = arccosh(k·⟨x,y⟩_M). | Nickel & Kiela (2018), eq. 9; Lee (2018), §5. |
| `LorentzManifold.log_0(y: 'torch.Tensor') -> 'torch.Tensor'` | Logarithmic map at the origin to an (n,)-dimensional Euclidean coordinate. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.minkowski_inner(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Minkowski inner product ⟨x, y⟩_M = −x_0 y_0 + Σ_{i≥1} x_i y_i. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.norm(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Riemannian norm √⟨v, v⟩_x. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.origin(batch_size: 'int' = 1) -> 'torch.Tensor'` | The canonical "north pole" `(1/√\|k\|, 0, …, 0)` of H^n_k. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.parallel_transport(x: 'torch.Tensor', y: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Parallel transport v ∈ T_x H^n_k → T_y H^n_k along the geodesic from x to y. | Nickel & Kiela (2018), §3 (Lorentz geometry). Lee (2018), §4 (Levi-Civita connection, parallel transport). |
| `LorentzManifold.projection(x: 'torch.Tensor', w: 'torch.Tensor') -> 'torch.Tensor'` | Project ambient direction `w` onto T_x H^n_k. | Nickel & Kiela (2018), eq. 6. |
| `LorentzManifold.random_point(batch_size: 'int' = 1, generator: 'torch.Generator \| None' = None) -> 'torch.Tensor'` | Sample B random points on H^n_k. | Nickel and Kiela (2018), Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, §3. |
| `LorentzManifold.retraction(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Retraction = exponential map. | Absil, Mahony, Sepulchre (2008), §4.1. |

### `LorentzianManifold`

| Signature | Summary | Citation |
|---|---|---|
| `LorentzianManifold(n: 'int', device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64)` | Flat pseudo-Riemannian manifold `R^{1, n-1}`. | Misner, C. W., Thorne, K. S., Wheeler, J. A. (1973). *Gravitation*. W. H. Freeman. Chapters 1–5 (Lorentzian-geometry foundations). Hawking, S. W., Ellis, G. F. R. (1973). *The Large Scale Structure of Space-Time*. Cambridge University Press, Ch. 4 (causal structure). O'Neill, B. (1983). *Semi-Riemannian Geometry With Applications to Relativity*. Academic Press, Ch. 5 (formal pseudo-Riemannian geometry). |
| `LorentzianManifold.ambient_dim (property) -> 'int'` | Ambient dimension `n`. | Misner, C. W., Thorne, K. S., Wheeler, J. A. (1973). *Gravitation*. W. H. Freeman. Chapters 1–5 (Lorentzian-geometry foundations). Hawking, S. W., Ellis, G. F. R. (1973). *The Large Scale Structure of Space-Time*. Cambridge University Press, Ch. 4 (causal structure). O'Neill, B. (1983). *Semi-Riemannian Geometry With Applications to Relativity*. Academic Press, Ch. 5 (formal pseudo-Riemannian geometry). |
| `LorentzianManifold.causal_type(x: 'torch.Tensor', y: 'torch.Tensor', null_atol: 'float' = 1e-09) -> 'torch.Tensor'` | Causal classification of point pairs `(x_i, y_i)`. | Hawking-Ellis (1973) §4.1 (causal structure). O'Neill (1983) §5.1 (timelike / null / spacelike vectors). |
| `LorentzianManifold.christoffel_symbols(x: 'torch.Tensor') -> 'torch.Tensor'` | Christoffel symbols `Γ^σ_μν` of the Levi-Civita connection. | MTW (1973) §8.5 (Christoffel symbols in flat spacetime); O'Neill (1983) §3.5 (Levi-Civita connection on semi-Riemannian manifolds). |
| `LorentzianManifold.dim (property) -> 'int'` | Manifold dimension `n`. | Misner, C. W., Thorne, K. S., Wheeler, J. A. (1973). *Gravitation*. W. H. Freeman. Chapters 1–5 (Lorentzian-geometry foundations). Hawking, S. W., Ellis, G. F. R. (1973). *The Large Scale Structure of Space-Time*. Cambridge University Press, Ch. 4 (causal structure). O'Neill, B. (1983). *Semi-Riemannian Geometry With Applications to Relativity*. Academic Press, Ch. 5 (formal pseudo-Riemannian geometry). |
| `LorentzianManifold.egrad_to_rgrad(point: 'torch.Tensor', egrad: 'torch.Tensor') -> 'torch.Tensor'` | Metric dual of an ambient covector in flat spacetime. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, metric duality. |
| `LorentzianManifold.exp(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Exponential map. | O'Neill (1983) §3.5 (flat semi-Riemannian geodesics). |
| `LorentzianManifold.inner(x: 'torch.Tensor', u: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Pseudo-Riemannian inner product `⟨u, v⟩_x = ⟨u, v⟩_M`. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.interval_sq(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Signed squared spacetime interval `⟨y - x, y - x⟩_M`. | MTW (1973) §1.5; Hawking-Ellis (1973) §4.1. |
| `LorentzianManifold.is_on_manifold(x: 'torch.Tensor', atol: 'float' = 1e-09) -> 'torch.Tensor'` | Always True per-batch element; `R^n` has no constraint surface. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.log(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Logarithmic map = `y - x` (the geodesic from `x` to `y` is a straight line; its tangent at `x` is the difference vector). | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.metric_tensor(x: 'torch.Tensor') -> 'torch.Tensor'` | Minkowski metric `g_μν = diag(-1, +1, +1, …, +1)`. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.minkowski_inner(u: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Signed Minkowski inner product `⟨u, v⟩_M = -u_0·v_0 + Σ_{i≥1} u_i·v_i`. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.norm_sq(v: 'torch.Tensor') -> 'torch.Tensor'` | Signed `⟨v, v⟩_M`. | MTW (1973) §2.3 (Minkowski metric); O'Neill (1983) §3.1. |
| `LorentzianManifold.origin(batch_size: 'int' = 1) -> 'torch.Tensor'` | Origin `(0, 0, …, 0) ∈ R^{1, n-1}`; convention for the spacetime origin event. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.projection(x: 'torch.Tensor', w: 'torch.Tensor') -> 'torch.Tensor'` | Tangent-space projection. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.proper_distance(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Proper distance `s = √⟨y - x, y - x⟩_M` for spacelike- separated pairs. | MTW (1973) §1.4 (proper distance / spatial interval). |
| `LorentzianManifold.proper_time(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Proper time `τ = √(-⟨y - x, y - x⟩_M)` for timelike- separated pairs. | MTW (1973) §1.4 (proper time, "wristwatch time"). |
| `LorentzianManifold.random_point(batch_size: 'int' = 1, generator: 'torch.Generator \| None' = None) -> 'torch.Tensor'` | Sample B random points in spacetime; Gaussian in each of the `n` coordinates. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.retraction(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Retraction = exponential map for flat space. | O'Neill (1983), Semi-Riemannian Geometry, Ch. 3, flat metric and its Levi-Civita connection. |
| `LorentzianManifold.ricci_tensor(x: 'torch.Tensor') -> 'torch.Tensor'` | Ricci tensor `R_μν = R^σ_μσν`. | MTW (1973) §17.2. |
| `LorentzianManifold.riemann_tensor(x: 'torch.Tensor') -> 'torch.Tensor'` | Riemann curvature tensor `R^σ_ρμν`. | MTW (1973) §11.3 (Riemann tensor in flat spacetime); O'Neill (1983) §3.5 (curvature tensor formulas). |
| `LorentzianManifold.scalar_curvature(x: 'torch.Tensor') -> 'torch.Tensor'` | Ricci scalar `R = g^μν R_μν`. | MTW (1973) §17.4 (scalar curvature). |

### `ProductManifold`

| Signature | Summary | Citation |
|---|---|---|
| `ProductManifold(manifolds: 'Sequence', weights: 'Sequence[float] \| None' = None, device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64)` | Riemannian product of one or more manifolds. | Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. (mixed-curvature embedding for deep generative models) Gu, A., Sala, F., Gunel, B., Ré, C. (2019). Learning mixed- curvature representations in product spaces. ICLR. |
| `ProductManifold.ambient_dim (property) -> 'int'` | Ambient dimension Σ ambient_dim_i. | Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. (mixed-curvature embedding for deep generative models) Gu, A., Sala, F., Gunel, B., Ré, C. (2019). Learning mixed- curvature representations in product spaces. ICLR. |
| `ProductManifold.component(x: 'torch.Tensor', index: 'int') -> 'torch.Tensor'` | Extract the `index`-th submanifold's coordinates from a flat point tensor `x`. | Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. (mixed-curvature embedding for deep generative models) Gu, A., Sala, F., Gunel, B., Ré, C. (2019). Learning mixed- curvature representations in product spaces. ICLR. |
| `ProductManifold.dim (property) -> 'int'` | Intrinsic manifold dimension Σ dim_i. | Skopek, O., Ganea, O.-E., Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. ICLR. (mixed-curvature embedding for deep generative models) Gu, A., Sala, F., Gunel, B., Ré, C. (2019). Learning mixed- curvature representations in product spaces. ICLR. |
| `ProductManifold.distance(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Pythagorean: `d² = Σ_i w_i · d_i²(x_i, y_i)`. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.egrad_to_rgrad(point: 'torch.Tensor', egrad: 'torch.Tensor') -> 'torch.Tensor'` | Apply each component's metric inverse, including its weight. | Absil, Mahony, Sepulchre (2008), §3.6, direct-sum metric duality. |
| `ProductManifold.exp(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Per-component exponential map. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.exp_0(v: 'torch.Tensor') -> 'torch.Tensor'` | Embed a flat Euclidean tangent `(B, Σ dim_i)` onto the product manifold via per-submanifold `exp_0`. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.inner(x: 'torch.Tensor', u: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Weighted direct-sum inner product: `⟨u, v⟩_x = Σ_i w_i · ⟨u_i, v_i⟩_{x_i}`. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.is_on_manifold(x: 'torch.Tensor', atol: 'float' = 1e-09) -> 'torch.Tensor'` | A product point is on the manifold iff every component is on its submanifold. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.log(x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor'` | Per-component logarithmic map. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.log_0(y: 'torch.Tensor') -> 'torch.Tensor'` | Inverse of `exp_0`: flat-ambient point → flat-intrinsic tangent. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.norm(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | `sqrt(⟨v, v⟩_x)`. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.origin(batch_size: 'int' = 1) -> 'torch.Tensor'` | Concatenated origins of every submanifold. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.projection(x: 'torch.Tensor', w: 'torch.Tensor') -> 'torch.Tensor'` | Project the ambient direction `w` (flat) onto the tangent space at `x`, per-submanifold. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.random_point(batch_size: 'int' = 1, generator: 'torch.Generator \| None' = None) -> 'torch.Tensor'` | Sample by drawing independently on each submanifold and concatenating. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |
| `ProductManifold.retraction(x: 'torch.Tensor', v: 'torch.Tensor') -> 'torch.Tensor'` | Per-component retraction. | Lee (2018), Introduction to Riemannian Manifolds, product metrics. The component formulas are cited in their manifold methods. |

### `SPDManifold`

| Signature | Summary | Citation |
|---|---|---|
| `SPDManifold(n: 'int', device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64)` | Affine-invariant SPD(n) manifold, batched. | Pennec, X., Fillard, P., Ayache, N. (2006). A Riemannian framework for tensor computing. International Journal of Computer Vision, 66(1):41–66. Bhatia, R. (2007). Positive Definite Matrices. Princeton University Press, chapters 4–6. Sra, S., Hosseini, R. (2015). Conic geometric optimization on the manifold of positive definite matrices. SIAM J. Optim. 25(1):713–739. |
| `SPDManifold.dim (property) -> 'int'` | Manifold dimension n(n+1)/2. | Pennec et al. (2006), §3.1. |
| `SPDManifold.distance(S: 'torch.Tensor', T: 'torch.Tensor', whitening: 'tuple[torch.Tensor, torch.Tensor] \| None' = None) -> 'torch.Tensor'` | Geodesic distance d(S, T) = ‖log(S^{−1/2} T S^{−1/2})‖_F. | Pennec et al. (2006), eq. 7. |
| `SPDManifold.egrad_to_rgrad(point: 'torch.Tensor', egrad: 'torch.Tensor') -> 'torch.Tensor'` | Convert an ambient gradient G to S sym(G) S, shape (B, n, n). | Absil, Mahony, Sepulchre (2008), §3.6; Pennec et al. (2006), eq. 4. |
| `SPDManifold.exp(S: 'torch.Tensor', V: 'torch.Tensor', whitening: 'tuple[torch.Tensor, torch.Tensor] \| None' = None) -> 'torch.Tensor'` | Exponential map exp_S(V) = S^{1/2} expm(S^{−1/2} V S^{−1/2}) S^{1/2}. | Pennec et al. (2006), eq. 5. |
| `SPDManifold.inner(S: 'torch.Tensor', U: 'torch.Tensor', V: 'torch.Tensor') -> 'torch.Tensor'` | Affine-invariant inner product ⟨U, V⟩_S = tr(S⁻¹ U S⁻¹ V). | Pennec et al. (2006), eq. 4. |
| `SPDManifold.is_spd(S: 'torch.Tensor') -> 'torch.Tensor'` | Per-batch test for SPD: symmetric and all eigenvalues > 0. | Pennec, Fillard, Ayache (2006), A Riemannian Framework for Tensor Computing, §3, positive-definite matrices and affine-invariant geometry. |
| `SPDManifold.log(S: 'torch.Tensor', T: 'torch.Tensor', whitening: 'tuple[torch.Tensor, torch.Tensor] \| None' = None) -> 'torch.Tensor'` | Logarithm map log_S(T) = S^{1/2} logm(S^{−1/2} T S^{−1/2}) S^{1/2}. | Pennec et al. (2006), eq. 6. |
| `SPDManifold.norm(S: 'torch.Tensor', V: 'torch.Tensor') -> 'torch.Tensor'` | Frobenius norm of the Cholesky-whitened tangent, shape (B,). | Pennec et al. (2006), eq. 4. |
| `SPDManifold.precompute_whitening(S: 'torch.Tensor') -> 'tuple[torch.Tensor, torch.Tensor]'` | Compute (S^{1/2}, S^{-1/2}) once for reuse across geodesic ops. | Pennec, Fillard, Ayache (2006), A Riemannian Framework for Tensor Computing, §3, positive-definite matrices and affine-invariant geometry. |
| `SPDManifold.projection(S: 'torch.Tensor', Z: 'torch.Tensor') -> 'torch.Tensor'` | Project ambient direction Z onto the tangent space at S. | Bhatia (2007), §6.1. |
| `SPDManifold.random_point(batch_size: 'int' = 1, generator: 'torch.Generator \| None' = None) -> 'torch.Tensor'` | Sample B SPD matrices via the Wishart-like construction A Aᵀ. | Anderson, T. W. (2003). An Introduction to Multivariate Statistical Analysis, 3rd ed., chapter 7; Wishart distribution. |
| `SPDManifold.retraction(S: 'torch.Tensor', V: 'torch.Tensor', whitening: 'tuple[torch.Tensor, torch.Tensor] \| None' = None) -> 'torch.Tensor'` | Retraction = exponential map. | Absil-Mahony-Sepulchre (2008), §4.1; exponential as retraction. |

## `holonomy_lib.optimization`

Supply an ambient gradient. The step converts it with `egrad_to_rgrad`, scales by `-lr`, and retracts. `RiemannianOptimizer` is the abstract interface.

| Signature | Summary | Citation |
|---|---|---|
| `riemannian_sgd_step(manifold: 'Any', point: 'Any', ambient_grad: 'torch.Tensor', lr: 'float') -> 'Any'` | Functional one-step Riemannian SGD. | Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization Algorithms on Matrix Manifolds. Princeton University Press, §4.1; retraction-based steepest descent. Bonnabel, S. (2013). Stochastic gradient descent on Riemannian manifolds. IEEE TAC 58(9):2217-2229, §III. |

### `RiemannianOptimizer`

| Signature | Summary | Citation |
|---|---|---|
| `RiemannianOptimizer(manifold: 'Any') -> 'None'` | Abstract base for Riemannian optimizers. | Absil, Mahony, Sepulchre (2008), §4.1. |
| `RiemannianOptimizer.step(point: 'Any', ambient_grad: 'Any') -> 'Any'` | Apply one optimizer step and return the new point. | Bonnabel (2013), Stochastic Gradient Descent on Riemannian Manifolds, §III. |

### `RiemannianSGD`

| Signature | Summary | Citation |
|---|---|---|
| `RiemannianSGD(manifold: 'Any', lr: 'float' = 0.01) -> 'None'` | Stateful wrapper around `riemannian_sgd_step`. | Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization Algorithms on Matrix Manifolds. Princeton University Press, §4.1-§4.2. Bonnabel, S. (2013). Stochastic gradient descent on Riemannian manifolds. IEEE TAC 58(9):2217-2229. |
| `RiemannianSGD.step(point: 'Any', ambient_grad: 'Any') -> 'Any'` | Convert the ambient gradient and retract one descent step. | Bonnabel (2013), Stochastic Gradient Descent on Riemannian Manifolds, §III. |

## `holonomy_lib.hyperbolic`

Fréchet means accept vector-coordinate batches `(B, N, ambient_dim)`, use backtracking, and warn when the residual remains above tolerance. Heat-kernel times and distances broadcast: n=1 is Gaussian, n=2 uses quadrature, n=3/5/7 have closed forms, and other dimensions use recurrence. Near zero, odd dimensions use a series and even dimensions a descent integral. The quadrature budget is not a uniform relative-error guarantee.

| Signature | Summary | Citation |
|---|---|---|
| `frechet_mean(points: 'torch.Tensor', manifold: 'Any', weights: 'torch.Tensor \| None' = None, max_iter: 'int' = 100, tol: 'float' = 1e-09, *, sync_every: 'int' = 8, max_backtracks: 'int' = 16) -> 'torch.Tensor'` | Compute a Fréchet mean with backtracking Karcher descent. | Karcher (1977), Riemannian center of mass and mollifier smoothing. Afsari, Tron, Vidal (2013), On the convergence of gradient descent for finding the Riemannian center of mass, §§2–3. Absil, Mahony, Sepulchre (2008), Optimization Algorithms on Matrix Manifolds, §4.2, line-search methods. |
| `hyperbolic_heat_kernel(t: 'torch.Tensor', distances: 'torch.Tensor', manifold: '_HyperbolicManifold', n_quad: 'int' = 32, tail_budget: 'float' = 20.0) -> 'torch.Tensor'` | Heat kernel `k^n_t(d)` on the hyperbolic manifold `manifold`. | Davies–Mandouvalos (1988); Grigor'yan (2009), Theorem 8.21; Grigor'yan–Noguchi (1998); Anker–Ostellari (2003). |
| `manifold_aware_inner(x: 'torch.Tensor', y: 'torch.Tensor', manifold: 'Any') -> 'torch.Tensor'` | Riemannian inner product `⟨log_o(x), log_o(y)⟩_o` at the manifold origin. | Pennec (2006), §3, tangent-space statistics on Riemannian manifolds. |

## `holonomy_lib.lie`

SO(3) functions are under `lie.so3`, not directly under `lie`. Spherical harmonics support degrees through 4; consult the docstring for real-basis ordering and normalization.

| Signature | Summary | Citation |
|---|---|---|
| `real_spherical_harmonics(directions: 'torch.Tensor', l_max: 'int' = 2) -> 'torch.Tensor'` | Real spherical harmonics `Y_lm` at the given direction vectors. | Wikipedia, "Table of spherical harmonics": real form. Edmonds (1957), §2.5. |

### `holonomy_lib.lie.so3`

| Signature | Summary | Citation |
|---|---|---|
| `axis_angle_to_matrix(axis: 'torch.Tensor', angle: 'torch.Tensor') -> 'torch.Tensor'` | Rodrigues formula: convert (unit axis, angle) → rotation matrix. | Hall (2015), Theorem 3.3. |
| `matrix_to_axis_angle(R: 'torch.Tensor') -> 'tuple[torch.Tensor, torch.Tensor]'` | Inverse Rodrigues: convert SO(3) matrix to (unit axis, angle). | Murray-Li-Sastry (1994), §2.3.3: the dual-branch logarithm. |
| `so3_exp(omega: 'torch.Tensor') -> 'torch.Tensor'` | Matrix exponential on so(3): ω ∈ R³ ↦ R ∈ SO(3). | Hall (2015). Lie Groups, Lie Algebras, and Representations, §3.1. |
| `so3_log(R: 'torch.Tensor') -> 'torch.Tensor'` | Principal rotation vector of an SO(3) matrix, with norm in [0, π]. | Murray, Li, Sastry (1994). A Mathematical Introduction to Robotic Manipulation, §2.3.3, the logarithm of a rotation matrix. |
| `compose(R1: 'torch.Tensor', R2: 'torch.Tensor') -> 'torch.Tensor'` | SO(3) group product: `R1 · R2`. | Hall (2015). Lie Groups, Lie Algebras, and Representations, §1.2. |
| `random_so3(batch_size: 'int', generator: 'torch.Generator \| None' = None, device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64) -> 'torch.Tensor'` | Haar-uniform random rotation matrices. | Shoemake (1992). |

## `holonomy_lib.info_geometry`

Batched distributions, Fisher information, and tangent updates. Categorical inputs are probability vectors; `0 * log(0/q)` contributes zero.

| Signature | Summary | Citation |
|---|---|---|
| `bregman_divergence(p: 'torch.Tensor', q: 'torch.Tensor', potential: 'Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]') -> 'torch.Tensor'` | Bregman divergence `D_F(p ‖ q) = F(p) − F(q) − ⟨∇F(q), p − q⟩`. | Bregman (1967). Banerjee et al. (2005), §2. |
| `fisher_information_categorical(p: 'torch.Tensor') -> 'torch.Tensor'` | Fisher information metric on the simplex Δ^{k−1}. | Amari (2016), §2.4 example 4. Nielsen, F. (2020). An elementary introduction to information geometry. Entropy 22(10):1100. |
| `fisher_information_gaussian_mean(Sigma: 'torch.Tensor') -> 'torch.Tensor'` | Fisher information for the mean μ of N(μ, Σ), at fixed Σ. | Amari (2016), §2.4 example 5. Petersen-Pedersen (2012), §6. |
| `kl_divergence_categorical(p: 'torch.Tensor', q: 'torch.Tensor') -> 'torch.Tensor'` | KL divergence between two discrete distributions on the simplex. | Cover, T. M., Thomas, J. A. (2006). Elements of Information Theory, 2nd ed., §2.3. Amari (2016), §2.4. |
| `kl_divergence_gaussian(mu_p: 'torch.Tensor', Sigma_p: 'torch.Tensor', mu_q: 'torch.Tensor', Sigma_q: 'torch.Tensor') -> 'torch.Tensor'` | KL divergence between two multivariate Gaussians. | Amari (2016), §2.4; exponential families. Petersen, K. B., Pedersen, M. S. (2012). The Matrix Cookbook, eq. 380; Gaussian-Gaussian KL. |
| `natural_gradient(grad: 'torch.Tensor', fisher_matrix: 'torch.Tensor') -> 'torch.Tensor'` | Natural gradient: F(θ)^{−1} · ∇L. | Amari (1998). Martens (2020), §2; modern derivation. |

## `holonomy_lib.tensor_calculus`

Batched tensors preserve the leading dimension through unfolding, mode products, and HOSVD. HOSVD ranks are bounded by each unfolding shape.

| Signature | Summary | Citation |
|---|---|---|
| `hosvd(T: 'torch.Tensor', ranks: 'Sequence[int]', mode: "Literal['exact', 'randomized']" = 'exact', generator: 'torch.Generator \| None' = None) -> 'tuple[torch.Tensor, list[torch.Tensor]]'` | Truncated Higher-Order SVD (HOSVD). | De Lathauwer, L., De Moor, B., Vandewalle, J. (2000). A multilinear singular value decomposition. SIAM J. Matrix Anal. Appl. 21(4):1253–1278. Vannieuwenhoven, N., Vandebril, R., Meerbergen, K. (2012). A new truncation strategy for the higher-order singular value decomposition. SIAM J. Sci. Comput. 34(2):A1027–A1052. Kolda, T. G., Bader, B. W. (2009), §4.2; Tucker decomposition. |
| `mode_product(T: 'torch.Tensor', A: 'torch.Tensor', axis: 'int') -> 'torch.Tensor'` | n-mode product T ×_axis A. | Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review 51(3):455–500, §2.5. |
| `mode_unfolding(T: 'torch.Tensor', axis: 'int') -> 'torch.Tensor'` | Matricize T along `axis`: move `axis` to position 1, flatten the rest. | Kolda, T. G., Bader, B. W. (2009), §2.4; mode-n matricization. |

## `holonomy_lib.algebra`

Dense matrix operations accept batches. Sparse eigensolvers accept one real symmetric `(n, n)` tensor and reject gradients. The default smallest-algebraic backend uses SciPy on CPU or native LOBPCG off CPU; largest-algebraic Lanczos uses PyTorch. SciPy computes in float64 and restores the output device/dtype. Full or near-full sparse spectrum requests can densify. Solvers warn about nonconverged or nonfinite batch members; inspect warnings and residuals.

| Signature | Summary | Citation |
|---|---|---|
| `cg(A: 'Operator', b: 'torch.Tensor', M: 'Callable[[torch.Tensor], torch.Tensor] \| None' = None, x0: 'torch.Tensor \| None' = None, rtol: 'float \| None' = None, atol: 'float' = 0.0, maxiter: 'int \| None' = None, sync_every: 'int' = 8) -> 'torch.Tensor'` | Conjugate gradient solve of symmetric positive-definite `A x = b`. | Hestenes-Stiefel (1952); Saad (2003) §6.7, §6.11. |
| `eigsh_sparse_sa(A: 'torch.Tensor', k: 'int', sigma: 'float \| None' = None, solver: 'Solver' = 'auto', generator: 'torch.Generator \| None' = None) -> 'tuple[torch.Tensor, torch.Tensor]'` | Return the smallest algebraic eigenpairs of sparse symmetric `A`. | Ericsson and Ruhe (1980); Knyazev (2001); Saad (2011), §3.2; Golub and Van Loan (2013), §7.2. |
| `lanczos_eigsh(A: 'torch.Tensor', k: 'int', n_iter: 'int \| None' = None, oversample: 'int' = 10, generator: 'torch.Generator \| None' = None, which: "Literal['LA', 'SA']" = 'LA', sigma: 'float \| None' = None, solver: "Literal['auto', 'eigsh', 'lobpcg', 'torch_lobpcg']" = 'auto') -> 'tuple[torch.Tensor, torch.Tensor]'` | Return extreme symmetric eigenpairs, or dense modes nearest `sigma`. | Lanczos (1950); Paige (1972); Saad (2011), §6.5. Ericsson, T., Ruhe, A. (1980). The spectral transformation Lanczos method for the numerical solution of large sparse generalized symmetric eigenvalue problems. Math. Comp. 35(152):1251-1268. |
| `lsqr(A: 'torch.Tensor', b: 'torch.Tensor', damp: 'float' = 0.0, atol: 'float \| None' = None, btol: 'float \| None' = None, maxiter: 'int \| None' = None, sync_every: 'int' = 8) -> 'torch.Tensor'` | LSQR: iterative least squares `min ‖Ax − b‖² (+ damp²·‖x‖²)`. | Paige-Saunders (1982), ACM TOMS 8(1):43-71, Alg. LSQR + §6 (stopping criteria). Reference implementation compared against: `scipy.sparse.linalg.lsqr` (tests). |
| `truncated_svd(M: 'torch.Tensor', r: 'int', mode: "Literal['exact', 'randomized']" = 'exact', oversample: 'int' = 5, n_iter: 'int' = 2, generator: 'torch.Generator \| None' = None) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]'` | Return `(U, S, Vh)` using exact or randomized SVD; `Vh` is the conjugate transpose. | Eckart, C., Young, G. (1936). The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218. Halko, N., Martinsson, P. G., Tropp, J. A. (2011). Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review, 53(2):217–288. |

## `holonomy_lib.convex`

Tikhonov least squares, with Cholesky, CG, and LSQR backends. See the historical [solver measurements](notes/benchmark_2026-07-08_solvers.md) for the workloads behind the default dispatch.

| Signature | Summary | Citation |
|---|---|---|
| `regularized_lstsq(A: 'torch.Tensor', b: 'torch.Tensor', lam: 'float', method: 'Method' = 'auto', rtol: 'float \| None' = None, maxiter: 'int \| None' = None) -> 'torch.Tensor'` | Solve the ridge problem `min_x ‖A x − b‖² + λ ‖x‖²`. | Tikhonov (1963); Björck (1996) §2.7; Paige-Saunders (1982) §1. See the module docstring. |

## `holonomy_lib.spectral`

Dense graph inputs are `(B, n, n)`; ordinary Laplacians also accept a single sparse `(n, n)` graph. Self-loops are dropped. Effective resistance and commute time are infinite across components, with zero diagonals. Diffusion time is a nonnegative integer; modes are ordered by absolute transition eigenvalue and retain signed powers. Disconnected diffusion maps warn; their distance interpretation is within components, and isolated-node diffusion distance is undefined.

| Signature | Summary | Citation |
|---|---|---|
| `commute_time(A: 'torch.Tensor') -> 'torch.Tensor'` | Component volume times resistance; cross-component pairs are `inf`. | Chandra et al. (1996), Theorem 2.1. Lovász (1993), §5. |
| `diffusion_map(A: 'torch.Tensor', k: 'int', t: 'int' = 1) -> 'tuple[torch.Tensor, torch.Tensor]'` | Stationary-normalized diffusion coordinates using signed integer powers of transition eigenvalues. | Coifman, R. R., Lafon, S. (2006). Diffusion maps. Applied and Computational Harmonic Analysis 21(1):5-30, §2.3-2.4. |
| `effective_resistance(A: 'torch.Tensor') -> 'torch.Tensor'` | Pairwise effective resistance; cross-component pairs are `inf`. | Klein-Randić (1993), Eq. (10) for the L⁺ expression. Doyle-Snell (1984), §1. |
| `heat_kernel_chebyshev(L: 'torch.Tensor', t: 'float', signal: 'torch.Tensor \| None' = None, K: 'int' = 30, lambda_max: 'float' = 2.0) -> 'torch.Tensor'` | Heat kernel `exp(−t · L)` (or `exp(−t · L) @ signal`) via Chebyshev expansion. | Hammond-Vandergheynst-Gribonval (2011), eq. 3.3. The polynomial recurrence is sequential in K: each term depends on the two preceding terms. Batch members and signal columns are processed together. Coefficients use SciPy on CPU and one transfer to L.device; PyTorch has no general-order scaled Bessel I_k. |
| `laplacian_eigenmaps(A: 'torch.Tensor', k: 'int', laplacian_type: 'LaplacianType' = 'symmetric_normalized') -> 'tuple[torch.Tensor, torch.Tensor]'` | Bottom-k Laplacian eigenmap embedding of a (batched) graph. | Belkin-Niyogi (2003), §2, Laplacian eigenmap. von Luxburg (2007), §5, practical considerations. |
| `spectral_dimension(eigenvalues: 'torch.Tensor', t: 'torch.Tensor') -> 'torch.Tensor'` | Spectral dimension `d_s = -2 · slope of log p(t) vs log t`, where `p(t) = mean_i exp(-t·λ_i)` is the heat-kernel return probability. | Rammal-Toulouse (1983); Durhuus-Jonsson-Wheater (2007). |

### `holonomy_lib.spectral.laplacian`

| Signature | Summary | Citation |
|---|---|---|
| `degree(A: 'torch.Tensor', signed: 'bool' = False) -> 'torch.Tensor'` | Degree vector of a (batched) weighted adjacency matrix. | Chung (1997), §1.2 : weighted degree. Kunegis et al. (2010), eq. 1 : signed degree. |
| `combinatorial(A: 'torch.Tensor') -> 'torch.Tensor'` | Combinatorial Laplacian L = D − A. | Chung (1997), §1.2. von Luxburg (2007), §3 : unnormalized Laplacian. |
| `symmetric_normalized(A: 'torch.Tensor') -> 'torch.Tensor'` | Symmetric normalized Laplacian L_sym = I − D^{−1/2} A D^{−1/2}. | Chung (1997), §1.2. von Luxburg (2007), §3 : L_sym. Cheng-Wu (2024) : pseudoinverse handling of isolated nodes. |
| `random_walk(A: 'torch.Tensor') -> 'torch.Tensor'` | Random-walk Laplacian L_rw = I − D^{−1} A. | Chung (1997), §1.5. von Luxburg (2007), §3 : L_rw. |
| `signed(A: 'torch.Tensor') -> 'torch.Tensor'` | Signed Laplacian L^σ = D^{\|σ\|} − A, D^{\|σ\|}_{ii} = Σ_j \|A_{ij}\|. | Kunegis et al. (2010), eq. 5 : definition of the signed Laplacian. Kunegis et al. (2010), Theorem 3.4 : PSD property and balance. Mercado-Tudisco-Hein (2019), NeurIPS : Signed Power Mean Laplacians generalize this to a one-parameter family. |

### `holonomy_lib.spectral.magnetic`

| Signature | Summary | Citation |
|---|---|---|
| `combinatorial(A: 'torch.Tensor', q: 'float' = 0.25) -> 'torch.Tensor'` | Combinatorial magnetic Laplacian L^(q) = D_s − H ⊙ A_s. | Furutani et al. (2020), eq. 8. Lieb-Loss (1993). |
| `symmetric_normalized(A: 'torch.Tensor', q: 'float' = 0.25) -> 'torch.Tensor'` | Symmetric-normalized magnetic Laplacian. | Furutani et al. (2020), Definition 4 and Prop. 1. |
| `sign_magnetic_combinatorial(A: 'torch.Tensor', q: 'float' = 0.25) -> 'torch.Tensor'` | MSGNN signed magnetic Hermitian Laplacian, with absolute-weight degree. | He, Perlmutter, Reinert, Cucuringu (2022). MSGNN. LoG 2022, PMLR 198, §3.2, equation 1 and Theorem 1. Kunegis et al. (2010) for the q = 0 (signed-undirected) reduction. |
| `sign_magnetic_symmetric_normalized(A: 'torch.Tensor', q: 'float' = 0.25) -> 'torch.Tensor'` | MSGNN signed magnetic Laplacian with absolute-degree normalization. | He, Perlmutter, Reinert, Cucuringu (2022). MSGNN. LoG 2022, PMLR 198, §3.2, equation 2 and Theorems 1 and 2. |

## `holonomy_lib.discrete_geometry`

Ollivier curvature uses stabilized, epsilon-scaled Sinkhorn transport on each pair's neighborhood supports, checks both marginals, and warns on nonconvergence. Cross-component pairs are masked to zero. `forman_ricci_augmented` is the binary, unweighted formula; use `forman_ricci_simple` for weighted graphs. Flow removes self-loops before normalization and surgery.

| Signature | Summary | Citation |
|---|---|---|
| `discrete_ricci_flow(A: 'torch.Tensor', n_steps: 'int', dt: 'float' = 1.0, alpha: 'float' = 0.0, normalize: 'bool' = True, reg: 'float' = 0.003, n_sinkhorn_iters: 'int' = 65536, sinkhorn_tol: 'float' = 0.0001, sinkhorn_tile_size: 'int' = 256, sinkhorn_sync_every: 'int' = 8) -> 'torch.Tensor'` | Discrete Ricci flow on edge weights (Sia 2019, Ni 2019). | Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci curvature-based method to community detection in complex networks. Scientific Reports 9:9800, eq. 3. Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community detection on networks with Ricci flow. Scientific Reports 9:9984. |
| `forman_ricci_augmented(A: 'torch.Tensor') -> 'torch.Tensor'` | Triangle-augmented Forman curvature for binary adjacency only. | Samal, A., et al. (2018). Comparative analysis of two discretizations of Ricci curvature for complex networks. Scientific Reports 8:8650, equations 9–10. |
| `forman_ricci_simple(A: 'torch.Tensor') -> 'torch.Tensor'` | Forman-Ricci curvature on every edge of a batched weighted graph. | Sreejith et al. (2016), eq. 1. Forman (2003), §"Combinatorial Ricci curvature for graphs". |
| `ollivier_ricci_curvature(A: 'torch.Tensor', alpha: 'float' = 0.0, reg: 'float' = 0.003, n_iter: 'int' = 65536, tol: 'float' = 0.0001, tile_size: 'int' = 256, sync_every: 'int' = 8) -> 'torch.Tensor'` | Pairwise Ollivier-Ricci curvature on a batched weighted graph. | Ollivier (2009), Definition 1. Cuturi (2013), Algorithm 1, for entropic Sinkhorn transport. |
| `ricci_flow_with_surgery(A: 'torch.Tensor', n_steps: 'int', surgery_period: 'int' = 10, surgery_threshold: 'float' = 3.0, dt: 'float' = 1.0, alpha: 'float' = 0.0, normalize: 'bool' = True, reg: 'float' = 0.003, n_sinkhorn_iters: 'int' = 65536, sinkhorn_tol: 'float' = 0.0001, sinkhorn_tile_size: 'int' = 256, sinkhorn_sync_every: 'int' = 8) -> 'torch.Tensor'` | Discrete Ricci flow with threshold-based edge surgery. | Sia, J., Jonckheere, E., Bogdan, P. (2019). Ollivier-Ricci curvature-based method to community detection in complex networks. Scientific Reports 9:9800. Ni, C.-C., Lin, Y.-Y., Luo, F., Gao, J. (2019). Community detection on networks with Ricci flow. Scientific Reports 9:9984, §3.2. |

## `holonomy_lib.sheaf`

Sheaf structures describe one graph with real restriction maps. Cochains may be batched. Dense and sparse operator assembly share the same maps; `sheaf_bottom_spectrum` drops the kernel using a matrix-size/dtype-aware threshold. Complex restrictions, fractional indices, self-loops, and duplicate edges in either orientation are rejected.

| Signature | Summary | Citation |
|---|---|---|
| `sheaf_bottom_spectrum(sheaf: 'GraphSheaf \| HeterogeneousGraphSheaf', k: 'int', drop_kernel: 'bool' = True, kernel_headroom: 'int' = 16, solver: "Literal['auto', 'eigsh', 'lobpcg', 'torch_lobpcg']" = 'auto', generator: 'torch.Generator \| None' = None) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]'` | Bottom-k spectral form of the sheaf Laplacian via a sparse solve. | Hansen-Ghrist (2019), §3.1 (kernel = global sections), §4 (harmonic/spectral reads of the sheaf Laplacian). Ericsson-Ruhe (1980); Knyazev (2001), for the sparse bottom-k backends, via `algebra.eigsh_sparse`. |
| `sheaf_coboundary(sheaf: 'GraphSheaf \| HeterogeneousGraphSheaf') -> 'torch.Tensor'` | Dense coboundary of shape `(sum_e d_e, sum_v d_v)`, including heterogeneous stalks. | Hansen-Ghrist (2019), eq. 2. |
| `sheaf_coboundary_sparse(sheaf: 'GraphSheaf \| HeterogeneousGraphSheaf') -> 'torch.Tensor'` | Sheaf coboundary `δ` as a coalesced sparse-COO `(Σd_e, Σd_v)` tensor. | Hansen-Ghrist (2019), eq. 2. |
| `sheaf_dirichlet_energy(sheaf: 'GraphSheaf \| HeterogeneousGraphSheaf', x: 'torch.Tensor') -> 'torch.Tensor'` | Per-batch Dirichlet energy `E(x) = x^T L_F x` for sheaf-valued signals. | Hansen-Ghrist (2019), §3.2. |
| `sheaf_laplacian(sheaf: 'GraphSheaf \| HeterogeneousGraphSheaf') -> 'torch.Tensor'` | Dense sheaf Laplacian `delta.T @ delta`, shape `(sum_v d_v, sum_v d_v)`. | Hansen-Ghrist (2019), §3.1. Bodnar et al. (2022), eq. 6. |
| `sheaf_laplacian_sparse(sheaf: 'GraphSheaf \| HeterogeneousGraphSheaf') -> 'torch.Tensor'` | Sheaf Laplacian `L_F = δ^T δ` as a coalesced sparse-COO tensor. | Hansen-Ghrist (2019), §3.1. Bodnar et al. (2022), eq. 6. |

### `GraphSheaf`

| Signature | Summary | Citation |
|---|---|---|
| `GraphSheaf(n_nodes: 'int', edges: 'torch.Tensor', node_stalk_dim: 'int', edge_stalk_dim: 'int', F_left: 'torch.Tensor', F_right: 'torch.Tensor') -> None` | A cellular sheaf on a graph. | Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology 3:315–358, §3. |
| `GraphSheaf.device (property) -> 'torch.device'` | Device of the restriction maps. | Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology 3:315–358, §3. |
| `GraphSheaf.dtype (property) -> 'torch.dtype'` | Floating dtype of the restriction maps. | Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology 3:315–358, §3. |
| `GraphSheaf.n_edges (property) -> 'int'` | Number of graph edges. | Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology 3:315–358, §3. |
| `GraphSheaf.total_node_dim (property) -> 'int'` | Length of the stacked node-stalk vector, `n_nodes · d_v`. | Hansen, J., Ghrist, R. (2019). Toward a spectral theory of cellular sheaves. Journal of Applied and Computational Topology 3:315–358, §3. |
| `GraphSheaf.trivial(n_nodes: 'int', edges: 'torch.Tensor', stalk_dim: 'int' = 1, device: 'str \| torch.device \| None' = None, dtype: 'torch.dtype' = torch.float64) -> 'GraphSheaf'` | Construct the *trivial* sheaf: identity restriction maps on every edge, uniform stalk dimension on nodes and edges. | Hansen-Ghrist (2019), §3.1, constant sheaves and graph Laplacians. |

### `HeterogeneousGraphSheaf`

| Signature | Summary | Citation |
|---|---|---|
| `HeterogeneousGraphSheaf(n_nodes: 'int', edges: 'torch.Tensor', node_stalk_dims: 'torch.Tensor', edge_stalk_dims: 'torch.Tensor', F_left: 'tuple[torch.Tensor, ...]', F_right: 'tuple[torch.Tensor, ...]') -> None` | A cellular sheaf with PER-NODE (heterogeneous) stalk dimensions. | Hansen-Ghrist (2019): cellular sheaves carry per-cell stalks of arbitrary finite dimension; the uniform `GraphSheaf` is the special case. |
| `HeterogeneousGraphSheaf.device (property) -> 'torch.device'` | Device of the first restriction map, or of node_stalk_dims for an edgeless sheaf. | Hansen-Ghrist (2019): cellular sheaves carry per-cell stalks of arbitrary finite dimension; the uniform `GraphSheaf` is the special case. |
| `HeterogeneousGraphSheaf.dtype (property) -> 'torch.dtype'` | Dtype of the first restriction map, or float64 for an edgeless sheaf. | Hansen-Ghrist (2019): cellular sheaves carry per-cell stalks of arbitrary finite dimension; the uniform `GraphSheaf` is the special case. |
| `HeterogeneousGraphSheaf.n_edges (property) -> 'int'` | Number of graph edges. | Hansen-Ghrist (2019): cellular sheaves carry per-cell stalks of arbitrary finite dimension; the uniform `GraphSheaf` is the special case. |
| `HeterogeneousGraphSheaf.total_node_dim (property) -> 'int'` | Length of the stacked node-stalk vector, `Σ_i d_v[i]`. | Hansen-Ghrist (2019): cellular sheaves carry per-cell stalks of arbitrary finite dimension; the uniform `GraphSheaf` is the special case. |

## `holonomy_lib.simplicial`

Dense complexes use padded batched simplex tables with validity masks; sparse complexes represent one instance. Boundary operators are dense or sparse respectively. Vietoris-Rips construction tiles candidate extensions with `candidate_tile_size`.

| Signature | Summary | Citation |
|---|---|---|
| `pairwise_distances(points: 'torch.Tensor') -> 'torch.Tensor'` | Pairwise Euclidean distances. | Standard Euclidean distance. See e.g. Hartle, J. B. (2003), Gravity: An Introduction to Einstein's General Relativity, §2.1 for the abstract metric formulation; computationally we use `\|\|p - q\|\|_2` directly. |
| `vietoris_rips_dense(distances: 'torch.Tensor', max_radius: 'float', max_dim: 'int', dtype: 'torch.dtype \| None' = None, *, candidate_tile_size: 'int' = 1024) -> 'DenseSimplicialComplex'` | Batched Vietoris-Rips construction with padded simplex tables. | Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a cohomology theory for metric spaces. Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. J. Appl. Comput. Topology 5:391-423. |
| `vietoris_rips_sparse(distances: 'torch.Tensor', max_radius: 'float', max_dim: 'int', *, candidate_tile_size: 'int' = 1024) -> 'SparseSimplicialComplex'` | Vietoris-Rips complex from a single distance matrix. | Hausmann, J.-C. (1995). On the Vietoris-Rips complexes and a cohomology theory for metric spaces. In Prospects in Topology, Annals of Math Studies 138. Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. J. Appl. Comput. Topology 5:391-423. |

### `DenseSimplicialComplex`

| Signature | Summary | Citation |
|---|---|---|
| `DenseSimplicialComplex(simplices_by_dim: 'dict[int, torch.Tensor]', valid_mask: 'dict[int, torch.Tensor]', n_vertices: 'int', device: 'torch.device' = device(type='cpu'), dtype: 'torch.dtype' = torch.float64) -> None` | Batched, padded simplicial complex. | Munkres, J. R. (1984). Elements of Algebraic Topology. Westview Press. §1 : simplicial complexes and chain complexes. Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review 51(3):455-500. (Tensor-layout conventions reused here.) |
| `DenseSimplicialComplex.batch_size (property) -> 'int'` | Leading batch dim. | Munkres, J. R. (1984). Elements of Algebraic Topology. Westview Press. §1 : simplicial complexes and chain complexes. Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review 51(3):455-500. (Tensor-layout conventions reused here.) |
| `DenseSimplicialComplex.boundary(k: 'int') -> 'torch.Tensor'` | Boundary operator `∂_k: C_k → C_{k-1}`. | Munkres, J. R. (1984). Elements of Algebraic Topology, §1. |
| `DenseSimplicialComplex.max_dim (property) -> 'int'` | Highest dimension with a table, including an empty table. | Munkres, J. R. (1984). Elements of Algebraic Topology. Westview Press. §1 : simplicial complexes and chain complexes. Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review 51(3):455-500. (Tensor-layout conventions reused here.) |
| `DenseSimplicialComplex.n_simplices(k: 'int') -> 'torch.Tensor'` | Per-batch count of valid k-simplices: shape `(B,)`. | Munkres, J. R. (1984). Elements of Algebraic Topology. Westview Press. §1 : simplicial complexes and chain complexes. Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review 51(3):455-500. (Tensor-layout conventions reused here.) |
| `DenseSimplicialComplex.to_sparse() -> 'SparseSimplicialComplex'` | Convert to a single-instance sparse complex. | Munkres, J. R. (1984). Elements of Algebraic Topology. Westview Press. §1 : simplicial complexes and chain complexes. Kolda, T. G., Bader, B. W. (2009). Tensor decompositions and applications. SIAM Review 51(3):455-500. (Tensor-layout conventions reused here.) |

### `SparseSimplicialComplex`

| Signature | Summary | Citation |
|---|---|---|
| `SparseSimplicialComplex(simplices_by_dim: 'dict[int, torch.Tensor]', n_vertices: 'int', device: 'torch.device' = device(type='cpu')) -> None` | Single-instance simplicial complex. | Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed. SIAM. §3.4 : CSR and CSC formats. Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological persistence and simplification. Discrete & Computational Geometry 28:511-533. Original sparse-boundary reduction algorithm. |
| `SparseSimplicialComplex.boundary(k: 'int', dtype: 'torch.dtype' = torch.float64) -> 'torch.Tensor'` | Sparse-CSC boundary operator `∂_k: C_k → C_{k-1}`. | Munkres, J. R. (1984). Elements of Algebraic Topology, §1. |
| `SparseSimplicialComplex.max_dim (property) -> 'int'` | Highest dimension with a table, including an empty table. | Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed. SIAM. §3.4 : CSR and CSC formats. Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological persistence and simplification. Discrete & Computational Geometry 28:511-533. Original sparse-boundary reduction algorithm. |
| `SparseSimplicialComplex.n_simplices(k: 'int') -> 'int'` | Number of k-simplices in this instance. | Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed. SIAM. §3.4 : CSR and CSC formats. Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological persistence and simplification. Discrete & Computational Geometry 28:511-533. Original sparse-boundary reduction algorithm. |
| `SparseSimplicialComplex.to_dense(dtype: 'torch.dtype' = torch.float64) -> 'DenseSimplicialComplex'` | Convert to a batched-with-B=1 dense complex. | Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed. SIAM. §3.4 : CSR and CSC formats. Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). Topological persistence and simplification. Discrete & Computational Geometry 28:511-533. Original sparse-boundary reduction algorithm. |

## `holonomy_lib.topology`

Hodge Laplacians are dense even for sparse complexes. Betti numbers use dtype-aware matrix-rank tolerances by default. Persistence accepts batches but performs H0 union-find and Z2 boundary reduction sequentially per member, with host-side metadata. Killed classes are not emitted as essential. H0 retains zero-length bars for coincident vertices; higher-dimensional zero-length bars are omitted.

| Signature | Summary | Citation |
|---|---|---|
| `betti_numbers(complex: 'DenseSimplicialComplex \| SparseSimplicialComplex', max_dim: 'int', threshold: 'float \| None' = None) -> 'torch.Tensor'` | Betti numbers `(β_0, β_1, …, β_max_dim)` for a simplicial complex. | Eckmann (1944); kernel of L_k equals harmonic k-forms. Lim (2020), §4. |
| `hodge_laplacian(complex: 'DenseSimplicialComplex \| SparseSimplicialComplex', k: 'int') -> 'torch.Tensor'` | Hodge Laplacian `L_k = ∂_{k+1} ∂_{k+1}^T + ∂_k^T ∂_k`. | Eckmann (1944). Harmonische Funktionen und Randwertaufgaben in einem Komplex. Comment. Math. Helv. 17:240–255. Lim (2020). Hodge Laplacians on graphs. SIAM Review 62:685–715. |
| `persistence_diagrams(points_or_distances: 'torch.Tensor', max_dim: 'int' = 2, max_radius: 'float' = inf, input_is_distance: 'bool' = False, reduction_backend: "Literal['python', 'torch']" = 'python') -> 'tuple[list[torch.Tensor], list[torch.Tensor]]'` | Persistence diagrams for dims 0..max_dim of a Vietoris-Rips filtration. | Edelsbrunner-Letscher-Zomorodian (2002). Cohen-Steiner-Edelsbrunner-Harer (2007) stability. |

## `holonomy_lib.experimental`

In progress. These names are exported only here. Attraction-only eigenmaps admit collapsed embeddings; heterogeneous curvature supplies pair-conditioned dissimilarities, not a variable-curvature metric. Citations give background, not a correctness or convergence claim for these constructions.

| Signature | Summary | Citation |
|---|---|---|
| `hyperbolic_laplacian_eigenmaps(adjacency: 'torch.Tensor', manifold: 'Any', max_steps: 'int' = 200, lr: 'float' = 0.05, generator: 'torch.Generator \| None' = None, init: 'torch.Tensor \| None' = None) -> 'torch.Tensor'` | Attraction-only graph embedding on a manifold; the unconstrained objective admits collapse. | Belkin–Niyogi (2003); Nickel–Kiela (2017); Liu et al. (2019). |

### `HeterogeneousKappaManifold`

| Signature | Summary | Citation |
|---|---|---|
| `HeterogeneousKappaManifold(n: 'int', combiner: 'str \| Callable' = 'arithmetic_mean', device: 'str \| torch.device' = 'cpu', dtype: 'torch.dtype' = torch.float64)` | Experimental per-point constant-curvature maps and pair-conditioned dissimilarities; no metric guarantee. | Bachmann, Bécigneul, Ganea (2020). Constant Curvature Graph Convolutional Networks, ICML, §2 (the homogeneous formulas used per point/pair). The pair-curvature combination is an experimental modeling choice, not a formula established by that reference. |
| `HeterogeneousKappaManifold.ambient_dim (property) -> 'int'` | Ambient coordinate dimension n. | Bachmann, Bécigneul, Ganea (2020). Constant Curvature Graph Convolutional Networks, ICML, §2 (the homogeneous formulas used per point/pair). The pair-curvature combination is an experimental modeling choice, not a formula established by that reference. |
| `HeterogeneousKappaManifold.dim (property) -> 'int'` | Intrinsic coordinate dimension n. | Bachmann, Bécigneul, Ganea (2020). Constant Curvature Graph Convolutional Networks, ICML, §2 (the homogeneous formulas used per point/pair). The pair-curvature combination is an experimental modeling choice, not a formula established by that reference. |
| `HeterogeneousKappaManifold.dissimilarity(x: 'torch.Tensor', kappa_x: 'torch.Tensor', y: 'torch.Tensor', kappa_y: 'torch.Tensor') -> 'torch.Tensor'` | Compare endpoints under a combined curvature; validates both chart domains but can violate metric axioms. | Bachmann et al. (2020), §2, constant-curvature distance only. Combining endpoint curvatures is an experimental modeling choice. |
| `HeterogeneousKappaManifold.exp_0(v: 'torch.Tensor', kappa: 'torch.Tensor') -> 'torch.Tensor'` | Per-point exp at the origin: `exp_0(v_i; κ_i)`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, homogeneous formula applied per point. |
| `HeterogeneousKappaManifold.is_on_manifold(x: 'torch.Tensor', kappa: 'torch.Tensor', atol: 'float' = 1e-09) -> 'torch.Tensor'` | Check finite coordinates and -κ\|\|x\|\|² < 1 - atol for κ < 0. | Bachmann et al. (2020), §2, homogeneous chart domains. |
| `HeterogeneousKappaManifold.log_0(y: 'torch.Tensor', kappa: 'torch.Tensor') -> 'torch.Tensor'` | Per-point log at the origin: `log_0(y_i; κ_i)`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, homogeneous formula applied per point. |
| `HeterogeneousKappaManifold.origin(batch_size: 'int' = 1) -> 'torch.Tensor'` | Origin `(0, …, 0) ∈ R^n`. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, homogeneous formula applied per point. |
| `HeterogeneousKappaManifold.random_point(batch_size: 'int' = 1, kappa: 'torch.Tensor \| None' = None, generator: 'torch.Generator \| None' = None) -> 'torch.Tensor'` | Sample `batch_size` random points on the manifold. | Bachmann, Bécigneul, Ganea (2020), Constant Curvature Graph Convolutional Networks, §2, homogeneous formula applied per point. |
