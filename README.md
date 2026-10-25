# holonomy_lib

Geometry, graph and topology primitives for PyTorch. Most functions take a batch
of inputs (a leading batch dimension) and run on the device and dtype you pass in.

## Install

Python 3.12 or later. Install PyTorch for your hardware first, using the
[PyTorch selector](https://pytorch.org/get-started/locally/), then:

```bash
pip install holonomy-lib
```

## Example

Effective resistance between the ends of a three-node path and of a triangle,
computed as one batch:

```python
import torch
from holonomy_lib.spectral import effective_resistance

device = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm builds also use "cuda"
adjacency = torch.tensor([
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],  # path 0-1-2
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]],  # triangle
], dtype=torch.float64, device=device)

R = effective_resistance(adjacency)
print(R.shape)        # torch.Size([2, 3, 3])
print(R[:, 0, 2])     # tensor([2.0000, 0.6667], dtype=torch.float64)
```

## What's in it

[CONTENTS.md](https://github.com/Synoros-io/holonomy_lib/blob/master/CONTENTS.md) lists every function with its signature and the
paper it implements.

| Subpackage | Contents |
|---|---|
| `manifolds` | Fixed-rank, SPD, Lorentz, Lorentzian, stereographic and product manifolds; model-space volumes. |
| `optimization` | Riemannian SGD. |
| `hyperbolic` | Fréchet means, manifold-aware inner products, hyperbolic heat kernels. |
| `lie` | SO(3) exponential and logarithm, rotations, real spherical harmonics. |
| `info_geometry` | Bregman and KL divergences, Fisher information, natural gradients. |
| `tensor_calculus` | Mode unfolding, mode products, HOSVD. |
| `algebra` | Truncated SVD, dense and sparse eigensolvers, conjugate gradient, LSQR. |
| `convex` | Regularized least squares. |
| `spectral` | Graph Laplacians, embeddings, heat diffusion, effective resistance, spectral dimension. |
| `discrete_geometry` | Ollivier and Forman curvature, Ricci flow with surgery. |
| `sheaf` | Graph sheaves, coboundaries, sheaf Laplacians and their spectra. |
| `simplicial` | Dense and sparse simplicial complexes, boundary operators, Vietoris-Rips. |
| `topology` | Hodge Laplacians, Betti numbers, persistent homology. |

## Explorer

`holonomy-explorer` opens a local page that runs the library live: curvature
and diffusion on graphs, geodesics as curvature passes through zero,
persistent homology of point clouds, and batch-size timings on your hardware.
Options: `--device`, `--dtype`, `--port`, `--no-browser`.

![The explorer's graph view](https://raw.githubusercontent.com/Synoros-io/holonomy_lib/master/docs/explorer.png)

## Experimental

`holonomy_lib.experimental` holds two unfinished pieces that are not part of
the stable API:

- `hyperbolic_laplacian_eigenmaps`: nothing in its objective keeps nodes apart,
  so an embedding can collapse to a point.
- `HeterogeneousKappaManifold`: a curvature per point. Its `dissimilarity` is
  not a metric.

## Limits

- Use `float64` when accuracy matters.
- Sparse eigensolvers, sparse simplicial complexes and sheaves handle one
  instance at a time. Sparse eigensolvers are not differentiable.
- Persistent homology runs its reduction one batch member at a time on the CPU.
- Ollivier curvature uses entropy-regularized transport, so it is close to the
  exact value rather than equal to it, and it warns if the solver did not
  converge. Edge weights act as both lengths and random-walk weights.
- Gradients through `cg` can be wrong when it converges in very few iterations.
- Effective resistance and commute time are `inf` between disconnected components.

## Development

```bash
uv sync --extra dev
python -m pytest -q
python -m ruff check src/ tests/
python -m mypy src/
python tools/audit.py src/holonomy_lib --strict
```

[CONVENTIONS.md](https://github.com/Synoros-io/holonomy_lib/blob/master/CONVENTIONS.md) has the coding and numerical rules;
[CHANGELOG.md](https://github.com/Synoros-io/holonomy_lib/blob/master/CHANGELOG.md) has the release history.

## Citation

```bibtex
@software{holonomy_lib,
  author = {John Vaught},
  title = {holonomy\_lib: Batched PyTorch primitives for geometry and topology},
  version = {0.8.0},
  year = {2026},
  url = {https://github.com/Synoros-io/holonomy_lib},
}
```

Please also cite the papers named in the docstrings of the functions you use.

## License

[FSL-1.1-ALv2](https://github.com/Synoros-io/holonomy_lib/blob/master/LICENSE.md). You can use and modify it freely; competing
commercial use is reserved to Synoros. Each release becomes Apache-2.0 two
years after it ships. Releases up to 0.7.0 remain AGPL-3.0-or-later. For other
terms, contact [contact@synoros.io](mailto:contact@synoros.io).
