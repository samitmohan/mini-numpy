# mini-numpy — a tiny NumPy-like ndarray from scratch

**mini-numpy** is a weekend project that implements a minimal `ndarray` with
- contiguous buffer (`float64`),
- `shape`/`strides`/views,
- basic indexing & slicing,
- broadcasting + elementwise ops,
- reductions (`sum`),
- linear algebra (`dot` for 1D/2D),
- and a small test/benchmark suite.

> Goal: learn array internals (layout, strides, broadcasting) and show systems thinking on a résumé.

---

## Features (MVP)

- `NDArray` with contiguous C-order buffer  
- Constructors: `from_list`, `zeros`, `ones`  
- Indexing: `a[i]`, `a[i, j]`, basic slices (step=1) → **views** (no copy)  
- Elementwise ops: `+  -  *  /` with **broadcasting**  
- Reductions: `sum(axis=None|int)`  
- Linear algebra: `dot` (vector·vector, matrix@matrix, matrix@vector)  
- Utilities: `reshape`, `T` (transpose), pretty `repr`

**Stretch (if time):** `mean/std`, negative/step slicing, dtypes, Numba/Cython kernel for `dot`/`sum`.

---

## Quickstart

```bash
git clone https://github.com/<you>/mini_numpy.git
cd mini_numpy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"   # installs pytest, numpy (for validation), ruff
pytest -q                  # run tests
```

Small demo:

```python
from mini_numpy.core import NDArray

A = NDArray.from_list([[1,2,3],[4,5,6]])  # shape=(2,3)
b = NDArray.from_list([[1],[10]])         # shape=(2,1)
C = A + b                                 # broadcasting to (2,3)
print(C.sum())                            # reduction
print(A.dot(A.T()).shape)                 # (2,2)
```

---

## Project Layout

```
mini_numpy/
├── mini_numpy/
│   ├── __init__.py
│   ├── core.py        # NDArray, elementwise ops, sum, reshape, T
│   ├── linalg.py      # dot (optionally split out)
│   └── utils.py       # strides, shape helpers
├── tests/
│   ├── test_array.py
│   ├── test_broadcast.py
│   ├── test_sum.py
│   └── test_dot.py
├── examples/
│   ├── demo.ipynb
│   └── benchmark.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Design Notes (how it works)

- **Buffer model:** contiguous `float64` using Python `array('d')` (or `bytearray`/`memoryview`).  
- **Strides (C-order):** strides computed right-to-left; views adjust `offset` + reuse buffer.  
- **Indexing:** ints reduce rank; basic slices (`start:stop:step=1`) produce **views**.  
- **Broadcasting:** aligns shapes right-to-left; dimensions of size 1 broadcast.  
- **Elementwise ops:** iterate output space; map to input offsets using broadcast rules.  
- **Reductions:** iterate full index space; accumulate into output tensor lacking the reduced axis.  
- **Dot:** naïve triple loop for matrix@matrix and vector special cases.  
- **Error handling:** shape mismatches for broadcasting/dot, bounds checks for indexing.  
- **Performance:** correctness-first; optional acceleration via **Numba/Cython** for hot loops.

---

## Testing & Benchmarks

```bash
pytest -q
python examples/benchmark.py     # prints tiny timing and correctness checks
```

The benchmark compares MiniNumPy results to NumPy (for correctness) and prints bare timings (education-only).

---

## Limitations

- Only `float64` dtype, C-order, and step=1 slices in MVP.  
- No advanced/boolean indexing, no memory ownership outside contiguous buffer.  
- Performance is educational; for real workloads use NumPy.

---

## Roadmap

- Negative/step slicing, `mean/std/where`  
- DType variants (`float32/int32/bool`)  
- **Numba/Cython** accelerator for `sum`/`dot`  
- Simple random module (LCG/PCG)  
- BLAS backend toggle for `dot`

---
