# basisHD

A Hyperdimensional Computing (HDC) / Vector Symbolic Architecture (VSA) extension built on top of [`torchhd`](https://github.com/hyperdimensional-computing/torchhd), providing deterministic, index-based hypervector generation with a binary/bipolar dual representation.

---

## Overview

`basisHD` extends `torchhd`'s `MAPTensor` with a **seed-based vector generator**. Instead of sampling random hypervectors independently, it:

- Pre-generates a **seed tensor** containing all basis vectors of dimension `d = 2^k`.
- Enumerates **all `2^k` index combinations** of the basis vectors.
- Constructs hypervectors by multiplying selected basis vectors element-wise (bipolar form), or converting to binary form.

This gives a **deterministic, reproducible, and orthogonal** set of hypervectors — useful for symbolic binding, encoding, and associative memory in HDC pipelines.

---

## Features

- ✅ **Power-of-2 dimension enforcement** (`dim & (dim - 1) == 0`).
- ✅ **Index-combination vector pool** — exactly `dim` unique hypervectors available per generator.
- ✅ **Bipolar / binary dual representation** (`+1/-1` vs `1/0`).
- ✅ **History recording** — optionally track which index combinations have been consumed.
- ✅ **Multiple pop strategies** — `popTensor` (random), `popFirstTensor`, `popLastTensor`.
- ✅ **`torchhd` compatible** — subclasses `MAPTensor`, interoperates with `torchhd.functional`.
- ✅ **Level hypervector support** — custom `level()` implementation using user-supplied start/end tensors.

---

## Project Structure
