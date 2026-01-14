# 01 – Linear Algebra Foundations (for ML & Fintech)

This module is about learning just enough **linear algebra** to comfortably understand and implement machine learning models (including deep learning and later transformers), not to become a mathematician.

## Why Linear Algebra Matters for ML

- Every **data point** is a vector in $\mathbb{R}^n$ (e.g., a transaction with *n* features).
- Every **neural network layer** is a matrix transforming one vector space into another (forward pass = matrix multiplication).
- Concepts like **similarity, projections, eigenvalues** show up in attention, optimization, and dimensionality reduction.

In fintech terms: your fraud/risk models are just linear algebra operations on large collections of vectors.

## Goals of This Folder

By the end of this module you should be able to:

- Represent vectors and matrices in NumPy and visualize them in 2D.
- Explain what $\mathbb{R}^n$ (real coordinate spaces) means and relate it to features in a dataset.
- Perform and understand:
  - Vector addition and scalar multiplication
  - Dot product and vector norms (length)
  - Basic matrix operations (multiplication, transpose, inverse for small matrices)
  - Eigenvalues and eigenvectors at an intuitive level
- Connect these operations to:
  - Neural network forward pass \( y = W x \)
  - Simple intuition for optimization and stability

You do **not** need full proof-level rigor; you need solid intuition plus the ability to code these operations.

## Files in This Folder

- `vectors_intro.py`  
  - Basic vectors in NumPy  
  - Real coordinate spaces $\mathbb{R}^1$, $\mathbb{R}^2$, $\mathbb{R}^3$ \) with examples  
  - Visualization of vectors and basis in 2D (matplotlib)

- *(Planned)* `vector_ops.ipynb`  
  - Vector addition, scalar multiplication, dot product, norms  
  - Geometric interpretation (length, angle, similarity)

- *(Planned)* `matrices_basics.ipynb`  
  - Matrix–vector and matrix–matrix multiplication  
  - Forward pass example: simple linear layer \( y = W x + b \)

- *(Planned)* `eigen_intuition.ipynb`  
  - Eigenvalues/eigenvectors using small 2×2 matrices  
  - How they relate to stretching/compressing directions in space

- `progress_week_1.md`  
  - Short weekly log of what was learned and implemented

## What I’m Using to Learn

- **Khan Academy – Linear Algebra (selected videos only):**  
  - Vectors & spaces, matrix multiplication, determinants, eigenvalues/eigenvectors
- **3Blue1Brown – Essence of Linear Algebra (for intuition)**  
  - Especially videos on vectors, linear combinations, and eigenvectors

I am intentionally **not** watching all 144 videos—only the subset directly useful for ML.

## How This Connects to Later Phases

- **Phase 2 (Gen AI):**  
  - Embeddings, attention scores, and transformer layers are all linear algebra operations on high‑dimensional vectors.

- **Phase 3 (Agentic AI):**  
  - Agents still run on models whose internals depend on these operations; understanding shapes and operations helps debug and design them.

- **Fintech Applications:**  
  - Fraud detection, credit scoring, portfolio features → all start as vectors in $\mathbb{R}^n$. Models you build later will be compositions of the operations practiced here.

## Done vs To‑Do

**Completed:**
- [x] Basic vector and coordinate space code (`vectors_intro.py`)

**Next:**
- [ ] Implement vector operations (add, scale, dot, norms)
- [ ] Implement basic matrix operations (mul, transpose, inverse of 2×2)
- [ ] Build a tiny “linear layer” example \( y = W x \) with real fintech‑like features
- [ ] Write a short note in `progress_week_1.md` summarizing key insights

---

This folder is about building *muscle memory* for linear algebra in code. Every concept should end in a NumPy example and a GitHub commit.
