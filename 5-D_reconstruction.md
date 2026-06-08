a **tomographic inverse problem with binary (thresholded) measurements**, i.e. a *set reconstruction problem with a latent 1D manifold*.

I’ll give you a concrete algorithm that actually makes sense computationally, not just notation.

---

# 1. What you are given

You observe only a binary regime-switch signal:

[
\chi(s) \in {0,1}, \quad s \in \mathcal{S}
]

with

[
\chi(s) = H(\Phi(S(s) - \gamma))
]

Interpreted geometrically:

* There is a hidden curve (\gamma(t) \subset \mathbb{R}^5)
* A scanning map (S(s)) probes space
* A scalar field (\Phi) changes sign near interaction with (\gamma)
* You only see **horizon crossings**

So your observable is effectively:

> “Did the scan ray intersect or cross a tubular neighborhood of (\gamma)?”

---

# 2. Key reformulation (tomography step)

We convert the binary signal into **constraint surfaces**.

Define:

[
\Omega = {x \in \mathbb{R}^5 : \Phi(x) > 0}
]

Then:

* (\chi(s)=1 \Rightarrow S(s)\in \Omega)
* (\chi(s)=0 \Rightarrow S(s)\notin \Omega)

So each scan gives a **half-space constraint in function space of (\gamma)**.

---

# 3. Discretize the unknown curve

Represent the curve as a spline / polyline:

[
\gamma(t) \approx {g_1, g_2, \dots, g_N}, \quad g_i \in \mathbb{R}^5
]

with smoothness prior:

[
\mathcal{R}(\gamma) = \sum_{i=2}^{N-1} |g_{i+1} - 2g_i + g_{i-1}|^2
]

---

# 4. Forward model (synthetic tomography operator)

Define a “hit function”:

[
h(s, \gamma) = \min_{t \in [0,1]} |S(s) - \gamma(t)|
]

Then:

[
\hat{\chi}(s;\gamma) = H(\epsilon - h(s,\gamma))
]

So reconstruction becomes:

> find curve whose tubular neighborhood explains all threshold flips.

---

# 5. Inverse problem (core optimization)

We solve:

[
\min_{\gamma}
\sum_{s \in \mathcal{S}}
\mathcal{L}(\chi(s), \hat{\chi}(s;\gamma))

* \lambda \mathcal{R}(\gamma)
  ]

where binary loss is:

[
\mathcal{L} =
\chi \log(\hat{p}) + (1-\chi)\log(1-\hat{p})
]

but since we have hard thresholds, we replace Heaviside with smooth approximation:

[
H(x) \approx \sigma(kx)
]

So:

[
\hat{\chi}(s;\gamma) = \sigma\big(k(\epsilon - h(s,\gamma))\big)
]

---

# 6. Tomographic reconstruction algorithm (practical)

## Algorithm: Threshold Tomography for Latent Curve Recovery

### Input:

* scan points (S(s_i))
* binary regime signal (\chi_i)

---

## Step 0: Initialize curve

Random walk or PCA initialization:

[
\gamma^{(0)} = \text{principal curve of } S(s_i) \text{ where } \chi=1
]

---

## Step 1: Compute distance field

For each scan point:

[
d_i^{(k)} = \min_t |S(s_i) - \gamma^{(k)}(t)|
]

---

## Step 2: Soft prediction

[
\hat{\chi}_i^{(k)} =
\sigma(k(\epsilon - d_i^{(k)}))
]

---

## Step 3: Gradient update

Minimize:

[
\mathcal{J} =
\sum_i (\chi_i - \hat{\chi}_i^{(k)})^2

* \lambda \sum_i |\Delta^2 \gamma_i|^2
  ]

Gradient descent on control points:

[
g_j \leftarrow g_j - \eta \frac{\partial \mathcal{J}}{\partial g_j}
]

---

## Step 4: Geometric refinement (critical step)

After convergence:

* Extract “hit set”:
  [
  \mathcal{H} = {S(s_i): \chi_i=1}
  ]

* Fit a **principal curve / spline through dense hit manifold**

Use:

* expectation-maximization principal curve update:

[
g_j^{(k+1)} = \mathbb{E}[x \mid x \in \mathcal{H}, \text{nearest to } g_j]
]

---

## Step 5: Topology stabilization

Enforce no self-intersections unless data supports it:

Penalize:

[
\mathcal{T}(\gamma) =
\sum_{i \neq j} \exp\left(-\frac{|g_i - g_j|^2}{\sigma^2}\right)
]

---

# 7. Final algorithm summary

You are iterating:

1. guess curve
2. compute distance-to-curve from scan space
3. convert to predicted binary regime
4. compare with observed regime flips
5. update curve via gradient + projection
6. enforce smooth + topological regularity

---

# 8. What makes this “tomography”

Classical tomography:

[
\text{integrals of field along rays}
]

Your case:

[
\text{binary phase-change induced by proximity to manifold}
]

So instead of reconstructing density, you reconstruct:

> a **codimension-4 object (curve in (\mathbb{R}^5)) from thresholded sampling of its tubular neighborhood**

That’s closer to:

* medial axis reconstruction
* 1D manifold learning from set membership queries
* level-set inverse problems

---
