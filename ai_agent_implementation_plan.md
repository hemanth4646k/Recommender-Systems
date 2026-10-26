# AI Agent Implementation Specification: Novel Algorithmic Similarity (NAS) for Collaborative Filtering

## 1. Objective & Scope
Implement and benchmark the **Novel Algorithmic Similarity (NAS)** measure for memory-based User-Based Collaborative Filtering (UBCF), as proposed by Birim & Tümtürk (2022) in *"A Novel Algorithmic Similarity Measure for Collaborative Filtering: A Recommendation System Based on Rating Distances"*.

The implementation must:
1. Reproduce the exact similarity scores on the paper's $4 \times 4$ sample user-item matrix (Table 1 $\to$ Table 2 & Table 3) as a mandatory unit test.
2. Implement **NAS** alongside the five benchmark similarity measures: **PCC**, **COS**, **CPC**, **PIP**, and **NHSM**.
3. Execute the experimental evaluation pipeline on **MovieLens 100K (ML-100K)** and **FilmTrust** across neighbor sizes $K \in \{10, 20, 30, 40, 60, 80\}$.
4. Compute and plot **MAE**, **RMSE**, **Precision**, and **Recall** (with relevance threshold $T = 4$ for ML-100K and an appropriate upper-quartile threshold for FilmTrust).

---

## 2. Recommended Tech Stack & Project Structure

### 2.1 Tech Stack
* **Language:** Python $3.10+$
* **Core Libraries:** `numpy`, `scipy`, `pandas` (for vectorized matrix operations and data manipulation)
* **Optimization (Optional but Recommended):** `numba` (via `@njit(parallel=True)`) for pairwise user-similarity loops over $O(|U|^2)$ user pairs
* **Testing:** `pytest`
* **Visualization:** `matplotlib`, `seaborn`

### 2.2 Directory Structure
```text
nas_collaborative_filtering/
├── data/
│   ├── ml-100k/                  # MovieLens 100K dataset (u.data)
│   └── filmtrust/                # FilmTrust dataset (ratings.txt)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, rating scales, K values
│   ├── data_loader.py            # Dataset fetching, parsing, and 70/30 splitting
│   ├── similarities/
│   │   ├── __init__.py
│   │   ├── traditional.py        # PCC, COS, CPC implementations
│   │   ├── heuristic.py          # PIP (Ahn 2008) and NHSM (Liu et al. 2014)
│   │   └── nas.py                # Proposed NAS implementation (Eqs. 10-20)
│   ├── recommender.py            # Top-K neighbor selection & rating prediction (Eq. 21)
│   ├── metrics.py                # MAE, RMSE, Precision@T, Recall@T (Eqs. 22-25)
│   └── visualize.py              # Reproduction of Figures 1a-4b
├── tests/
│   └── test_table1_sample.py     # Exact numerical verification against Tables 1, 2, 3
├── run_experiments.py            # End-to-end CLI runner
└── requirements.txt
```

---

## 3. Mathematical Specification & Edge-Case Rules for AI Agents

Let $U = \{u_1, u_2, \dots, u_n\}$ be the set of users and $P = \{p_1, p_2, \dots, p_m\}$ be the set of items. Let $R = \{r_{u,j}\}$ be the $n \times m$ user-item rating matrix, where $r_{u,j} = 0$ (or `NaN`) indicates an unrated item.

### 3.1 Global & Per-Entity Precomputed Quantities
Before computing pairwise similarities on training ratings:
1. **Median of Rating Scale ($r_{med}$):**
   * Defined by the rating scale bounds $[r_{min}, r_{max}]$:
     $$r_{med} = \frac{r_{min} + r_{max}}{2}$$
   * For **Table 1 & ML-100K**: $r_{min} = 1.0, r_{max} = 5.0 \implies r_{med} = 3.0$.
   * For **FilmTrust**: $r_{min} = 0.5, r_{max} = 4.0 \implies r_{med} = 2.25$.
2. **User Rated Item Set ($A_u$):**
   * $A_u = \{j \in P \mid r_{u,j} \text{ is observed in training}\}$, with cardinality $n(A_u) = |A_u|$.
   * Common items between $u_a$ and $u_b$: $C = A_{u_a} \cap A_{u_b}$. If $|C| = 0$, return similarity $0.0$.
3. **User Mean ($\bar{r}_u$) and Standard Deviation ($\sigma_u$):**
   * Computed over all observed ratings $j \in A_u$ for user $u$.
4. **Item Mean ($\bar{r}_j$):**
   * Computed over all training users $u \in U_j$ who rated item $j$:
     $$\bar{r}_j = \frac{1}{|U_j|} \sum_{u \in U_j} r_{u,j}$$
5. **Sigmoid Function (Eq. 9):**
   $$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}$$

---

### 3.2 Novel Algorithmic Similarity (NAS) — Step-by-Step Implementation
> **CRITICAL CORRECTION TO PAPER TYPOS:**
> 1. In Equation (10) and Equation (20), the paper omits the explicit summation symbol over $j \in C$, stating it only in prose in Step 4 (Page 63). You **must** sum the product of Proximity and Singularity over all $j \in C$:
>    $$PS(u_a, u_b) = \sum_{j \in C} \text{Proximity}(s_{u_a, j}, s_{u_b, j}) \cdot \text{Singularity}(r_{u_a, j}, r_{u_b, j})$$
> 2. In Equation (17), the paper has an index typo writing $\hat{b}_j = \frac{1}{N}\sum_{j=1}^N |r_{u_j} - r_{med}|$. $\hat{b}_j$ is an **item-specific** scale parameter averaged across the set of users $U_j$ who rated item $j$.

#### Step 1: User Factor ($UF_u$) — Eq. (11)
For each user $u \in U$, compute their tendency to rate close to the median across **all** items they rated ($A_u$):
$$UF_u = 1 - \text{sigmoid}\left( \frac{\sum_{j \in A_u} |r_{u,j} - r_{med}|}{n(A_u)} \right)$$
* *Edge Case:* If $n(A_u) = 0$, set $UF_u = 0.0$.

#### Step 2: Updated Score ($s_{u,j}$) — Eq. (12)
For every observed rating $r_{u,j}$, adjust the rating away from the median using $UF_u$:
$$s_{u,j} = \begin{cases} r_{u,j} + UF_u & \text{if } r_{u,j} > r_{med} \\ r_{u,j} - UF_u & \text{if } r_{u,j} < r_{med} \\ r_{u,j} & \text{if } r_{u,j} = r_{med} \end{cases}$$

#### Step 3: Proximity and Singularity on Common Items ($j \in C$) — Eqs. (10, 13, 14)
For a pair of users $(u_a, u_b)$ and each common item $j \in C = A_{u_a} \cap A_{u_b}$:
$$\text{Proximity}(s_{u_a, j}, s_{u_b, j}) = 1 - \text{sigmoid}\left( |s_{u_a, j} - s_{u_b, j}| \right)$$
$$\text{Singularity}(r_{u_a, j}, r_{u_b, j}) = \text{sigmoid}\left( |r_{u_a, j} - \bar{r}_j| \cdot |r_{u_b, j} - \bar{r}_j| \right)$$
Aggregate over $C$ to obtain $PS(u_a, u_b)$:
$$PS(u_a, u_b) = \sum_{j \in C} \text{Proximity}(s_{u_a, j}, s_{u_b, j}) \cdot \text{Singularity}(r_{u_a, j}, r_{u_b, j})$$

#### Step 4: Surprisal-Based Vector Similarity ($SVS$) — Eqs. (15, 16, 17, 18)
1. For each item $j \in P$, compute the Laplace scale parameter $\hat{b}_j$ over all users $U_j$ who rated item $j$:
   $$\hat{b}_j = \frac{1}{|U_j|} \sum_{u \in U_j} |r_{u,j} - r_{med}|$$
   * *Edge Case:* If $\hat{b}_j < \epsilon$ (e.g., all users gave item $j$ a rating equal to $r_{med}$), clamp $\hat{b}_j = \max(\hat{b}_j, 10^{-6})$ to prevent $\ln(0)$ and division by zero.
2. For each observed rating $r_{u,j}$, compute the quantity of information $I(r_{u,j})$:
   $$I(r_{u,j}) = \ln(2\hat{b}_j) + \frac{|r_{u,j} - r_{med}|}{\hat{b}_j}$$
3. Compute the signed surprisal entry $S_{u,j}$:
   $$S_{u,j} = \text{sign}(r_{u,j} - r_{med}) \cdot I(r_{u,j})$$
   where $\text{sign}(x) = 1$ if $x > 0$, $-1$ if $x < 0$, and $0$ if $x = 0$.
4. For user pair $(u_a, u_b)$, compute the cosine similarity of their surprisal vectors over $j \in C$:
   $$SVS(u_a, u_b) = \frac{\sum_{j \in C} S_{a,j} \cdot S_{b,j}}{\sqrt{\sum_{j \in C} S_{a,j}^2} \cdot \sqrt{\sum_{j \in C} S_{b,j}^2}}$$
   * *Edge Case:* Notice that Eq. (18) restricts both the numerator and denominator summations to $j \in C$. If either denominator norm is $0$ (which happens if a user rated all $j \in C$ as $r_{med}$), set $SVS(u_a, u_b) = 0.0$.

#### Step 5: Jaccard Similarity & Final NAS Score — Eqs. (19, 20)
$$\text{Jaccard}(u_a, u_b) = \frac{|A_{u_a} \cap A_{u_b}|}{|A_{u_a} \cup A_{u_b}|} = \frac{|C|}{n(A_{u_a}) + n(A_{u_b}) - |C|}$$
$$NAS(u_a, u_b) = PS(u_a, u_b) \cdot SVS(u_a, u_b) \cdot \text{Jaccard}(u_a, u_b)$$

---

### 3.3 Benchmark Similarity Specifications
To reproduce Figures 1a–4b, implement the five baseline metrics:

1. **Pearson Correlation Coefficient (PCC — Eq. 1):**
   $$PCC(u_a, u_b) = \frac{\sum_{j \in C} (r_{u_a, j} - \bar{r}_{u_a})(r_{u_b, j} - \bar{r}_{u_b})}{\sqrt{\sum_{j \in C} (r_{u_a, j} - \bar{r}_{u_a})^2} \sqrt{\sum_{j \in C} (r_{u_b, j} - \bar{r}_{u_b})^2}}$$
   *(Note: In Table 2, $\bar{r}_u$ is computed over $C$. If denominator is $0$, return $0.0$.)*

2. **Cosine Similarity (COS — Eq. 2):**
   $$COS(u_a, u_b) = \frac{\sum_{j \in C} r_{u_a, j} \cdot r_{u_b, j}}{\sqrt{\sum_{j \in C} r_{u_a, j}^2} \sqrt{\sum_{j \in C} r_{u_b, j}^2}}$$

3. **Constrained Pearson Correlation (CPC — Eq. 3):**
   $$CPC(u_a, u_b) = \frac{\sum_{j \in C} (r_{u_a, j} - r_{med})(r_{u_b, j} - r_{med})}{\sqrt{\sum_{j \in C} (r_{u_a, j} - r_{med})^2} \sqrt{\sum_{j \in C} (r_{u_b, j} - r_{med})^2}}$$

4. **Proximity-Impact-Popularity (PIP — Ahn 2008, Eqs. 4–5):**
   * Agreement boolean $\text{Agree}(r_1, r_2)$: `True` if $(r_1 > r_{med} \text{ and } r_2 > r_{med})$ or $(r_1 < r_{med} \text{ and } r_2 < r_{med})$, else `False` (treat $r = r_{med}$ as non-disagreeing or follow Ahn 2008).
   * Distance $D(r_1, r_2) = |r_1 - r_2|$ if $\text{Agree}(r_1, r_2)$ else $2|r_1 - r_2|$.
   * $\text{Proximity}(r_1, r_2) = \left( (2(r_{max} - r_{min}) + 1) - D(r_1, r_2) \right)^2$.
   * $\text{Impact}(r_1, r_2) = (|r_1 - r_{med}| + 1)(|r_2 - r_{med}| + 1)$ if $\text{Agree}(r_1, r_2)$ else $\frac{1}{(|r_1 - r_{med}| + 1)(|r_2 - r_{med}| + 1)}$.
   * $\text{Popularity}(r_1, r_2, \bar{r}_j) = 1 + \left(\frac{r_1 + r_2}{2} - \bar{r}_j\right)^2$ if $(r_1 > \bar{r}_j \text{ and } r_2 > \bar{r}_j)$ or $(r_1 < \bar{r}_j \text{ and } r_2 < \bar{r}_j)$, else $1$.
   * $SIM_{PIP}(u_a, u_b) = \sum_{j \in C} \text{Proximity}(r_{u_a, j}, r_{u_b, j}) \cdot \text{Impact}(r_{u_a, j}, r_{u_b, j}) \cdot \text{Popularity}(r_{u_a, j}, r_{u_b, j}, \bar{r}_j)$.

5. **New Heuristic Similarity Measure (NHSM — Liu et al. 2014, Eqs. 6–8):**
   * For each $j \in C$:
     * $\text{Proximity}_{PSS}(r_1, r_2) = 1 - \text{sigmoid}(|r_1 - r_2|)$
     * $\text{Significance}(r_1, r_2) = \text{sigmoid}(|r_1 - r_{med}| \cdot |r_2 - r_{med}|)$
     * $\text{Singularity}_{PSS}(r_1, r_2, \bar{r}_j) = 1 - \text{sigmoid}\left(\left|\frac{r_1 + r_2}{2} - \bar{r}_j\right|\right)$
     * $PSS(r_1, r_2) = \text{Proximity}_{PSS} \cdot \text{Significance} \cdot \text{Singularity}_{PSS}$
   * $J_{NHSM}(u_a, u_b) = \frac{|A_{u_a} \cap A_{u_b}|}{n(A_{u_a}) \cdot n(A_{u_b})}$ (Eq. 7)
   * $URP(u_a, u_b) = 1 - \text{sigmoid}(|\bar{r}_{u_a} - \bar{r}_{u_b}| \cdot |\sigma_{u_a} - \sigma_{u_b}|)$ *(Note: Eq. 8 in the paper prints $|\sigma_a * \sigma_b|$, whereas Liu et al. 2014 defines $|\sigma_a - \sigma_b|$. Notice in Table 1 all users have $\sigma_u = 0.4330$, so $|\sigma_a - \sigma_b| = 0$. To match Table 2's NHSM values `[0.0174, 0.0078, 0.0216, ...]`, check both $|\sigma_a - \sigma_b|$ and $|\sigma_a \cdot \sigma_b|$ in unit testing).*
   * $NHSM(u_a, u_b) = \left(\sum_{j \in C} PSS(r_{u_a, j}, r_{u_b, j})\right) \cdot J_{NHSM}(u_a, u_b) \cdot URP(u_a, u_b)$.

---

## 4. Experimental Protocol & Evaluation Pipeline

### 4.1 Train/Test Split Protocol (Section 4.1)
1. **User Split:** Randomly partition the set of all users $U$ into $70\%$ **training users** ($U_{train}$) and $30\%$ **testing users** ($U_{test}$). Use a fixed random seed (e.g., `seed=42`) and support $k$-fold or repeated random sub-sampling for stability.
2. **Test-User Item Split:** For each testing user $u_t \in U_{test}$, randomly select $70\%$ of their rated items as **active/probe training ratings** ($A_{u_t}^{train}$) used for computing similarity and user statistics ($\bar{r}_{u_t}, UF_{u_t}$), and hold out the remaining $30\%$ ($A_{u_t}^{test}$) as **target items** for prediction and evaluation.
3. **Candidate Neighbors:** For a target test user $u_t$ and target item $p \in A_{u_t}^{test}$, select the top-$K$ most similar neighbors $N_K(u_t)$ (or among training users who have rated item $p$, ordered by descending $\text{sim}(u_t, u_i)$).

### 4.2 Rating Prediction Formula (Section 4.2, Eq. 21)
*(Note: Equation 21 in the PDF is partially garbled due to font encoding. The standard mean-centered Resnick prediction formula referenced from [19] and [34] is:)*
$$\hat{r}_{u_t, p} = \bar{r}_{u_t} + \frac{\sum_{u_i \in K} \text{sim}_{u_t, u_i} \cdot (r_{u_i, p} - \bar{r}_{u_i})}{\sum_{u_i \in K} |\text{sim}_{u_t, u_i}|}$$
* **Fallback Rule:** If $\sum_{u_i \in K} |\text{sim}_{u_t, u_i}| = 0$ or no neighbor in $K$ has rated item $p$, fall back to $\hat{r}_{u_t, p} = \bar{r}_{u_t}$. Optionally clip $\hat{r}_{u_t, p}$ to $[r_{min}, r_{max}]$.

### 4.3 Evaluation Metrics (Section 4.2, Eqs. 22–25)
Evaluate across $K \in \{10, 20, 30, 40, 60, 80\}$:
1. **Mean Absolute Error (MAE — Eq. 22):**
   $$MAE = \frac{1}{N_{pred}} \sum_{j=1}^{N_{pred}} |r_j - \hat{r}_j|$$
2. **Root Mean Squared Error (RMSE — Eq. 23):**
   $$RMSE = \sqrt{\frac{1}{N_{pred}} \sum_{j=1}^{N_{pred}} (r_j - \hat{r}_j)^2}$$
3. **Classification Accuracy (Recall & Precision — Eqs. 24–25):**
   * Define relevance threshold $T$: for ML-100K, $T = 4.0$ (as shown in Figures 3a and 4a: `RECALL T=4`, `PRECISION T=4`). For FilmTrust (scale $0.5$ to $4.0$), test $T = 3.0$ or $T = 3.5$.
   * For each test user $u_t$:
     * $\text{Liked}_{actual} = \{p \in A_{u_t}^{test} \mid r_{u_t, p} \ge T\}$
     * $\text{Recommended} = \{p \in A_{u_t}^{test} \mid \hat{r}_{u_t, p} \ge T\}$ (or top-$N$ recommended items where $\hat{r}_{u_t, p} \ge T$)
     $$\text{Recall} = \frac{|\text{Liked}_{actual} \cap \text{Recommended}|}{|\text{Liked}_{actual}|}, \quad \text{Precision} = \frac{|\text{Liked}_{actual} \cap \text{Recommended}|}{|\text{Recommended}|}$$

---

## 5. Phased Execution Plan & Acceptance Criteria

### Phase 1: Toy Matrix Unit Test (`tests/test_table1_sample.py`) — MANDATORY FIRST STEP
Before loading any external dataset, implement `src/similarities/nas.py` and run `pytest` on the $4 \times 4$ matrix from Table 1:
$$R_{sample} = \begin{bmatrix} 2 & 1 & 2 & 2 \\ 5 & 4 & 5 & 5 \\ 4 & 5 & 4 & 4 \\ 1 & 2 & 1 & 1 \end{bmatrix}, \quad r_{med} = 3.0$$
**Acceptance Criterion:** The unit test MUST assert that `compute_nas_matrix(R_sample, r_med=3.0)` matches **Table 3** to $4$ decimal places (`atol=1e-4`):
* `NAS(User1, User2) == -0.1126`
* `NAS(User1, User3) == -0.1863`
* `NAS(User1, User4) ==  0.9376`
* `NAS(User2, User3) ==  0.9376`
* `NAS(User2, User4) == -0.1064`
* `NAS(User3, User4) == -0.1126`

### Phase 2: Baseline Similarities & Data Pipeline
1. Implement `traditional.py` (PCC, COS, CPC) and `heuristic.py` (PIP, NHSM) and verify against Table 2.
2. Implement `data_loader.py` to automatically download/parse **MovieLens 100K** ($943$ users, $1682$ items, $100,000$ ratings) and **FilmTrust** ($1508$ users, $2071$ items, $35,497$ ratings).
3. Implement the $70/30$ user and $70/30$ test-item splitter.

### Phase 3: Vectorized / Numba-Accelerated Pairwise Engine
1. Computing $943 \times 943$ or $1508 \times 1508$ pairwise similarities in pure Python loops is slow. Precompute $UF_u$, $s_{u,j}$, $\bar{r}_j$, $\hat{b}_j$, and $S_{u,j}$ as dense/sparse arrays, and use NumPy broadcasting or `@njit` to compute the full similarity matrix in $< 2$ seconds per metric.

### Phase 4: Experiment Runner & Figure Generation
1. Run `run_experiments.py` across all $6$ similarity metrics and $K \in \{10, 20, 30, 40, 60, 80\}$ on both datasets.
2. Save tabular CSV results and generate the $8$ benchmark plots corresponding to Figures 1a, 1b, 2a, 2b, 3a, 3b, 4a, and 4b.