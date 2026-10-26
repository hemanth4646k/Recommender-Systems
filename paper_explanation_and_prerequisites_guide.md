# Understanding the Paper & Implementation: A Complete Guide

**Paper Title:** *A Novel Algorithmic Similarity Measure for Collaborative Filtering: A Recommendation System Based on Rating Distances*  
**Authors:** Şule Öztürk Birim & Ayça Tümtürk (2022), *Academic Platform Journal of Engineering and Smart Systems (APJESS)*, 10(1), 57–69.

---

## Part 1: Prerequisites to Look Up Separately

If you want to build a rock-solid foundation before or while reading this paper, here are the core concepts ordered from foundational to advanced, along with exact search terms to look up:

### 1. Recommender Systems & Collaborative Filtering Basics
* **Memory-Based vs. Model-Based Collaborative Filtering:**
  * *What it is:* Memory-based methods (like the one in this paper) use the entire user-item rating database directly to find similar users or items using a mathematical formula. Model-based methods train a machine learning model (like Matrix Factorization or Neural Networks) offline.
  * *Search terms:* `"User-Based Collaborative Filtering (UBCF)"`, `"Memory-based vs Model-based collaborative filtering"`, `"Resnick prediction formula collaborative filtering"`.
* **Sparsity and the Cold-Start Problem:**
  * *What it is:* In real e-commerce or movie platforms, most users rate less than $1\%$ to $6\%$ of available items. When two users share very few commonly rated items, traditional similarity metrics break down.
  * *Search terms:* `"Data sparsity and cold start problem in recommender systems"`.

### 2. Mathematical & Statistical Building Blocks
* **Vector Spaces & Classical Similarity Metrics:**
  * *What to look up:* How **Cosine Similarity** measures the angle between two vectors, how **Pearson Correlation Coefficient (PCC)** measures linear correlation around the mean, and how **Jaccard Similarity** measures set overlap ($\frac{|A \cap B|}{|A \cup B|}$).
  * *Search terms:* `"Cosine similarity vs Pearson correlation recommender systems"`, `"Jaccard similarity index"`.
* **The Sigmoid (Logistic) Function:**
  * *What it is:* $\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}$. It squashes any real number into the range $(0, 1)$. For non-negative distances $x \ge 0$, $\text{sigmoid}(x) \in [0.5, 1)$, so $1 - \text{sigmoid}(x) \in (0, 0.5]$ acts as a smooth decay function that rewards small distances and penalizes large ones.
  * *Search terms:* `"Logistic sigmoid function properties normalization"`.
* **Likert Scales & Central Tendency Bias:**
  * *What it is:* On a $1$-to-$5$ rating scale (Likert scale), the midpoint is the median ($r_{med} = 3$). In psychology and survey research, **central tendency bias** is the tendency of certain individuals to avoid extreme ratings ($1$ or $5$) and cluster their scores around the median ($2, 3, 4$).
  * *Search terms:* `"Central tendency bias Likert scale"`, `"Constrained Pearson Correlation (CPC) Shardanand Maes"`.

### 3. Advanced Heuristic & Information-Theoretic Concepts (Direct Predecessors to NAS)
* **PIP (Proximity-Impact-Popularity) & NHSM (New Heuristic Similarity Measure):**
  * *What they are:* Two famous papers by Ahn (2008) and Liu et al. (2014) that replaced Pearson/Cosine with multi-factor heuristics combining rating distance (*Proximity*), extremity (*Impact/Significance*), and item uniqueness (*Popularity/Singularity*).
  * *Search terms:* `"PIP similarity measure Ahn 2008"`, `"NHSM Liu et al 2014 collaborative filtering"`.
* **Surprisal & Self-Information (Information Theory) + Laplace Distribution MLE:**
  * *What it is:* In information theory, common events carry little information, whereas rare/extreme events carry high information ("surprisal"). Luo et al. (2008) modeled user ratings around an average attitude using a Laplace distribution, where the scale parameter $\hat{b}_j$ is estimated via Maximum Likelihood Estimation (MLE) as the mean absolute deviation.
  * *Search terms:* `"Self-information surprisal information theory"`, `"Laplace distribution maximum likelihood estimator mean absolute deviation"`.

---

## Part 2: Core Motivation — What Problem Does the Paper Solve?

Look at **Table 1** from the paper, where $4$ users rate $4$ items on a $1$-to-$5$ scale ($r_{med} = 3$):

| User | Item-1 | Item-2 | Item-3 | Item-4 | Intuitive Meaning of User's Ratings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **User-1** | $2$ | $1$ | $2$ | $2$ | **Dislikes** all items (avoids extreme $1$s mostly) |
| **User-2** | $5$ | $4$ | $5$ | $5$ | **Loves** all items (gives mostly $5$s) |
| **User-3** | $4$ | $5$ | $4$ | $4$ | **Loves** all items (cautious rater: gives mostly $4$s) |
| **User-4** | $1$ | $2$ | $1$ | $1$ | **Dislikes** all items (gives mostly $1$s) |

### Why Traditional Metrics Fail Here (Table 2):
1. **Pearson Correlation (PCC) Failure:**
   * Compare **User-1** ($[2, 1, 2, 2]$) and **User-2** ($[5, 4, 5, 5]$). User-1 hates every movie; User-2 loves every movie. Yet their ratings move up and down in the exact same pattern ($r_{2,j} = r_{1,j} + 3$).
   * Consequently, $PCC(\text{User-1}, \text{User-2}) = +1.0$ (perfect positive correlation!), while $PCC(\text{User-2}, \text{User-3}) = -1.0$ even though both User-2 and User-3 love all four movies!
2. **Cosine Similarity (COS) Failure:**
   * Because $[2, 1, 2, 2]$ and $[5, 4, 5, 5]$ point in nearly the same geometric direction in $\mathbb{R}^4$, $COS(\text{User-1}, \text{User-2}) = 0.9885$ (the highest similarity in the table!), whereas $COS(\text{User-1}, \text{User-4}) = 0.8386$ (the lowest similarity, even though both dislike all four items!).
3. **Why PIP and NHSM Still Fall Short:**
   * **PIP** does not normalize by the proportion of common items and repeatedly squares/penalizes differences, causing explosive variance (ranging from $6$ to $3822$ on Table 1) and no negative signs for opposite preferences.
   * **NHSM** modifies Jaccard into $\frac{|A \cap B|}{|A| \cdot |B|}$, which unfairly penalizes active users who have rated many items in common (e.g., if two users both rate $100$ items in common, their factor is $\frac{100}{100 \times 100} = 0.01$!).
   * **Crucially:** Neither PIP nor NHSM accounts for **central tendency bias**—the fact that User-3's rating of $4$ represents just as strong a preference for User-3 as User-2's rating of $5$ does for User-2, because User-3 naturally tends to rate closer to the median ($3$).

---

## Part 3: Step-by-Step Mathematical Breakdown of NAS

The **Novel Algorithmic Similarity (NAS)** combines three multiplicative components for any pair of users $(u_a, u_b)$:
$$NAS(u_a, u_b) = PS(u_a, u_b) \times SVS(u_a, u_b) \times \text{Jaccard}(u_a, u_b)$$

Here is how each component works and why it is designed that way:

### Component 1: Proximity-Singularity ($PS$) with Median-Tendency Adjustment
#### Step 1A: User Factor ($UF_u$) — Measuring Central Tendency Bias (Eq. 11)
First, for each user $u$, we measure their average distance from the scale median $r_{med}$ across all items they rated ($A$):
$$UF_u = 1 - \text{sigmoid}\left( \frac{\sum_{r \in A} |r - r_{med}|}{n(A)} \right)$$
* **Intuition:** If a user always rates close to the median (small average $|r - r_{med}|$), the sigmoid is small, making $UF_u$ **larger**. If a user frequently uses extreme ratings ($1$ and $5$), their average deviation from $r_{med}$ is large, making $UF_u$ **smaller**.

#### Step 1B: Adjusting Ratings Away from the Median ($s_{u,j}$) (Eq. 12)
To put cautious raters (like User-3) on equal footing with extreme raters (like User-2), we stretch each rating away from $r_{med}$ by adding or subtracting $UF_u$:
$$s_{u,j} = \begin{cases} r_{u,j} + UF_u & \text{if } r_{u,j} > r_{med} \\ r_{u,j} - UF_u & \text{if } r_{u,j} < r_{med} \\ r_{u,j} & \text{if } r_{u,j} = r_{med} \end{cases}$$

#### Step 1C: Proximity & Singularity on Common Items (Eqs. 10, 13, 14)
For each item $j \in C$ rated by both $u_a$ and $u_b$:
1. **Proximity** measures how close their $UF$-adjusted ratings are:
   $$\text{Proximity}(s_{u_a, j}, s_{u_b, j}) = 1 - \text{sigmoid}\left( |s_{u_a, j} - s_{u_b, j}| \right)$$
2. **Singularity** measures how much both users' raw ratings deviate from the item's average rating $\bar{r}_j$:
   $$\text{Singularity}(r_{u_a, j}, r_{u_b, j}) = \text{sigmoid}\left( |r_{u_a, j} - \bar{r}_j| \cdot |r_{u_b, j} - \bar{r}_j| \right)$$
   * **Intuition:** If an item has an average rating of $\bar{r}_j = 3.0$ and both users rate it $5$, they share a distinctive opinion on that item ($|5-3|\cdot|5-3| = 4 \implies \text{high singularity}$).
3. **Sum over Common Items ($PS$):**
   $$PS(u_a, u_b) = \sum_{j \in C} \text{Proximity}(s_{u_a, j}, s_{u_b, j}) \cdot \text{Singularity}(r_{u_a, j}, r_{u_b, j})$$

---

### Component 2: Surprisal-Based Vector Similarity ($SVS$) (Eqs. 15–18)
 Notice that $PS(u_a, u_b)$ is always positive. How does NAS know whether two users agree in the **same direction** (positive similarity) or **opposite directions** (negative similarity), and how does it weight extreme preferences? It uses **Surprisal-Based Vector Similarity ($SVS$)** adapted from Luo et al. (2008), replacing the mean with the median $r_{med}$:

1. **Item Scale Parameter ($\hat{b}_j$ — Eq. 17):** For each item $j$, compute the average distance of all users' ratings from $r_{med}$:
   $$\hat{b}_j = \frac{1}{N} \sum_{i=1}^{N} |r_{u_i, j} - r_{med}|$$
2. **Quantity of Information ($I(r_{u,j})$ — Eq. 16):** Based on the negative log-likelihood (self-information) of a Laplace distribution centered at $r_{med}$:
   $$I(r_{u,j}) = \ln(2\hat{b}_j) + \frac{|r_{u,j} - r_{med}|}{\hat{b}_j}$$
   * Ratings further from $r_{med}$ carry more "surprisal" (larger $I$).
3. **Signed Surprisal Vector ($S_u$ — Eq. 15):** Attach the sign of $(r_{u,j} - r_{med})$ so positive attitudes ($r > r_{med}$) are positive and negative attitudes ($r < r_{med}$) are negative:
   $$S_{u,j} = \text{sign}(r_{u,j} - r_{med}) \cdot I(r_{u,j})$$
4. **Vector Cosine of Surprisal Vectors ($SVS$ — Eq. 18):**
   $$SVS(u_a, u_b) = \frac{\sum_{j \in C} S_{a,j} \cdot S_{b,j}}{\sqrt{\sum_{j \in C} S_{a,j}^2} \cdot \sqrt{\sum_{j \in C} S_{b,j}^2}} \in [-1, 1]$$
   * Because $S_{u,j}$ is signed relative to $r_{med}$, $SVS$ naturally gives a **positive** value when users agree and a **negative** value when one likes what the other dislikes!

---

### Component 3: True Jaccard Proportion (Eq. 19)
Instead of NHSM's distorted denominator, NAS multiplies by the classical Jaccard index:
$$\text{Jaccard}(u_a, u_b) = \frac{|A_{u_a} \cap A_{u_b}|}{|A_{u_a} \cup A_{u_b}|}$$

---

## Part 4: Worked Numerical Example (Verifying Table 3 by Hand)

Let's verify **User-1** ($[2, 1, 2, 2]$) and **User-4** ($[1, 2, 1, 1]$) from Table 1 ($r_{med} = 3$) to see how the paper gets **`0.9376`** in Table 3:

1. **User Factors ($UF_1, UF_4$):**
   * User-1 deviations $|r - 3|$: $[1, 2, 1, 1] \implies \text{mean} = \frac{5}{4} = 1.25$.
     $$UF_1 = 1 - \text{sigmoid}(1.25) = 1 - 0.7773 = 0.2227$$
   * User-4 deviations $|r - 3|$: $[2, 1, 2, 2] \implies \text{mean} = \frac{7}{4} = 1.75$.
     $$UF_4 = 1 - \text{sigmoid}(1.75) = 1 - 0.8520 = 0.1480$$

2. **Updated Scores ($s_1, s_4$):** (Since all ratings for User-1 and User-4 are $< 3$, subtract $UF_u$):
   * $s_1 = [2 - 0.2227, 1 - 0.2227, 2 - 0.2227, 2 - 0.2227] = [1.7773, 0.7773, 1.7773, 1.7773]$
   * $s_4 = [1 - 0.1480, 2 - 0.1480, 1 - 0.1480, 1 - 0.1480] = [0.8520, 1.8520, 0.8520, 0.8520]$

3. **Proximity & Singularity ($PS$):**
   * Distance $|s_{1,j} - s_{4,j}|$:
     * Items 1, 3, 4: $|1.7773 - 0.8520| = 0.9253 \implies \text{Proximity} = 1 - \text{sigmoid}(0.9253) = 0.2839$.
     * Item 2: $|0.7773 - 1.8520| = 1.0747 \implies \text{Proximity} = 1 - \text{sigmoid}(1.0747) = 0.2545$.
   * Item column means in Table 1: every column has ratings $\{1, 2, 4, 5\}$, so $\bar{r}_j = 3.0$ for all $j$.
   * For all $4$ items, $|r_{1,j} - 3| \cdot |r_{4,j} - 3| = 1 \times 2 = 2 \implies \text{Singularity} = \text{sigmoid}(2) = 0.8808$.
   * Summing across the $4$ items:
     $$PS(u_1, u_4) = (3 \times 0.2839 + 0.2545) \times 0.8808 = 1.1061 \times 0.8808 = 0.9743$$

4. **Surprisal Vector Similarity ($SVS$):**
   * For each item $j$, $|r_{u,j} - 3|$ across the $4$ users is $\{1, 2, 1, 2\}$, so $\hat{b}_j = \frac{1+2+1+2}{4} = 1.5$.
   * When $|r - 3| = 1$: $I = \ln(2 \times 1.5) + \frac{1}{1.5} = \ln(3) + 0.6667 = 1.7653$.
   * When $|r - 3| = 2$: $I = \ln(3) + \frac{2}{1.5} = 2.4319$.
   * Thus:
     * $S_1 = [-1.7653, -2.4319, -1.7653, -1.7653]$
     * $S_4 = [-2.4319, -1.7653, -2.4319, -2.4319]$
   * Cosine of $S_1$ and $S_4$:
     $$SVS(u_1, u_4) = \frac{4 \times (1.7653 \times 2.4319)}{\sqrt{3(1.7653)^2 + (2.4319)^2} \cdot \sqrt{(1.7653)^2 + 3(2.4319)^2}} = \frac{17.1723}{3.9068 \times 4.5672} = 0.9624$$

5. **Final NAS Score:**
   * Since both users rated all $4$ items, $\text{Jaccard}(u_1, u_4) = \frac{4}{4} = 1.0$.
   * Multiplying the three parts:
     $$NAS(u_1, u_4) = 0.9743 \times 0.9624 \times 1.0 = \mathbf{0.9376}$$
   This matches **Table 3 (`0.9376`)** to the exact ten-thousandth place!

---

## Part 5: Important Errata & Quirks in the Original Paper

When reading the PDF, keep the following typographical errors in mind so they do not confuse you:
1. **Missing Summation in Equations (10) and (20):** Equation (10) defines $PS(r_1, r_2)$ for a single rating pair, and Equation (20) multiplies $PS(u_a, u_b)$ directly. As confirmed in Step 4 on page 63 and by our numerical proof above, $PS(u_a, u_b)$ is the **sum** of $PS(r_{u_a, j}, r_{u_b, j})$ over all commonly rated items $j \in C$.
2. **Subscript Typo in Equation (17):** The paper writes $\hat{b}_j = \frac{1}{N} \sum_{j=1}^N |r_{u_j} - r_{med}|$. The summation index should be over users $i = 1 \dots N$ who rated item $j$, i.e., $\hat{b}_j = \frac{1}{|U_j|} \sum_{u \in U_j} |r_{u,j} - r_{med}|$.
3. **Garbled Math Font in Equations (1), (2), (21), and (23):** Due to PDF font embedding issues in the journal print, square root symbols and summation subscripts disappeared in Equations (1), (2), (21), and (23). All four are standard formulas (Pearson, Cosine, Resnick mean-centered prediction, and RMSE) and have been fully restored in `plan.md`.
4. **Why $K \in \{20, 30, 40\}$ Performs Best (Section 5):** In Collaborative Filtering, if $K$ is too small ($K=10$), you exclude informative similar neighbors. If $K$ is too large ($K=60, 80$), you include "false neighbors" (dissimilar users) who add noise to the weighted average prediction. NAS shines brightest in the sweet spot $K \in \{20, 30, 40\}$.