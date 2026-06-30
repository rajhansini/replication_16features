# Value Function Approximation for Sequential Genetic Testing
## Mathematical Presentation

---

## SLIDE 1 — Problem Statement

**Clinical setting**: A family pedigree of N individuals may carry a hereditary disease
mutation. Each person can be genetically tested. Testing costs money. Untested
carriers incur future clinical costs. The goal is an optimal sequential testing policy:
*who to test next, and when to stop.*

**Two disease genes**: GeneA and GeneB. Each person's genotype per gene is
g ∈ {0, 1, 2} where 0 = NN (non-carrier), 1 = ND (heterozygous carrier), 2 = DD (affected).

**8 canonical problem instances** = 2 families × 2 allele regimes × 2 cost presets:

| Family          | N  | Structure                                   |
|-----------------|----|---------------------------------------------|
| ThreeGeneration | 5  | Grandparent + parents + proband + sibling   |
| Extended        | 6  | Grandparents + parents + uncle + proband    |

| Regime     | p(GeneA) | p(GeneB) |
|------------|----------|----------|
| LowHigh    | 0.02     | 0.15     |
| MediumEven | 0.08     | 0.08     |

| Preset     | a(GeneA) | a(GeneB) | b(GeneA) | b(GeneB) | δ(GeneA) | δ(GeneB) | c_fixed | c_var |
|------------|----------|----------|----------|----------|----------|----------|---------|-------|
| Base       | −0.08    | −0.06    | −0.04    | −0.03    | 0.60     | 0.70     | 0.01    | 0.02  |
| Aggressive | −0.12    | −0.09    | −0.06    | −0.045   | 0.70     | 0.80     | 0.01    | 0.02  |

Non-founder individuals (parents, proband) have 2× the a and b values of founders (grandparents)
via `child_multiplier=2.0` in the config — matching the PI's ABCD-16 package exactly.

---

## SLIDE 2 — Genetic Model

### 2.1 Population Prior (Hardy-Weinberg Equilibrium)

For a founder (no parents in pedigree), given allele frequency p:
```
P(g = 0) = (1 − p)²        (NN)
P(g = 1) = 2p(1 − p)       (ND)
P(g = 2) = p²               (DD)
```

### 2.2 Mendelian Inheritance CPD

For a child with parents of genotypes g_f, g_m, let transmission probability be:
```
f_pass(g) = 0    if g = 0
           = 0.5  if g = 1
           = 1    if g = 2
```
Then:
```
P(g_child = 0 | g_f, g_m) = (1 − f_pass(g_f)) · (1 − f_pass(g_m))
P(g_child = 2 | g_f, g_m) = f_pass(g_f) · f_pass(g_m)
P(g_child = 1 | g_f, g_m) = 1 − P(0) − P(2)
```
A 3×9 conditional probability table per gene.

### 2.3 Independent Assortment (Two Genes)

GeneA and GeneB segregate independently:
```
P(g_A, g_B | person) = P(g_A | person) · P(g_B | person)
```
This reduces the joint state space from O(9^N) to O(3^N) per gene.

---

## SLIDE 3 — Belief State & Exact Update

### 3.1 State and Belief

Observation set after k tests: O = {(i₁,o₁), ..., (iₖ,oₖ)}, oⱼ ∈ {0,1,2}².

Belief state (Bayesian posterior):
```
b_O(x) = P(X = x | O)
```

### 3.2 Belief Update

After observing (i*, g*):
```
P(X = x | O ∪ {(i*, g*)}) ∝ P(X = x | O) · 1[x_{i*} = g*]
```
Computed via **exact Bayesian network message passing** (belief propagation)
over the pedigree graph using pgmpy VariableElimination — deterministic,
computational precision, no approximation.

### 3.3 Per-Gene Marginals

Marginal posterior for person i, gene k:
```
b_O^k(i, g) = P(X_i^k = g | O)   for g ∈ {0, 1, 2}
```
Carrier probability:
```
p_i^k = b_O^k(i, 1) + b_O^k(i, 2)
```

### 3.4 Factorised Computation

For each gene k independently:
1. Build full joint P(X^k | O^k) over 3^N states via pedigree BN
2. Marginalise to b_O^k(i, ·) per person

Combined under independent assortment:
```
P(X_i^A = gA, X_i^B = gB | O) = b_O^A(i, gA) · b_O^B(i, gB)
```
Complexity: O(3^N) per gene.
States: ThreeGeneration = 20,816; Extended = 107,728.

---

## SLIDE 4 — Reward Model

### 4.1 Stopping Reward

For person i ∉ tested(O), per gene k:
```
r_stop(i, b_O, k) = a_k · (p_i^k − δ_k · (p_i^k)²)
                  + b_k · (p_i^k − (p_i^k)²)
```
- a_k < 0: cost of missing a carrier
- b_k < 0: uncertainty penalty (Bernoulli variance)
- δ_k: homozygosity discount

Total stopping value:
```
V_stop(O) = Σ_{i ∉ tested(O)} Σ_{k ∈ {A,B}} r_stop(i, b_O, k)
```

### 4.2 Testing Reward (Immediate)

```
r_test(i, b_O) = Σ_{k ∈ {A,B}} [ a_k · (1 − δ_k) · p_i^k ]
               − c_fixed
               − c_var · P(any positive | i, b_O)
```
where:
```
P(any positive | i, b_O) = 1 − Π_{k ∈ {A,B}} (1 − p_i^k)
```

---

## SLIDE 5 — POMDP, Optimal Value Function, and ratio2

### 5.1 Q-Function

```
Q*(O, STOP)   = V_stop(O)
Q*(O, test i) = r_test(i, b_O) + Σ_{o ∈ {0,1,2}²} P(o | i, b_O) · V*(O ∪ {(i,o)})
```

### 5.2 Bellman Optimality

```
V*(O) = max( V_stop(O),  max_{i ∉ tested(O)} Q*(O, test i) )
```
Boundary: V*(O) = 0 when tested(O) = {1,...,N}.

Solved exactly by **backward induction** over acyclic belief MDP:
enumerate states by |O| descending, apply Bellman equation at each.

### 5.3 Evaluation Metric: ratio2

Let L = true expected value of the approximate policy π̂ (computed by exact recursion,
no approximation substituted):
```
ratio2 = (V*(O₀) − L) / (V*(O₀) − V_stop(O₀))
```
```
ratio2 = 0  →  policy is perfectly optimal
ratio2 = 1  →  policy is as bad as stopping immediately
```
ratio2 = 0.05 means the policy captures 95% of achievable improvement.

True policy value L computed by exact recursion — V̂_θ is used only to choose
actions, not to evaluate them:
```
V^{π̂}(O) = r_test(π̂(O), b_O) + Σ_o P(o|π̂(O), b_O) · V^{π̂}(O ∪ {(π̂(O),o)})
           if π̂(O) ≠ STOP,  else  V_stop(O)
```

---

## SLIDE 6 — ADP Baseline (ABCD-16)

### 6.1 Approximate Linear Program

The ABCD-16 ADP constructs φ: S → ℝ^{16} (16 hand-crafted features) and solves:
```
min_{θ ∈ ℝ^{16}}  Σ_{s ∈ S} ρ(s) · θᵀφ(s)

subject to:
   θᵀφ(s) ≥ V_stop(s)                                         ∀s ∈ S
   θᵀφ(s) ≥ r_test(i,s) + Σ_o P(o|i,s) · θᵀφ(s')           ∀s ∈ S, ∀i ∉ tested(s)
```
Bellman inequality constraints ensure θᵀφ(s) upper-bounds V*(s) everywhere.
Rearranged second constraint:
```
[φ(s) − Σ_o P(o|i,s) φ(s')] · θ ≥ r_test(i,s)
```
Solved with **Gurobi** (Academic license 2839186). Per-stage θ (θ varies by stage |O|),
EXHAUSTIVE_BELLMAN=1. Solve time: ~300–360s (Extended), ~50–60s (ThreeGeneration).

### 6.2 ADP Greedy Policy

```
π_ADP(O) = argmax_{a ∈ A(O)} Q_ADP(O, a)

Q_ADP(O, STOP)   = V_stop(O)
Q_ADP(O, test i) = r_test(i, b_O) + Σ_o P(o|i,b_O) · θ*ᵀ φ(O ∪ {(i,o)})
```
Requires re-solving the LP per family (new parameters → new θ*).

---

## SLIDE 7 — MLP: Architecture & Results

### 7.1 Architecture

Input: belief vector zero-padded to max family size (6 people × 6 gene probs = 36 dims).
```
MLPValueNet: ℝ^{36} → ℝ^{128} → ℝ^{64} → ℝ^{32} → ℝ

h^{(l)} = ReLU(W^{(l)} h^{(l−1)} + b^{(l)})
V̂_θ(s) = W^{(4)} h^{(3)} + b^{(4)}    (linear output)
```
Parameters: 36×128 + 128×64 + 64×32 + 32×1 + biases = **15,105**

### 7.2 Training

Supervised on exact V*(s) labels from backward induction.
Train: 6 families (4 Extended + 2 ThreeGeneration×LowHigh), 472,544 states.
Test: 2 families (ThreeGeneration×MediumEven), never seen during training.
```
Loss:       L(θ) = (1/|S|) Σ_s (V̂_θ(s) − V*(s))²
Epochs:     800,  batch 512,  Adam lr=1×10⁻³
Final loss: 3.97×10⁻⁴
```

### 7.3 Results

ADP: freshly computed with Gurobi, per-stage θ, 16 ABCD features, same reward parameters.
```
Family                              Split   V*(∅)    V_stop(∅)  L(MLP)   r2(MLP)  r2(ADP)  vs ADP
────────────────────────────────────────────────────────────────────────────────────────────────────
Extended_LowHigh_Base               TRAIN  −0.1306  −0.2164    −0.1356   0.0573   0.0867   MLP 1.5×
Extended_LowHigh_Aggressive         TRAIN  −0.1337  −0.3182    −0.1382   0.0241   0.0501   MLP 2.1×
Extended_MediumEven_Base            TRAIN  −0.1421  −0.2563    −0.1447   0.0227   0.1069   MLP 4.7×
Extended_MediumEven_Aggressive      TRAIN  −0.1488  −0.3800    −0.1565   0.0334   0.0163   ADP 2.0×
ThreeGeneration_LowHigh_Base        TRAIN  −0.1092  −0.1683    −0.1107   0.0262   0.1090   MLP 4.2×
ThreeGeneration_LowHigh_Aggressive  TRAIN  −0.1114  −0.2475    −0.1149   0.0259   0.0430   MLP 1.7×
────────────────────────────────────────────────────────────────────────────────────────────────────
ThreeGeneration_MediumEven_Base     TEST   −0.1177  −0.1994    −0.1206   0.0357   0.0334   ADP 1.1×
ThreeGeneration_MediumEven_Aggressive TEST −0.1227  −0.2956    −0.1266   0.0223   0.0143   ADP 1.6×
```
MLP wins on 5/6 training families. **ADP wins on both held-out test families.**

---

## SLIDE 8 — GNN: Architecture & Results

### 8.1 Graph Representation

Each belief state → graph with one node per person.

**Node features** (7 per node):
```
h_i^{(0)} = [P(gA=0|i,O), P(gA=1|i,O), P(gA=2|i,O),
              P(gB=0|i,O), P(gB=1|i,O), P(gB=2|i,O),
              1_{i ∈ tested(O)}]  ∈ ℝ^7
```
**Edges**: directed parent → child (pedigree structure).

### 8.2 Message Passing (2 Rounds)

At each round t ∈ {1, 2}:
```
m_{j→i}^{(t)} = ReLU( W_msg^{(t)} [h_j^{(t−1)} ; h_i^{(t−1)}] + b_msg^{(t)} )

agg_i^{(t)} = (1/|N(i)|) Σ_{j ∈ N(i)} m_{j→i}^{(t)}

h_i^{(t)} = ReLU( W_upd^{(t)} [h_i^{(t−1)} ; agg_i^{(t)}] + b_upd^{(t)} )
```
Layer dimensions:
```
Round 1: W_msg^{(1)} ∈ ℝ^{32×14},  W_upd^{(1)} ∈ ℝ^{32×39}
Round 2: W_msg^{(2)} ∈ ℝ^{32×64},  W_upd^{(2)} ∈ ℝ^{32×64}
```

### 8.3 Readout

```
h_G = (1/N) Σ_i h_i^{(2)}  ∈ ℝ^{32}

V̂_θ(O) = W^{(out2)} · ReLU(W^{(out1)} h_G + b^{(out1)}) + b^{(out2)}
```
Parameters: **6,465**. Same MSE loss and train/test split as MLP.
Final train loss: 1.53×10⁻⁵ (~26× lower than MLP).

### 8.4 Results

```
Family                              Split   r2(ADP)  r2(MLP)  r2(GNN)  GNN vs ADP
──────────────────────────────────────────────────────────────────────────────────
Extended_LowHigh_Base               TRAIN   0.0867   0.0573   0.0300   GNN  2.9×
Extended_LowHigh_Aggressive         TRAIN   0.0501   0.0241   0.0367   GNN  1.4×
Extended_MediumEven_Base            TRAIN   0.1069   0.0227   0.0270   GNN  4.0×
Extended_MediumEven_Aggressive      TRAIN   0.0163   0.0334   0.0365   ADP  2.2×
ThreeGeneration_LowHigh_Base        TRAIN   0.1090   0.0262   0.0014   GNN 76.5×
ThreeGeneration_LowHigh_Aggressive  TRAIN   0.0430   0.0259   0.0111   GNN  3.9×
──────────────────────────────────────────────────────────────────────────────────
ThreeGeneration_MediumEven_Base     TEST    0.0334   0.0357   0.0415   ADP  1.2×
ThreeGeneration_MediumEven_Aggressive TEST  0.0143   0.0223   0.0236   ADP  1.6×
```
GNN wins on 5/6 training families. **ADP wins on both held-out test families.**

---

## SLIDE 9 — Limitations

**1. Curse of dimensionality on training data**
MLP/GNN require exact V*(s) labels for all reachable states.
States grow exponentially: N=5 → 20,816; N=6 → 107,728; N=10 → ~10⁹ (intractable).
ADP requires no V* labels and re-solves per family via LP — scales freely.

**2. Belief update is exact, not approximated**
The belief update (Bayesian conditioning via message passing on the pedigree BN)
is computed to computational precision using pgmpy VariableElimination.
MLP/GNN approximate V*(s), not the belief update. Both methods receive identical,
exact beliefs as input. The approximation is in the value function only.

**3. Test families do not generalise out-of-distribution**
ADP wins on both held-out test families (ThreeGeneration×MediumEven). The neural
methods have an information advantage on training families (access to V* labels)
that disappears on unseen families. ADP is structurally correct and does not rely
on training data from any specific family.

**4. Single seed, no confidence intervals**
Results are one training run. DP evaluation is deterministic; only Adam mini-batch
ordering varies across seeds. Multiple seeds are the natural next step.

---

## SLIDE 10 — Summary

| Property                     | ADP (ABCD-16)              | MLP                       | GNN                        |
|------------------------------|----------------------------|---------------------------|----------------------------|
| Value approx                 | Linear: θᵀφ(s)             | Nonlinear MLP             | Nonlinear GNN              |
| Features                     | 16 hand-crafted            | Raw beliefs (flat)        | Raw beliefs + pedigree     |
| Training data needed         | No                         | Yes — exact V*(s) labels  | Yes — exact V*(s) labels   |
| Re-solve per family          | Yes (~5 min, Gurobi)       | No (one forward pass)     | No (one forward pass)      |
| Scales to N=10+              | Yes                        | No (intractable labels)   | No (intractable labels)    |
| Test family performance      | **Wins both**              | Loses both                | Loses both                 |
| Parameters                   | 16 (θ)                     | 15,105                    | 6,465                      |

**Key result**: On training families, MLP and GNN beat ADP (access to exact V* labels
is a strong advantage). On held-out test families, ADP wins — neural methods trained
on small families do not generalise out-of-distribution. ADP is the correct structural
baseline: no training required, exact beliefs, re-solves per family via LP.

---

## APPENDIX A — Notation

| Symbol       | Definition |
|--------------|-----------|
| N            | Number of individuals in pedigree |
| g            | Genotype ∈ {0=NN, 1=ND, 2=DD} |
| p            | Population allele frequency |
| O            | Observation set (current test results) |
| b_O          | Belief state — posterior P(X\|O) |
| p_i^k        | Carrier probability: P(X_i^k ∈ {1,2} \| O) |
| V*(O)        | Exact optimal value at state O |
| V_stop(O)    | Value of stopping at state O |
| L            | True expected value of net's greedy policy |
| ratio2       | (V*(O₀) − L) / (V*(O₀) − V_stop(O₀)) |
| a_k, b_k, δ_k | Per-gene reward coefficients |
| c_fixed, c_var | Testing costs |
| φ(s)         | ABCD-16 16-dim feature vector |
| θ            | ADP weight vector (per-stage) |
| V̂_θ(s)     | Approximate value function output |

---

## APPENDIX B — Pedigree Structures

**ThreeGeneration (N=5)**:
```
[Grandfather] ─── [Grandmother]
                       │
              [Father] ─── [Mother]
                    │
                [Child]   [Sibling]
```

**Extended (N=6)**:
```
[Grandfather] ──── [Grandmother]
      │                   │
  [Father]             [Uncle]
  [Father] ──── [Mother]
                    │
                 [Child]
```
Edges (parent→child): Grandfather→Father, Grandmother→Father, Grandfather→Uncle,
Grandmother→Uncle, Father→Child, Mother→Child (6 directed edges).
