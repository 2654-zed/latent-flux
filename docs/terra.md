Here’s your report cleanly formatted as a **Markdown (.md) file**, ready to drop into GitHub / Notion / docs.

---

```markdown
# Terra/UST Collapse — Adversarial System Stress Report
### A Quantitative Analysis of Algorithmic Peg Failure Under Coordinated Liquidity Shock

---

## 1. Executive Summary

- The collapse of UST was **not a smart contract exploit**, but a **predictable failure of an under-collateralized reflexive system under adversarial liquidity pressure**.
- The system’s **defensive capacity (liquidity + reserves)** was materially lower than the **capital required to destabilize it**.
- A **liquidity shock (~$300–$400M)** applied at a structurally vulnerable moment (Curve pool imbalance + reduced liquidity depth) triggered:
  - Peg deviation  
  - Reflexive mint/burn expansion  
  - Confidence collapse  
  - Terminal death spiral  

**Conclusion:**  
UST functioned as a **soft peg without sufficient depth**, making it vulnerable to a **Soros-style attack translated into DeFi mechanics**.

---

## 2. System Architecture (Failure Surface)

### 2.1 Core Components

- **UST (stablecoin)** — soft peg to $1  
- **LUNA (absorber asset)** — volatility sink via mint/burn  
- **Curve pools** — primary liquidity venue  
- **Luna Foundation Guard (LFG)** — BTC reserves as defense  

---

### 2.2 Stability Mechanism

UST peg relied on:

- Arbitrage:
  - Burn 1 UST → mint $1 of LUNA  
  - Mint 1 UST → burn $1 of LUNA  

This assumes:
- **LUNA retains market value**
- **Liquidity exists to absorb flows**

---

### 2.3 Critical Assumption (Hidden Weakness)

> The system assumes **redemption demand < LUNA market absorption capacity**

This is the exact point of failure.

---

## 3. Liquidity Physics Model

### 3.1 Simplified Model

Let:

- `L` = available liquidity depth  
- `S` = size of sell pressure  
- `P` = price impact  

Then:

```

P ∝ S / L

```

### Interpretation:

- Small `S` → negligible impact  
- Large `S` relative to `L` → nonlinear collapse  

---

### 3.2 Curve Pool Fragility

At time of attack:

- Liquidity had been **partially withdrawn/rebalanced**
- Pool imbalance increased sensitivity

Result:
- Effective `L` ↓  
- Required `S` to break peg ↓  

---

## 4. Attack Sequence (Time-Structured)

### Phase 1 — Positioning
- Accumulate UST via OTC (low visibility)
- Establish short exposure:
  - BTC (LFG reserve asset)
  - LUNA (reflexive failure target)

---

### Phase 2 — Liquidity Shock
- Deploy ~$300–350M into Curve pool
- Force imbalance → price deviation

---

### Phase 3 — Peg Break

- UST trades < $1
- Arbitrage loop activates:
  - UST redeemed → LUNA minted

---

### Phase 4 — Reflexive Collapse Loop

```

UST ↓ → LUNA Supply ↑ → LUNA Price ↓ → Confidence ↓ → UST Redemptions ↑

```

---

### Phase 5 — Reserve Failure

- LFG deploys BTC reserves
- BTC market impact:
  - Downward pressure
  - Further profit for attackers

---

### Phase 6 — Terminal State

- Hyperinflation of LUNA
- UST loses peg permanently
- System enters irreversible state

---

## 5. Reflexivity Model (Core Insight)

UST wasn’t just under-collateralized—it was:

> **Reflexively dependent on market confidence + liquidity depth**

Let:

- `C` = confidence  
- `Lp` = LUNA price  
- `R` = redemption rate  

```

C ↓ → R ↑ → Lp ↓ → C ↓

```

This is a **positive feedback loop → unstable equilibrium**.

---

## 6. Comparison to Historical Precedent

### Black Wednesday (1992 UK Currency Crisis)

| Dimension | UK Pound (1992) | Terra (2022) |
|------|------|------|
| Peg type | Fixed FX band | Algorithmic peg |
| Defender | Central bank | LFG reserves |
| Weakness | Insufficient reserves | Reflexive supply expansion |
| Attack vector | Short selling | Liquidity + redemption loop |
| Outcome | Forced devaluation | Total collapse |

---

## 7. Capital Efficiency of Attack

Key insight:

> You don’t need to match system size—you only need to exceed **defensive liquidity at the weakest point**

### Attack condition:

```

S > L_effective

```

Where:

- `L_effective` = actual usable liquidity during stress (not headline TVL)

---

## 8. Why This Was Inevitable

From a systems perspective:

- No hard collateral floor  
- Reflexive supply expansion  
- Public, predictable defense mechanism  
- Finite liquidity pools  
- Transparent attack surface  

---

## 9. Strategic Takeaways (Institutional Level)

### 9.1 For Protocol Designers
- Avoid reflexive death spirals
- Model **worst-case coordinated capital attack**
- Treat liquidity as **finite defense, not infinite abstraction**

---

### 9.2 For Traders / Funds
- Identify:
  - Peg systems with weak reserves  
  - Reflexive feedback loops  
  - Liquidity bottlenecks  

These are **attackable structures**

---

### 9.3 Analytical Framework Shift

The key question is not:

> "Is the contract safe?"

The correct question is:

> "What happens when capital applies pressure at the weakest point?"

---

## 10. Final Reframe

This was not:

- A hack  
- A bug  
- A failure of code  

It was:

> **A system behaving exactly as designed under adversarial conditions**

---
