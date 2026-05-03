## Full System Walkthrough

### Step 1 — Tokenization
Take a rolling window of price data and convert it to words:

```
Raw prices: 100, 101, 100.5, 103, 102, 98...
Daily moves: +1%, -0.5%, +2.5%, -1%, -4%
Tokens:       UP_WEAK DOWN_WEAK UP_STRONG DOWN_WEAK DOWN_STRONG
```

---

### Step 2 — Sliding Window Bag of Words

Take a 20-day window, just count word frequencies — order doesn't matter:

```
Window 1 (days 1-20):
  UP_STRONG:   4
  UP_WEAK:     6
  DOWN_WEAK:   7
  DOWN_STRONG: 3
  
→ frequency vector [4, 6, 7, 3]
```

Slide the window forward one day, recount. This gives you a **continuous stream of frequency vectors** — which are far more stationary than raw prices.

---

### Step 3 — Feed Into Random Gel Configurations

Each frequency vector gets fed into multiple randomly strained gel configurations simultaneously:

```
Frequency vector [4, 6, 7, 3]
        ↓
Config A (strain pattern 1) → nonlinear feature set A
Config B (strain pattern 2) → nonlinear feature set B
Config C (strain pattern 3) → nonlinear feature set C
```

Each random configuration "sees" the same input differently — like shining light through different lenses.

---

### Step 4 — Readout Layer

A simple linear regression is trained on top of each gel's electrode outputs:

```
Feature set A → weight vector → prediction A (e.g. 60% chance UP_STRONG tomorrow)
Feature set B → weight vector → prediction B (e.g. 45% chance UP_STRONG tomorrow)
Feature set C → weight vector → prediction C (e.g. 55% chance UP_STRONG tomorrow)
```

You're predicting **next window's word distribution**, not a price.

---

### Step 5 — Ensemble Vote

```
Average prediction: 53% UP_STRONG
                    → cautious long signal
```

Confidence is low → small position or no trade. Confidence is high → larger position.

---

### What You're Actually Predicting

Not "will price go up" but rather **"which market regime comes next"** — a much more tractable question because regimes persist for days to weeks, giving the model something real to latch onto.

---

### The Training Loop

```
Historical data
      ↓
Generate bag-of-words vectors
      ↓
Feed through gel (fixed random configs)
      ↓
Train readout weights via linear regression
      ↓
Walk-forward validation (never train on future data)
      ↓
Deploy on live stream
```

The gel weights never change — only the thin linear readout layer is trained. This is fast, cheap, and resistant to overfitting.

---

### Why This Particular Combination Is Interesting

- **Bag of words** handles stationarity
- **Random gel ensemble** handles nonlinear feature diversity  
- **Linear readout** keeps the trained component simple and interpretable
- **Word prediction** is a cleaner target than price prediction
- The whole inference step runs in **microseconds** once trained

The closest software analogy is a **Random Kitchen Sink** method combined with a Hidden Markov Model — but implemented in analog physics rather than matrix multiplications.
