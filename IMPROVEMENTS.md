# Planned Improvements

## 1. Class Imbalance in Rating Distribution

### Problem
The training data has a heavily skewed rating distribution:
- There is a sharp spike at rating **2.8** (~750 samples)
- Ratings between **3.5–3.8** form a secondary cluster
- Very few samples exist below **2.0** or above **4.5**
- This causes the model to predict only in the **3.0–3.45 range**, ignoring extremes

### Root Cause
The dataset itself is imbalanced — most restaurants cluster around average ratings,
so the model learns to predict "safe" middle values.

### Planned Fixes
- [ ] **Resampling**: Oversample underrepresented rating ranges (< 2.5 and > 4.3)
      using SMOTE or manual duplication
- [ ] **Undersampling**: Reduce dominance of the 2.8 spike
- [ ] **Custom loss function**: Use weighted MSE to penalize errors on rare rating ranges more
- [ ] **Binning + classification**: Convert to a classification problem (bins: Low/Mid/High)
      to better handle imbalance
- [ ] **Data augmentation**: Collect or synthesize more samples at the extremes
- [ ] **Stratified train/test split**: Ensure rare ratings appear in both splits

### Expected Outcome
Model should predict across the full 0–5 range rather than collapsing to 3.0–3.45.