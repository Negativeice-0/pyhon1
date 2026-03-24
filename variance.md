# Evaluation Metrics for Your Linear Regression Model

Based on your Boston Housing model, here are the key evaluation metrics and questions worth exploring:

---

## 📊 Standard Regression Metrics

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|

| **MAE** | Mean Absolute Error | Average dollar error per prediction |
| **MSE** | Mean Squared Error | Penalizes large errors more heavily |
| **RMSE** | √MSE | Same unit as MEDV (easier to interpret) |
| **R² Score** | 1 - SS_res/SS_tot | % of variance explained (0–1) |
| **Adjusted R²** | Penalizes extra features | Better for multi-feature models |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X = df.drop(columns=['MEDV'])
y = df['MEDV']
y_pred = model.predict(X)

mae  = mean_absolute_error(y, y_pred)
mse  = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred)

# Adjusted R²
n, p = X.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE:         {mae:.2f}")
print(f"RMSE:        {rmse:.2f}")
print(f"R²:          {r2:.4f}")
print(f"Adjusted R²: {adj_r2:.4f}")
```

> 📌 For the Boston dataset, a well-fitted OLS model typically achieves **R² ≈ 0.74** on training data — meaning ~74% of price variance is explained.

---

### ⚠️ Critical Questions to Consider

**1. Are you evaluating on training data only?**
Your current setup trains and predicts on the *same* 506 rows — this **inflates** your metrics. Use a train/test split:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**2. Is the $300k–$500k prediction realistic?**
The Boston dataset's `MEDV` is in **1970s dollars** (max = $50k). Your model predicts in those units — so `$12.9k` is actually reasonable, not `$300k`. The scaling to modern prices requires a separate adjustment.

**3. Are OLS assumptions met?**

- **Linearity** — is the relationship truly linear?
- **Homoscedasticity** — constant variance in residuals?
- **No multicollinearity** — check with VIF scores

**4. Which features matter most?**

```python
import pandas as pd
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coef_df.sort_values('Coefficient', ascending=False))
```

**5. Cross-validation score** (more reliable than a single split):

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f}")
```

---

### 🔑 Bottom Line

The most important next step is **splitting your data** before evaluating — otherwise your metrics are overly optimistic. Want me to walk through a full evaluation pipeline with residual plots?
