# 🛒 Market Basket Analysis & Reorder Prediction

This project tackles a key business problem in e-commerce: **Which products are users likely to reorder in their next purchase?**  
Using the Instacart online grocery dataset, we build a machine learning pipeline to model customer behavior and accurately predict product reorders. This enables **personalized product recommendations**, **inventory optimization**, and **customer retention strategies** for online retailers.

---

## 🎯 Business Objective

Reordering behavior is a strong indicator of customer satisfaction and loyalty. By identifying which products a user is most likely to reorder, Instacart and similar platforms can:

- 📈 Improve **customer experience** through personalized reorder suggestions.
- 💰 Boost **basket size and revenue** by surfacing high-likelihood items.
- 📦 Optimize **inventory planning** for frequently reordered products.
- 📬 Power **targeted marketing** (e.g., reminder emails, loyalty programs).

---

## ⚙️ Machine Learning Pipeline

### 📁 Step-by-Step Workflow

**Step 1: Data Description**
- Processed six core datasets: `orders`, `order_products__prior/train`, `products`, `aisles`, `departments`.
- Merged them into a unified customer–product interaction dataset (~3.2M rows).

**Step 2: Exploratory Data Analysis**
- Discovered peak order times (Wed/Thurs, 10 AM–2 PM), high reorder rates (~58%), and staple-heavy baskets.
- Analyzed aisle trends and reorder latency (7–14 days most common).

**Step 3: Customer Segmentation**
- Applied K-Means clustering on behavioral features.
- Identified 5 distinct customer personas (e.g., “Seltzer Loyalists”, “Heavy Produce Buyers”, “Infrequent Users”).

**Step 4: Market Basket Analysis (Association Rules)**
- Transformed prior order data into a transaction matrix for frequent itemset mining.
- Applied the **Apriori algorithm** to discover commonly co-purchased products.
- Generated association rules with thresholds on **support**, **confidence**, and **lift**.
- Identified strong product affinities like:
  - 🧀 *“Cheddar Cheese” → “Whole Milk”* (Lift > 3.0)
  - 🍞 *“Bread” + “Eggs” → “Butter”* (Confidence > 0.6)

> These insights can power **cross-selling strategies**, **product bundling**, and **UI enhancements** ("Customers who bought this also bought...").

**Step 5: Feature Engineering**

To capture user loyalty, product stickiness, and recency-driven behaviors, we engineered custom features from prior orders and integrated them at the user-product level.

🔧 **Key engineered features:**

| Feature Name | Description |
|--------------|-------------|
| `total_product_orders_by_user` | Number of times the user has purchased this product |
| `total_product_reorders_by_user` | Number of times the product was reordered by the user |
| `user_product_reorder_percentage` | Ratio of reorders to total orders for that product-user pair |
| `avg_add_to_cart_by_user` | Average position in the cart when this product was added |
| `avg_days_since_last_bought` | Average days since prior order when this product was purchased |
| `last_ordered_in` | Most recent order number when the user bought the product |
| `is_reorder_1`, `is_reorder_2`, `is_reorder_3` | Flags indicating whether the product was reordered in the user's last 3 orders |

🧠 These features encode:
- **User-product affinity** (via reorder percentage and total reorders),
- **Recency and frequency signals** (via `last_ordered_in`, `avg_days_since_last_bought`),
- **Positional relevance** in shopping carts (products added early are often habitual items).

> This behavioral feature matrix became the foundation for training both the ANN and XGBoost models.


**Step 6: Data Preprocessing**
- Engineered a supervised learning problem using prior orders as features and train orders as labels.
- Cleaned nulls, encoded categories, and split train/test sets based on last order.


**Step 7: ANN Modeling**
- Developed a multi-layer perceptron (MLP) using TensorFlow/Keras with:
  - Input layer reflecting engineered features
  - Two hidden layers with ReLU activation and dropout for regularization
  - Binary cross-entropy loss and Adam optimizer
- Employed **early stopping** on validation AUC to prevent overfitting
- Final performance on the test set:
  - 🎯 **Accuracy:** 82.1%
  - 📈 **F1 Score:** 0.74
  - 🧠 **ROC-AUC:** 0.79
  - 📉 **Precision:** 0.72 | **Recall:** 0.77


**Step 8: XGBoost Tuning**
- Implemented a gradient boosting classifier using the XGBoost framework
- Performed hyperparameter tuning with cross-validation:
  - Parameters tuned: `max_depth`, `eta` (learning rate), `subsample`, `colsample_bytree`
  - Handled class imbalance via `scale_pos_weight` adjustment
- Final performance on the test set:
  - 🎯 **Accuracy:** 84.0%
  - 📈 **F1 Score:** 0.79
  - 🧠 **ROC-AUC:** 0.85
  - 📉 **Precision:** 0.76 | **Recall:** 0.82

**Step 9: Model Comparison**

To evaluate model effectiveness, both the Artificial Neural Network (ANN) and XGBoost were assessed using identical train/test splits and the same engineered feature set. Metrics compared include ROC-AUC, F1 Score, Precision, and Recall.

| Model     | ROC-AUC | F1 Score | Precision | Recall | Key Observations |
|-----------|---------|----------|-----------|--------|------------------|
| **ANN**   | 0.79    | 0.74     | 0.72      | 0.77   | Captures non-linear patterns, but training is slower and interpretability is limited |
| **XGBoost** | **0.85** | **0.79** | **0.76**  | **0.82** | Outperforms ANN across all metrics; faster training and more interpretable via feature importance |

🧠 **Why XGBoost Wins:**
- Higher predictive performance on imbalanced classes
- Inherently handles missing values and requires less feature scaling
- Outputs feature importance directly — making it easier to explain predictions

🔍 **Top Predictive Features from XGBoost:**
1. `user_product_reorder_percentage`
2. `total_product_reorders_by_user`
3. `last_ordered_in`
4. `avg_days_since_last_bought`
5. `avg_add_to_cart_by_user`

> These features highlight user-product affinity, recency, and behavioral consistency as the strongest drivers of reorder likelihood.


### 🧠 When to Choose ANN vs. XGBoost

| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| 🔍 You need **interpretability** (e.g., feature importance, lift) | **XGBoost** | Tree-based models provide clear insights into feature impact |
| ⚡ You want **fast inference in production** | **XGBoost** | Much faster than neural nets for tabular data |
| 📊 You have a **structured, tabular dataset** with engineered features | **XGBoost** | Gradient boosting is highly optimized for this setting |
| 🧬 You expect **non-linear or high-dimensional interactions** without much feature engineering | **ANN** | Neural nets may discover complex patterns if tuned well |
| 🧪 You’re **experimenting or benchmarking** across modeling approaches | Both | Use ANN for model diversity; ensemble for better generalization |

> 🔧 **Final Choice in This Project**:  
> Based on superior **F1 Score**, **ROC-AUC**, faster training, and better interpretability, **XGBoost** was chosen as the final model for deployment.


---

## 💡 Key Insights

- ⏳ **Recency matters**: Reorders spike 7–14 days after prior order.
- 🥦 **Top aisles**: Fresh produce, dairy, beverages dominate reorder traffic.
- 🔁 **Reorder rate** is the strongest signal for future purchase likelihood.
- 👤 **Cluster analysis** reveals niche loyalty (e.g., sparkling water superfans).

---

## 🔧 Tools & Libraries

- **Python**: pandas, NumPy, scikit-learn
- **Deep Learning**: TensorFlow / Keras
- **Gradient Boosting**: XGBoost
- **Visualization**: Seaborn, Matplotlib
- **Clustering**: KMeans (scikit-learn)

---

## 📁 Dataset Source

This project uses the [Instacart Market Basket Dataset (2017)](https://www.instacart.com/datasets/grocery-shopping-2017), which includes anonymized orders from 200k+ users and 3M+ prior order records.

---


## 🧠 Author

**Ishan Kapadia**  
MS in Data Science 


---



