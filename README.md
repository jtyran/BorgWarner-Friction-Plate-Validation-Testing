🚗 BorgWarner Friction Plate Validation Testing

Machine Learning for Manufacturing Quality Prediction

📌 Project Overview
BorgWarner is transitioning friction plate production from its Heidelberg, Germany facility to a new plant in Rzeszów, Poland. A major risk in this transition is the pre-production performance verification test, where approximately 25% of batches historically fail, triggering costly rework, production delays, and inventory buildup.

This project applies machine learning classification models to historical manufacturing data to determine whether process parameters can reliably predict pass/fail outcomes before full production begins. The goal is to replace intuition-driven parameter selection with a repeatable, data-driven decision tool that improves first-pass yield during production ramp-up. 

🎯 Problem Statement
- Pre-production verification failures cause delays, rework, and added cost
- Parameter adjustments were historically based on engineering intuition
- The manufacturing process exhibits non-linear behavior

Objective: Predict Pass/Fail outcomes using historical process data to reduce trial-and-error

📊 Data Overview
9 manufacturing process parameters describing material and bonding conditions

Binary target variable: Pass vs. Fail

Data split into training and validation sets to ensure fair model evaluation

Dataset provided as part of the BorgWarner case study 

🧠 Modeling Approach
Why Machine Learning?
Traditional techniques (correlation analysis, linear regression, and DOE) failed to uncover meaningful relationships due to:
Machine learning models were selected because they can capture hidden, non-linear relationships that traditional statistical methods cannot. 

Models Evaluated
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Boosted Tree
- Neural Network
- K-Means Clustering (exploratory)

🌳 Decision Tree Insights
The decision tree revealed clear hierarchical relationships among variables:
var7 emerged as the dominant driver
When var7 ≥ 3, batches almost always failed
When var7 < 3, outcomes depended heavily on var1
This structure provides interpretable decision rules engineers can follow
This confirms that interactions between variables, rather than single predictors, determine test outcomes.

🌲 Random Forest Feature Importance
Random forest results reinforced the decision tree findings:
var7 ranked as the most important predictor by both:
Mean Decrease in Accuracy
Mean Decrease in Gini
var1 and var3 were also influential
Lower-ranked variables had minimal standalone predictive value
These results validate that ensemble methods consistently identify the same critical drivers, increasing confidence in model reliability.

📈 Model Performance Comparison
Model	Accuracy
Logistic Regression	0.54
Decision Tree	0.97
Random Forest	0.81
Boosted Tree	0.89
Neural Network	0.70

Logistic regression performed poorly, confirming non-linearity
Decision trees provided interpretability
Boosted trees offered the best balance of accuracy and generalization
Clustering failed to separate pass/fail outcomes, indicating unsupervised learning is unsuitable

🔑 Key Findings
- Manufacturing outcomes are driven by non-linear interactions
- No single variable predicts outcomes in isolation
- Tree-based ensemble models outperform linear approaches
- Boosted trees provide the most reliable predictions for operational use

✅ Recommendation
A boosted tree classification model is recommended as a decision-support tool for BorgWarner’s pre-production process. Deploying this model allows engineers to:

- Predict pass/fail outcomes before testing
- Reduce rework and inventory buildup
- Improve first-pass yield
- Minimize reliance on heuristic decision-making during plant transition

This approach supports a smoother, more scalable production ramp-up at the Rzeszów facility. 

🛠️ Tools & Technologies
R
caret, randomForest, gbm, rpart, factoextra
Classification modeling

Feature importance analysis
Confusion matrices & ROC-based evaluation
