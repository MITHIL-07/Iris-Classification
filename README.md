# Iris Flower Classification 🌸

A multi-class classification project that predicts the species of an iris flower (setosa, versicolor, or virginica) from its sepal and petal measurements. I compared four classifiers, evaluated them with cross-validation, and analysed which features matter most.

# Dataset
Source: the classic Iris dataset (Fisher, 1936), loaded from scikit-learn
Size: 150 flowers, 50 per species (perfectly balanced)
Features (cm): sepal_length, sepal_width, petal_length, petal_width
Target: species (3 classes)
Data quality: no missing values, so the focus is on modelling and evaluation rather than cleaning
Approach
Exploratory Data Analysis: pairplot, boxplots and a correlation heatmap to see which features separate the species
Train/test split: 80/20, stratified so each species keeps the same proportion in both sets
Model comparison: Logistic Regression, K-Nearest Neighbors (k=5), Decision Tree, Random Forest (100 trees)
Robust evaluation: 5-fold cross-validation instead of relying on a single 30-flower test set
Error analysis: confusion matrix and classification report (precision, recall, F1)
Interpretation: Random Forest feature importance
Hyperparameter tuning: choosing K for KNN using cross-validated accuracy
Results
Model	Test Accuracy	5-Fold CV Accuracy
Logistic Regression	96.67%	97.33% (± 2.49)
KNN (k=5)	100.00%	97.33% (± 2.49)
Decision Tree	93.33%	95.33% (± 3.40)
Random Forest	90.00%	96.67% (± 2.11)

Best model: Logistic Regression and KNN tie at 97.33% cross-validated accuracy. Random Forest is within noise of them (96.67%) and is used for the feature-importance analysis.

Key Findings
Setosa is perfectly separable. It is never confused with the other two species.
Versicolor and virginica overlap slightly. Almost all misclassifications happen between these two.
Petal measurements are the strongest predictors. Petal width and petal length make up roughly 87% of the Random Forest's feature importance, while sepal width contributes almost nothing. This matched what the EDA suggested.
Cross-validation matters on small data. With only 30 test flowers, one wrong prediction changes accuracy by about 3.3%. Random Forest scored 90% on the single test split but 96.67% under 5-fold cross-validation, and KNN's 100% on the test split dropped to 97.33%, so single-split rankings are mostly noise.
K in KNN barely matters once it is reasonable. Cross-validated accuracy is lowest at K=1 to 2 (about 95 to 96%), peaks around 98% at K=6 to 7 and K=10 to 12, and slowly declines for K above 13. The gaps are only one or two flowers.
How to Run
bash
git clone https://github.com/MITHIL-07/iris-classification.git
cd iris-classification
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook iris-classification.ipynb

The dataset loads directly from scikit-learn, so no file download is needed.

Project Structure
iris-classification/
├── iris-classification.ipynb   # full analysis: EDA, models, evaluation
└── README.md


# Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter

# What I Learned
-> How to compare multiple classifiers fairly using cross-validation
-> Reading a confusion matrix, and the difference between precision and recall
-> Why stratified splitting matters on small datasets
-> How the choice of K in KNN trades off overfitting against underfitting
-> Using EDA to form a hypothesis, then confirming it with feature importance
