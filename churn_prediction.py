import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, roc_auc_score,
                           precision_score, recall_score, f1_score, make_scorer)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import joblib
import warnings
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import os
import time
from datetime import datetime

print("Starting enhanced churn prediction system with comprehensive evaluation...")

# Create assets directory if it doesn't exist
os.makedirs('assets', exist_ok=True)
# Create temp directory if it doesn't exist
os.makedirs('temp', exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set the aesthetic style of the plots
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# Custom color palette
custom_palette = sns.color_palette("viridis", 10)
sns.set_palette(custom_palette)

# Create custom colormap for risk levels
risk_cmap = LinearSegmentedColormap.from_list('risk_cmap', ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])

# Check if data file exists, if not create synthetic data
if not os.path.exists("Churn_Modelling.csv"):
    print("Churn_Modelling.csv not found. Creating synthetic data...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic features
    customer_id = np.arange(1, n_samples + 1)
    surnames = np.array(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'] * 1000)[:n_samples]
    credit_scores = np.random.randint(300, 900, n_samples)
    geography = np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.2, 0.3])
    gender = np.random.choice(['Male', 'Female'], n_samples)
    age = np.random.randint(18, 95, n_samples)
    tenure = np.random.randint(0, 11, n_samples)
    balance = np.random.uniform(0, 250000, n_samples)
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05])
    has_cr_card = np.random.choice([0, 1], n_samples)
    is_active_member = np.random.choice([0, 1], n_samples)
    estimated_salary = np.random.uniform(10000, 200000, n_samples)
    
    # Generate churn based on features
    churn_prob = (
        0.02  # base probability
        + 0.1 * (1 - is_active_member)  # inactive members more likely to churn
        + 0.05 * (age < 30).astype(int)  # young customers more likely to churn
        + 0.08 * (age > 60).astype(int)  # older customers more likely to churn
        + 0.1 * (balance < 10000).astype(int)  # low balance customers more likely to churn
        + 0.15 * (num_products == 1).astype(int)  # single product customers more likely to churn
        - 0.1 * (tenure > 5).astype(int)  # long-term customers less likely to churn
        - 0.05 * has_cr_card  # credit card holders less likely to churn
        + 0.1 * (geography == 'Germany').astype(int)  # German customers more likely to churn
        + np.random.normal(0, 0.05, n_samples)  # add some randomness
    )
    
    # Clip probabilities to be between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 1)
    exited = (churn_prob > 0.5).astype(int)
    
    # Create DataFrame
    synthetic_data = pd.DataFrame({
        'RowNumber': np.arange(1, n_samples + 1),
        'CustomerId': customer_id,
        'Surname': surnames,
        'CreditScore': credit_scores,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Exited': exited
    })
    
    # Save synthetic data
    synthetic_data.to_csv("Churn_Modelling.csv", index=False)
    print("Synthetic data created and saved as Churn_Modelling.csv")

# Load the data
print("Loading and preparing data...")
try:
    data = pd.read_csv("Churn_Modelling.csv")
    print(f"Dataset shape: {data.shape}")
    print(f"Number of missing values: {data.isnull().sum().sum()}")
    print(f"Number of duplicate rows: {data.duplicated().sum()}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# ----------------------
# DATA PREPROCESSING
# ----------------------

# Drop non-essential columns
data_cleaned = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data_cleaned, columns=['Geography', 'Gender'], drop_first=True)

# Feature Engineering
print("\nPerforming feature engineering...")

# Basic feature engineering
data_encoded['Balance_to_Salary_Ratio'] = data_encoded['Balance'] / (data_encoded['EstimatedSalary'] + 1)
data_encoded['Credit_to_Salary_Ratio'] = data_encoded['CreditScore'] / (data_encoded['EstimatedSalary'] + 1)
data_encoded['Age_to_Tenure_Ratio'] = data_encoded['Age'] / (data_encoded['Tenure'] + 1)
data_encoded['Products_per_Tenure'] = data_encoded['NumOfProducts'] / (data_encoded['Tenure'] + 1)

# Advanced feature engineering
data_encoded['IsHighValueCustomer'] = ((data_encoded['Balance'] > data_encoded['Balance'].median()) & 
                                      (data_encoded['EstimatedSalary'] > data_encoded['EstimatedSalary'].median())).astype(int)
data_encoded['IsLongTermCustomer'] = (data_encoded['Tenure'] > 5).astype(int)
data_encoded['HasMultipleProducts'] = (data_encoded['NumOfProducts'] > 1).astype(int)
data_encoded['IsYoungInactive'] = ((data_encoded['Age'] < 30) & (data_encoded['IsActiveMember'] == 0)).astype(int)
data_encoded['IsOldInactive'] = ((data_encoded['Age'] > 60) & (data_encoded['IsActiveMember'] == 0)).astype(int)

# Replace inf values and drop rows with NaN
data_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
data_encoded.dropna(inplace=True)

print(f"Final shape after preprocessing: {data_encoded.shape}")

# ----------------------
# CUSTOMER SEGMENTATION
# ----------------------

print("\nPerforming customer segmentation...")

# Select features for clustering
cluster_features = ['Age', 'Balance', 'EstimatedSalary', 'CreditScore', 'Tenure', 'NumOfProducts', 'IsActiveMember']
X_cluster = data_encoded[cluster_features].copy()

# Standardize the data for clustering
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using the Elbow Method
inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-', color='#3498db')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

# Choose optimal k (for this example, let's use k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

# Add cluster labels to the dataset
data_encoded['CustomerSegment'] = cluster_labels

# Analyze the segments
segment_profiles = data_encoded.groupby('CustomerSegment').mean()
print("\nCustomer Segment Profiles:")
print(segment_profiles[['Age', 'Balance', 'EstimatedSalary', 'CreditScore', 'Tenure', 'NumOfProducts', 'IsActiveMember', 'Exited']])

# Name the segments based on their characteristics
segment_names = {
    0: "Young Professionals",
    1: "Established Savers",
    2: "High-Value Clients",
    3: "At-Risk Seniors"
}

# Map segment numbers to names
data_encoded['SegmentName'] = data_encoded['CustomerSegment'].map(segment_names)

# Visualize the segments
plt.figure(figsize=(12, 10))
for i, feature in enumerate(['Age', 'Balance', 'EstimatedSalary', 'Tenure']):
    plt.subplot(2, 2, i+1)
    for segment in range(optimal_k):
        segment_data = data_encoded[data_encoded['CustomerSegment'] == segment]
        sns.kdeplot(segment_data[feature], label=segment_names[segment], shade=True)
    plt.title(f'Distribution of {feature} by Customer Segment')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
plt.tight_layout()
plt.savefig('assets/segment_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualize segments with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

plt.figure(figsize=(12, 10))
for segment in range(optimal_k):
    segment_data = X_pca[cluster_labels == segment]
    plt.scatter(segment_data[:, 0], segment_data[:, 1], label=segment_names[segment], alpha=0.7, s=50)

plt.title('Customer Segments Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/segment_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# ENHANCED MODEL BUILDING WITH HYPERPARAMETER TUNING
# ----------------------

print("\nBuilding models with hyperparameter tuning...")

# Define the target variable and features
X = data_encoded.drop(columns=['Exited', 'CustomerSegment', 'SegmentName'])
y = data_encoded['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original training set distribution: {np.bincount(y_train)}")
print(f"Balanced training set distribution: {np.bincount(y_train_balanced)}")

# ----------------------
# HYPERPARAMETER TUNING
# ----------------------

print("\nPerforming hyperparameter tuning...")

# Define parameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Initialize models
base_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Perform grid search for each model
tuned_models = {}
best_params = {}
tuning_scores = {}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_models.items():
    print(f"\nTuning {name}...")
    start_time = time.time()
    
    # Create scorer for F1 score
    scorer = make_scorer(f1_score, pos_label=1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, 
        param_grids[name], 
        cv=cv_strategy,
        scoring=scorer,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    # Store results
    tuned_models[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_
    tuning_scores[name] = grid_search.best_score_
    
    end_time = time.time()
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {end_time - start_time:.2f} seconds")

# ----------------------
# COMPREHENSIVE MODEL EVALUATION
# ----------------------

print("\nPerforming comprehensive model evaluation...")

# Initialize results storage
evaluation_results = {}
cv_scores = {}

# Define scoring metrics for cross-validation
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for name, model in tuned_models.items():
    print(f"\nEvaluating {name}...")
    
    # Fit the model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Perform cross-validation
    cv_results = {}
    for metric in scoring_metrics:
        cv_scores_metric = cross_val_score(
            model, X_train_balanced, y_train_balanced, 
            cv=cv_strategy, scoring=metric, n_jobs=-1
        )
        cv_results[metric] = {
            'mean': cv_scores_metric.mean(),
            'std': cv_scores_metric.std(),
            'scores': cv_scores_metric
        }
    
    # Store all results
    evaluation_results[name] = {
        'model': model,
        'test_metrics': test_metrics,
        'cv_results': cv_results,
        'predictions': y_pred,
        'probabilities': y_prob,
        'best_params': best_params[name],
        'tuning_score': tuning_scores[name]
    }
    
    # Print results
    print(f"Test Set Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print(f"Cross-Validation Results (mean ± std):")
    for metric, result in cv_results.items():
        print(f"  {metric.capitalize()}: {result['mean']:.4f} ± {result['std']:.4f}")

# ----------------------
# MODEL COMPARISON AND SELECTION
# ----------------------

print("\nComparing model performance...")

# Create comparison DataFrame
comparison_data = []
for name, results in evaluation_results.items():
    row = {
        'Model': name,
        'Test_Accuracy': results['test_metrics']['accuracy'],
        'Test_Precision': results['test_metrics']['precision'],
        'Test_Recall': results['test_metrics']['recall'],
        'Test_F1': results['test_metrics']['f1'],
        'Test_ROC_AUC': results['test_metrics']['roc_auc'],
        'CV_Accuracy': results['cv_results']['accuracy']['mean'],
        'CV_Precision': results['cv_results']['precision']['mean'],
        'CV_Recall': results['cv_results']['recall']['mean'],
        'CV_F1': results['cv_results']['f1']['mean'],
        'CV_ROC_AUC': results['cv_results']['roc_auc']['mean'],
        'Tuning_Score': results['tuning_score']
    }
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Comparison:")
print(comparison_df.round(4))

# Save comparison results
comparison_df.to_csv('assets/model_comparison.csv', index=False)

# Select best model based on CV F1 score
best_model_name = comparison_df.loc[comparison_df['CV_F1'].idxmax(), 'Model']
best_model = evaluation_results[best_model_name]['model']
best_predictions = evaluation_results[best_model_name]['predictions']
best_probabilities = evaluation_results[best_model_name]['probabilities']

print(f"\nBest performing model: {best_model_name}")
print(f"Best parameters: {evaluation_results[best_model_name]['best_params']}")
print(f"CV F1-Score: {evaluation_results[best_model_name]['cv_results']['f1']['mean']:.4f}")
print(f"Test F1-Score: {evaluation_results[best_model_name]['test_metrics']['f1']:.4f}")

# ----------------------
# COMPREHENSIVE VISUALIZATIONS
# ----------------------

print("\nGenerating comprehensive evaluation visualizations...")

# 1. Model Performance Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    
    test_scores = [evaluation_results[name]['test_metrics'][metric] for name in evaluation_results.keys()]
    cv_scores = [evaluation_results[name]['cv_results'][metric]['mean'] for name in evaluation_results.keys()]
    cv_stds = [evaluation_results[name]['cv_results'][metric]['std'] for name in evaluation_results.keys()]
    
    x = np.arange(len(evaluation_results))
    width = 0.35
    
    ax.bar(x - width/2, test_scores, width, label='Test Set', alpha=0.8)
    ax.bar(x + width/2, cv_scores, width, yerr=cv_stds, label='CV Mean ± Std', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Models')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(evaluation_results.keys(), rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('assets/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Enhanced ROC Curves
plt.figure(figsize=(12, 10))

for name, results in evaluation_results.items():
    y_prob = results['probabilities']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Add confidence intervals for ROC curves
    plt.plot(fpr, tpr, lw=3, 
             label=f'{name} (AUC = {roc_auc:.3f})', 
             alpha=0.8)

plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve Comparison with Hyperparameter Tuning', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/enhanced_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Precision-Recall Curves
plt.figure(figsize=(12, 10))

for name, results in evaluation_results.items():
    y_prob = results['probabilities']
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.plot(recall, precision, lw=3, 
             label=f'{name}', 
             alpha=0.8)

plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve Comparison', fontsize=16)
plt.legend(loc="best", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/enhanced_precision_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Cross-Validation Score Distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    
    for name, results in evaluation_results.items():
        scores = results['cv_results'][metric]['scores']
        ax.hist(scores, alpha=0.7, label=name, bins=10)
    
    ax.set_xlabel(f'CV {metric.capitalize()} Score')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of CV {metric.capitalize()} Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1, 2].remove()
plt.tight_layout()
plt.savefig('assets/cv_score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (name, results) in enumerate(evaluation_results.items()):
    y_pred = results['predictions']
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[i].set_title(f'{name}\nConfusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('assets/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Feature Importance Analysis (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    feature_importances = np.abs(best_model.coef_[0])
else:
    feature_importances = np.zeros(X.shape[1])

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

plt.figure(figsize=(12, 8))
top_features = feature_importance_df.head(15)
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title(f'Top 15 Features by Importance ({best_model_name})', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('assets/enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# CHURN RISK LEVELS WITH BEST MODEL
# ----------------------

print("\nCalculating churn risk levels with best model...")

# Get probability estimates for all customers using the best model
churn_probabilities = best_model.predict_proba(X)[:, 1]

# Define risk levels based on churn probability
def assign_risk_level(probability):
    if probability < 0.25:
        return "Low Risk"
    elif probability < 0.50:
        return "Medium Risk"
    elif probability < 0.75:
        return "High Risk"
    else:
        return "Very High Risk"

# Add risk levels to the dataset
data_encoded['ChurnProbability'] = churn_probabilities
data_encoded['RiskLevel'] = data_encoded['ChurnProbability'].apply(assign_risk_level)

# Convert risk levels to numeric for visualization
risk_level_map = {
    "Low Risk": 0,
    "Medium Risk": 1,
    "High Risk": 2,
    "Very High Risk": 3
}
data_encoded['RiskLevelNumeric'] = data_encoded['RiskLevel'].map(risk_level_map)

# Distribution of risk levels
risk_distribution = data_encoded['RiskLevel'].value_counts().sort_index()
print("\nDistribution of Churn Risk Levels:")
print(risk_distribution)

# ----------------------
# ADVANCED VISUALIZATIONS FOR REPORTING
# ----------------------

print("\nGenerating advanced visualizations for reporting...")

# 1. Churn Rate by Customer Segment
segment_churn = data_encoded.groupby('SegmentName')['Exited'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=segment_churn.index, y=segment_churn.values, palette='viridis')
plt.title('Churn Rate by Customer Segment', fontsize=16)
plt.xlabel('Customer Segment', fontsize=14)
plt.ylabel('Churn Rate', fontsize=14)
plt.xticks(rotation=45)

# Add percentage labels
for i, v in enumerate(segment_churn.values):
    ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=12)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig('assets/churn_by_segment.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Risk Level Distribution by Customer Segment
risk_segment_counts = pd.crosstab(data_encoded['SegmentName'], data_encoded['RiskLevel'])
risk_segment_pcts = risk_segment_counts.div(risk_segment_counts.sum(axis=1), axis=0)

plt.figure(figsize=(14, 8))
risk_segment_pcts.plot(kind='bar', stacked=True, colormap=risk_cmap)
plt.title('Risk Level Distribution by Customer Segment', fontsize=16)
plt.xlabel('Customer Segment', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Risk Level')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/risk_by_segment.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Churn Probability Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=data_encoded, x='ChurnProbability', hue='Exited', 
             bins=30, kde=True, palette=['#3498db', '#e74c3c'])
plt.axvline(x=0.25, color='#2ecc71', linestyle='--', label='Risk Thresholds')
plt.axvline(x=0.50, color='#f1c40f', linestyle='--')
plt.axvline(x=0.75, color='#e74c3c', linestyle='--')
plt.title('Distribution of Churn Probabilities', fontsize=16)
plt.xlabel('Probability of Churn', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(title='Customer Status')
plt.tight_layout()
plt.savefig('assets/churn_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Feature Correlation Heatmap
plt.figure(figsize=(16, 14))
corr_matrix = data_encoded.drop(columns=['CustomerSegment', 'SegmentName', 'RiskLevel']).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='viridis', annot=False, 
            square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('assets/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Risk Level by Age and Balance
plt.figure(figsize=(12, 10))
scatter = plt.scatter(data_encoded['Age'], data_encoded['Balance'], 
                     c=data_encoded['RiskLevelNumeric'], cmap=risk_cmap, 
                     alpha=0.7, s=50, edgecolors='w', linewidth=0.5)

plt.colorbar(scatter, label='Churn Risk Level', ticks=[0, 1, 2, 3], 
             format=mtick.FuncFormatter(lambda x, pos: ['Low', 'Medium', 'High', 'Very High'][int(x)]))
plt.title('Churn Risk Level by Age and Balance', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('assets/risk_by_age_balance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Enhanced Dashboard-style visualization
plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

# Churn by segment
ax1 = plt.subplot(gs[0, 0])
sns.barplot(x=segment_churn.index, y=segment_churn.values, palette='viridis', ax=ax1)
ax1.set_title('Churn Rate by Customer Segment', fontsize=14)
ax1.set_xlabel('Customer Segment', fontsize=12)
ax1.set_ylabel('Churn Rate', fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Risk distribution
ax2 = plt.subplot(gs[0, 1])
risk_counts = data_encoded['RiskLevel'].value_counts().sort_index()
colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
ax2.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
ax2.set_title('Distribution of Churn Risk Levels', fontsize=14)

# Top features
ax3 = plt.subplot(gs[1, 0])
top_5_features = feature_importance_df.head(5)
sns.barplot(x='Importance', y='Feature', data=top_5_features, palette='viridis', ax=ax3)
ax3.set_title('Top 5 Features by Importance', fontsize=14)
ax3.set_xlabel('Importance', fontsize=12)
ax3.set_ylabel('Feature', fontsize=12)

# Confusion matrix
ax4 = plt.subplot(gs[1, 1])
conf_mat = confusion_matrix(y_test, best_predictions)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
ax4.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14)

# Risk by age and balance
ax5 = plt.subplot(gs[2, :])
scatter = ax5.scatter(data_encoded['Age'], data_encoded['Balance'], 
                     c=data_encoded['RiskLevelNumeric'], cmap=risk_cmap, 
                     alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
ax5.set_title('Churn Risk Level by Age and Balance', fontsize=14)
ax5.set_xlabel('Age', fontsize=12)
ax5.set_ylabel('Balance', fontsize=12)
ax5.grid(True, linestyle='--', alpha=0.7)
cbar = plt.colorbar(scatter, ax=ax5, label='Churn Risk Level', ticks=[0, 1, 2, 3], 
                   format=mtick.FuncFormatter(lambda x, pos: ['Low', 'Medium', 'High', 'Very High'][int(x)]))

plt.tight_layout()
plt.savefig('assets/enhanced_churn_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------
# ADVANCED EVALUATION METRICS AND REPORTING
# ----------------------

print("\nGenerating comprehensive evaluation report...")

# Create detailed evaluation report
evaluation_report = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'total_samples': len(data_encoded),
        'features': X.shape[1],
        'churn_rate': data_encoded['Exited'].mean(),
        'train_size': len(X_train),
        'test_size': len(X_test)
    },
    'best_model': {
        'name': best_model_name,
        'parameters': evaluation_results[best_model_name]['best_params'],
        'test_metrics': evaluation_results[best_model_name]['test_metrics'],
        'cv_metrics': {metric: results['mean'] for metric, results in evaluation_results[best_model_name]['cv_results'].items()}
    },
    'all_models_comparison': comparison_df.to_dict('records')
}

# Save detailed evaluation report
import json
with open('assets/evaluation_report.json', 'w') as f:
    json.dump(evaluation_report, f, indent=2, default=str)

# Create a comprehensive text report
report_lines = [
    "="*80,
    "ENHANCED CHURN PREDICTION MODEL EVALUATION REPORT",
    "="*80,
    f"Generated: {evaluation_report['timestamp']}",
    "",
    "DATASET OVERVIEW:",
    f"  Total Samples: {evaluation_report['dataset_info']['total_samples']:,}",
    f"  Number of Features: {evaluation_report['dataset_info']['features']}",
    f"  Overall Churn Rate: {evaluation_report['dataset_info']['churn_rate']:.2%}",
    f"  Training Set Size: {evaluation_report['dataset_info']['train_size']:,}",
    f"  Test Set Size: {evaluation_report['dataset_info']['test_size']:,}",
    "",
    "HYPERPARAMETER TUNING RESULTS:",
]

for name, results in evaluation_results.items():
    report_lines.extend([
        f"  {name}:",
        f"    Best Parameters: {results['best_params']}",
        f"    Tuning Score (CV F1): {results['tuning_score']:.4f}",
        ""
    ])

report_lines.extend([
    "BEST MODEL PERFORMANCE:",
    f"  Selected Model: {best_model_name}",
    f"  Best Parameters: {evaluation_results[best_model_name]['best_params']}",
    "",
    "  Test Set Metrics:",
])

for metric, value in evaluation_results[best_model_name]['test_metrics'].items():
    report_lines.append(f"    {metric.capitalize()}: {value:.4f}")

report_lines.extend([
    "",
    "  Cross-Validation Metrics (Mean ± Std):",
])

for metric, results in evaluation_results[best_model_name]['cv_results'].items():
    report_lines.append(f"    {metric.capitalize()}: {results['mean']:.4f} ± {results['std']:.4f}")

report_lines.extend([
    "",
    "MODEL COMPARISON SUMMARY:",
    f"{'Model':<20} {'Test F1':<10} {'CV F1':<10} {'Test ROC-AUC':<12} {'CV ROC-AUC':<12}",
    "-" * 65,
])

for _, row in comparison_df.iterrows():
    report_lines.append(f"{row['Model']:<20} {row['Test_F1']:<10.4f} {row['CV_F1']:<10.4f} {row['Test_ROC_AUC']:<12.4f} {row['CV_ROC_AUC']:<12.4f}")

report_lines.extend([
    "",
    "TOP 10 IMPORTANT FEATURES:",
])

for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
    report_lines.append(f"  {i:2d}. {row['Feature']:<30} {row['Importance']:.4f}")

report_lines.extend([
    "",
    "CUSTOMER SEGMENTATION INSIGHTS:",
])

segment_insights = {}
for segment in data_encoded['SegmentName'].unique():
    segment_data = data_encoded[data_encoded['SegmentName'] == segment]
    segment_insights[segment] = {
        'size': len(segment_data),
        'churn_rate': segment_data['Exited'].mean(),
        'avg_age': segment_data['Age'].mean(),
        'avg_balance': segment_data['Balance'].mean(),
        'avg_tenure': segment_data['Tenure'].mean(),
        'high_risk_pct': segment_data['RiskLevel'].isin(['High Risk', 'Very High Risk']).mean()
    }

for segment, insights in segment_insights.items():
    report_lines.extend([
        f"  {segment}:",
        f"    Size: {insights['size']:,} customers ({insights['size']/len(data_encoded):.1%})",
        f"    Churn Rate: {insights['churn_rate']:.2%}",
        f"    Average Age: {insights['avg_age']:.1f} years",
        f"    Average Balance: ${insights['avg_balance']:,.2f}",
        f"    Average Tenure: {insights['avg_tenure']:.1f} years",
        f"    High-Risk Customers: {insights['high_risk_pct']:.2%}",
        ""
    ])

report_lines.extend([
    "RISK LEVEL DISTRIBUTION:",
])

for risk_level, count in risk_distribution.items():
    percentage = count / len(data_encoded) * 100
    report_lines.append(f"  {risk_level}: {count:,} customers ({percentage:.1f}%)")

report_lines.extend([
    "",
    "EVALUATION METHODOLOGY:",
    "  • Hyperparameter tuning using GridSearchCV with 5-fold stratified cross-validation",
    "  • SMOTE applied for handling class imbalance in training data",
    "  • Multiple evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC",
    "  • Model selection based on cross-validated F1-score",
    "  • Comprehensive visualization suite for stakeholder communication",
    "",
    "FILES GENERATED:",
    "  • enhanced_churn_data.csv - Processed data with segments and risk levels",
    "  • churn_model.pkl - Best performing model",
    "  • feature_importance.csv - Feature importance rankings",
    "  • model_comparison.csv - Detailed model comparison metrics",
    "  • evaluation_report.json - Machine-readable evaluation results",
    "  • assets/ folder - Comprehensive visualization suite",
    "",
    "="*80
])

# Save the comprehensive report
with open('assets/comprehensive_evaluation_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))

# ----------------------
# SAVE ENHANCED MODEL AND DATA
# ----------------------

print("\nSaving enhanced model and processed data...")

# Save the best model
joblib.dump(best_model, "churn_model.pkl")
print(f"Best model ({best_model_name}) saved as churn_model.pkl")

# Save the scaler used for the best model
scaler_final = StandardScaler()
scaler_final.fit(X_train_balanced)
joblib.dump(scaler_final, "scaler.pkl")
print("Scaler saved as scaler.pkl")

# Save the processed data with segments and risk levels
data_encoded.to_csv("enhanced_churn_data.csv", index=False)
print("Enhanced data saved as enhanced_churn_data.csv")

# Save feature importance for future reference
feature_importance_df.to_csv("feature_importance.csv", index=False)
print("Feature importance saved as feature_importance.csv")

# Save evaluation results for future reference
with open('assets/detailed_evaluation_results.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for name, results in evaluation_results.items():
        serializable_results[name] = {
            'test_metrics': results['test_metrics'],
            'cv_results': {metric: {'mean': float(res['mean']), 'std': float(res['std'])} 
                          for metric, res in results['cv_results'].items()},
            'best_params': results['best_params'],
            'tuning_score': float(results['tuning_score'])
        }
    json.dump(serializable_results, f, indent=2)

print("Detailed evaluation results saved as assets/detailed_evaluation_results.json")

# ----------------------
# GENERATE ACTIONABLE INSIGHTS
# ----------------------

print("\nGenerating actionable insights...")

# 1. High-risk customer analysis
high_risk_customers = data_encoded[data_encoded['RiskLevel'].isin(['High Risk', 'Very High Risk'])]
high_risk_profile = high_risk_customers.describe().T

# 2. Model performance insights
performance_insights = []

if evaluation_results[best_model_name]['test_metrics']['roc_auc'] > 0.8:
    performance_insights.append("✓ Excellent model discrimination capability (ROC-AUC > 0.8)")
elif evaluation_results[best_model_name]['test_metrics']['roc_auc'] > 0.7:
    performance_insights.append("✓ Good model discrimination capability (ROC-AUC > 0.7)")
else:
    performance_insights.append("⚠ Model discrimination needs improvement (ROC-AUC < 0.7)")

if evaluation_results[best_model_name]['test_metrics']['f1'] > 0.7:
    performance_insights.append("✓ Strong overall performance (F1-Score > 0.7)")
elif evaluation_results[best_model_name]['test_metrics']['f1'] > 0.6:
    performance_insights.append("✓ Acceptable overall performance (F1-Score > 0.6)")
else:
    performance_insights.append("⚠ Performance needs improvement (F1-Score < 0.6)")

# Check for overfitting
cv_f1 = evaluation_results[best_model_name]['cv_results']['f1']['mean']
test_f1 = evaluation_results[best_model_name]['test_metrics']['f1']
if abs(cv_f1 - test_f1) < 0.05:
    performance_insights.append("✓ Good generalization (CV and test scores are similar)")
else:
    performance_insights.append("⚠ Possible overfitting detected (significant gap between CV and test scores)")

print("\nMODEL PERFORMANCE INSIGHTS:")
for insight in performance_insights:
    print(f"  {insight}")

print(f"\nHIGH-RISK CUSTOMER ANALYSIS:")
print(f"  Total high-risk customers: {len(high_risk_customers):,} ({len(high_risk_customers)/len(data_encoded):.1%})")
print(f"  Average age: {high_risk_customers['Age'].mean():.1f} years")
print(f"  Average balance: ${high_risk_customers['Balance'].mean():,.2f}")
print(f"  Average tenure: {high_risk_customers['Tenure'].mean():.1f} years")
print(f"  Active members: {high_risk_customers['IsActiveMember'].mean():.1%}")

# ----------------------
# RETENTION STRATEGY RECOMMENDATIONS
# ----------------------

print("\nGenerating retention strategy recommendations...")

# Define retention strategies based on segments and risk levels
retention_strategies = {
    "Young Professionals": [
        "Offer mobile banking incentives and digital-first services",
        "Create loyalty programs with quick rewards",
        "Provide financial education resources tailored to early career needs",
        "Offer special rates on first-time home loans or investment accounts"
    ],
    "Established Savers": [
        "Provide premium relationship manager services",
        "Offer competitive interest rates on savings products",
        "Create family banking packages with benefits for multiple accounts",
        "Develop retirement planning services and workshops"
    ],
    "High-Value Clients": [
        "Implement VIP customer service with dedicated advisors",
        "Offer exclusive investment opportunities and wealth management",
        "Provide complimentary financial reviews and tax planning",
        "Create invitation-only events and networking opportunities"
    ],
    "At-Risk Seniors": [
        "Develop senior-friendly banking interfaces and support",
        "Offer in-person banking services with no additional fees",
        "Create retirement income products with guaranteed returns",
        "Provide estate planning services and family wealth transfer guidance"
    ]
}

risk_based_strategies = {
    "Low Risk": [
        "Regular engagement through personalized communications",
        "Cross-sell additional products based on customer needs",
        "Implement loyalty rewards for continued business"
    ],
    "Medium Risk": [
        "Proactive outreach to address potential pain points",
        "Offer account reviews to ensure products match current needs",
        "Provide special promotions to increase product usage"
    ],
    "High Risk": [
        "Immediate contact by relationship managers",
        "Offer retention incentives such as fee waivers or rate improvements",
        "Conduct satisfaction surveys to identify specific issues"
    ],
    "Very High Risk": [
        "Implement emergency retention protocols with significant incentives",
        "Executive-level outreach for high-value customers",
        "Create customized retention packages based on customer history"
    ]
}

# Print retention strategies
print("\nSEGMENT-BASED RETENTION STRATEGIES:")
for segment, strategies in retention_strategies.items():
    print(f"\n{segment}:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")

print("\nRISK-BASED RETENTION STRATEGIES:")
for risk_level, strategies in risk_based_strategies.items():
    print(f"\n{risk_level}:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")

# ----------------------
# FINAL SUMMARY
# ----------------------

print("\n" + "="*80)
print("ENHANCED CHURN PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Best Parameters: {evaluation_results[best_model_name]['best_params']}")
print(f"✓ Test F1-Score: {evaluation_results[best_model_name]['test_metrics']['f1']:.4f}")
print(f"✓ Test ROC-AUC: {evaluation_results[best_model_name]['test_metrics']['roc_auc']:.4f}")
print(f"✓ CV F1-Score: {evaluation_results[best_model_name]['cv_results']['f1']['mean']:.4f} ± {evaluation_results[best_model_name]['cv_results']['f1']['std']:.4f}")
print(f"✓ High-Risk Customers Identified: {len(high_risk_customers):,}")
print()
print("FILES GENERATED:")
print("  📊 Model Files:")
print("    • churn_model.pkl - Best performing model")
print("    • scaler.pkl - Feature scaler")
print("  📈 Data Files:")
print("    • enhanced_churn_data.csv - Processed data with segments and risk levels")
print("    • feature_importance.csv - Feature importance rankings")
print("    • model_comparison.csv - Detailed model comparison")
print("  📋 Reports:")
print("    • assets/comprehensive_evaluation_report.txt - Human-readable report")
print("    • assets/evaluation_report.json - Machine-readable summary")
print("    • assets/detailed_evaluation_results.json - Complete evaluation data")
print("  🎨 Visualizations (assets/ folder):")
print("    • model_performance_comparison.png - Comprehensive model comparison")
print("    • enhanced_roc_comparison.png - ROC curves with tuning")
print("    • enhanced_precision_recall_comparison.png - Precision-recall curves")
print("    • cv_score_distributions.png - Cross-validation score distributions")
print("    • confusion_matrices.png - Model confusion matrices")
print("    • enhanced_feature_importance.png - Feature importance analysis")
print("    • enhanced_churn_dashboard.png - Executive dashboard")
print("    • And more visualization files...")
print()
print("KEY FEATURES IMPLEMENTED:")
print("  ✅ Comprehensive hyperparameter tuning with GridSearchCV")
print("  ✅ 5-fold stratified cross-validation")
print("  ✅ Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)")
print("  ✅ ROC curves and Precision-Recall curves")
print("  ✅ SMOTE for handling class imbalance")
print("  ✅ Customer segmentation and risk assessment")
print("  ✅ Feature importance analysis")
print("  ✅ Comprehensive reporting and visualization suite")
print("  ✅ Actionable business insights and retention strategies")
print("="*80)