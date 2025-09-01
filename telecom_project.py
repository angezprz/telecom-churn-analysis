#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# In[2]:


file_path = (r"C:\Users\USER\Downloads\telecom\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(file_path)


# In[3]:


df.head()


# # Cleaning

# In[4]:


# Drop customerID (not useful for modeling)
df = df.drop("customerID", axis=1)


# In[5]:


# Convert TotalCharges to numeric (coerce errors -> NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# In[6]:


# Likely tenure=0 customers will have NaN in TotalCharges → Drop them
df = df[df["TotalCharges"].notnull()]


# In[7]:


# Reset index
df = df.reset_index(drop=True)


# In[8]:


# Convert SeniorCitizen to categorical (0 = No, 1 = Yes)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})


# In[9]:


print(df["TotalCharges"].isnull().sum())  # should show 11


# # Preprocessing

# In[10]:


X = df.drop("Churn", axis=1)   # features
y = df["Churn"].map({"No":0, "Yes":1})  # target as binary


# In[11]:


# Identify categorical & numeric
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(include=["int64","float64"]).columns

# One-Hot Encode categoricals
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("Shape after encoding:", X_encoded.shape)


# In[12]:


scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)


# # EDA

# In[14]:


sns.countplot(x=y, palette="Set2")
plt.title("Churn Distribution")
plt.show()
print(y.value_counts(normalize=True))


# In[15]:


for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    sns.boxplot(x=df["Churn"], y=df[col], palette="Set2")
    plt.title(f"{col} vs Churn")
    plt.show()


# In[16]:


sns.countplot(x="Contract", hue="Churn", data=df, palette="Set1")
plt.title("Contract Type vs Churn")
plt.show()

sns.countplot(x="InternetService", hue="Churn", data=df, palette="Set1")
plt.title("Internet Service vs Churn")
plt.show()


# In[17]:


sns.countplot(x="Contract", hue="Churn", data=df, palette="Set1")
plt.title("Contract Type vs Churn")
plt.show()

sns.countplot(x="InternetService", hue="Churn", data=df, palette="Set1")
plt.title("Internet Service vs Churn")
plt.show()


# # Modelling

# # Baseline Model (Logistic Regression)

# In[18]:


logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:,1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))


# # === 2. Random Forest ===

# In[19]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))


# # === 3. XGBoost ==

# In[20]:


from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                    random_state=42, scale_pos_weight=3)  # handle imbalance
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:,1]

print("\n=== XGBoost ===")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))


# # Baseline Model (KMeans)

# In[21]:


from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# 1. Prepare features
cluster_features = ["tenure", "MonthlyCharges", "TotalCharges"]
X_cluster = df[cluster_features]

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)


# In[22]:


print("Silhouette Scores for different k:")
for k in range(2,8):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_cluster_scaled)
    score = silhouette_score(X_cluster_scaled, labels)
    print(f"KMeans k={k}, Silhouette={score:.3f}")


# In[23]:


km2 = KMeans(n_clusters=2, random_state=42)
df["Cluster2"] = km2.fit_predict(X_cluster_scaled)

profile2 = df.groupby("Cluster2")[["tenure","MonthlyCharges","TotalCharges"]].mean()
profile2["ChurnRate"] = df.groupby("Cluster2")["Churn"].apply(lambda x: (x=="Yes").mean()*100)

print("\n=== KMeans k=2 Cluster Profiles ===")
print(profile2.round(2))
print("\nCluster Sizes (k=2):\n", df["Cluster2"].value_counts())

# Assign business-friendly names for k=2
cluster2_names = {}
for cluster in profile2.index:
    avg_churn = profile2.loc[cluster,"ChurnRate"]
    avg_charge = profile2.loc[cluster,"MonthlyCharges"]
    avg_tenure = profile2.loc[cluster,"tenure"]
    
    if avg_churn > 25:
        cluster2_names[cluster] = "At-Risk Mid Spenders"
    else:
        cluster2_names[cluster] = "Loyal High Spenders"

df["Cluster2_Name"] = df["Cluster2"].map(cluster2_names)


# In[24]:


cluster2_names = {}
for cluster in profile2.index:
    avg_churn = profile2.loc[cluster,"ChurnRate"]
    avg_charge = profile2.loc[cluster,"MonthlyCharges"]
    avg_tenure = profile2.loc[cluster,"tenure"]
    
    if avg_churn > 25:
        cluster2_names[cluster] = "At-Risk Mid Spenders"
    else:
        cluster2_names[cluster] = "Loyal High Spenders"

df["Cluster2_Name"] = df["Cluster2"].map(cluster2_names)


# In[25]:


km4 = KMeans(n_clusters=4, random_state=42)
df["Cluster4"] = km4.fit_predict(X_cluster_scaled)

profile4 = df.groupby("Cluster4")[["tenure","MonthlyCharges","TotalCharges"]].mean()
profile4["ChurnRate"] = df.groupby("Cluster4")["Churn"].apply(lambda x: (x=="Yes").mean()*100)

print("\n=== KMeans k=4 Cluster Profiles ===")
print(profile4.round(2))
print("\nCluster Sizes (k=4):\n", df["Cluster4"].value_counts())

# Assign business-friendly names for k=4
cluster4_names = {}
for cluster in profile4.index:
    avg_churn = profile4.loc[cluster,"ChurnRate"]
    avg_charge = profile4.loc[cluster,"MonthlyCharges"]
    avg_tenure = profile4.loc[cluster,"tenure"]
    
    if avg_churn > 40 and avg_charge > 70:
        cluster4_names[cluster] = "At-Risk High Spenders"
    elif avg_churn < 20 and avg_tenure > 30:
        cluster4_names[cluster] = "Loyal Premium Customers"
    elif avg_charge < 50 and avg_tenure < 15:
        cluster4_names[cluster] = "New Budget Users"
    else:
        cluster4_names[cluster] = "Stable Low Spenders"

df["Cluster4_Name"] = df["Cluster4"].map(cluster4_names)


# # Classification + Segmentation

# In[26]:


# Use your best classification model (Logistic Regression or XGBoost)
df["Churn_Prob"] = logreg.predict_proba(X_encoded)[:,1]  # if using Logistic Regression


# In[27]:


# Average churn probability per cluster
risk2 = df.groupby("Cluster2_Name")["Churn_Prob"].mean().sort_values(ascending=False)
risk4 = df.groupby("Cluster4_Name")["Churn_Prob"].mean().sort_values(ascending=False)

print("\nAverage Churn Probability (k=2):\n", risk2.round(2))
print("\nAverage Churn Probability (k=4):\n", risk4.round(2))


# In[30]:


# Create a function to evaluate models
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Churn=1)": precision_score(y_test, y_pred),
        "Recall (Churn=1)": recall_score(y_test, y_pred),
        "F1 (Churn=1)": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
    }

# Evaluate all models
results = []
results.append(evaluate_model("Logistic Regression", logreg, X_test, y_test))
results.append(evaluate_model("Random Forest", rf, X_test, y_test))
results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

# Create DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# If you want a cleaner display (excluding confusion matrices):
display_cols = ["Model","Accuracy","Precision (Churn=1)","Recall (Churn=1)","F1 (Churn=1)","ROC-AUC"]
print("\n=== Performance Summary (without confusion matrices) ===")
print(results_df[display_cols].round(3))


# In[ ]:




