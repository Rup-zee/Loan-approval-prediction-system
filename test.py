import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle as pk

# Load and preprocess data
data = pd.read_csv("loan_approval_dataset.csv")
data.drop(columns=['loan_id'], inplace=True)
data.columns = data.columns.str.strip()

# Create combined assets feature
data['Assets'] = (data['residential_assets_value'] + 
                 data['commercial_assets_value'] + 
                 data['luxury_assets_value'] + 
                 data['bank_asset_value'])
data.drop(columns=['residential_assets_value', 'commercial_assets_value', 
                 'luxury_assets_value', 'bank_asset_value'], inplace=True)

# Create financial ratios that will help balance the decision
data['loan_to_income'] = data['loan_amount'] / (data['income_annum'] + 1)
data['asset_coverage'] = data['Assets'] / (data['loan_amount'] + 1)
data['debt_burden'] = data['loan_amount'] / (data['income_annum'] * (data['loan_term'] / 12) + 1)

# Clean and encode categorical features
def clean_data(st):
    return st.strip()

data['education'] = data['education'].apply(clean_data)
data['education'] = data['education'].replace(['Graduate', 'Not Graduate'], [1, 0])

data['self_employed'] = data['self_employed'].apply(clean_data)
data['self_employed'] = data['self_employed'].replace(['No', 'Yes'], [0, 1])

data['loan_status'] = data['loan_status'].apply(clean_data)
data['loan_status'] = data['loan_status'].replace(['Approved', 'Rejected'], [1, 0])

# Prepare data for training
input_data = data.drop(columns=['loan_status'])
output_data = data['loan_status']
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train model with balanced class weights
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(x_train_scaled, y_train)

# Evaluate model
print("Test set performance:")
print(classification_report(y_test, model.predict(x_test_scaled)))

# Save model artifacts
with open('loan_model.pkl', 'wb') as f:
    pk.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pk.dump(scaler, f)

# Calculate and save feature importance
feature_importance = model.feature_importances_
feature_names = input_data.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

with open('feature_importance.pkl', 'wb') as f:
    pk.dump(importance_df, f)

print("\nFeature Importance:")
print(importance_df)
print("\nModel training complete. Artifacts saved.")
