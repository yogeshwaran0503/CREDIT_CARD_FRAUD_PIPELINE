from sklearn.ensemble import RandomForestClassifier

def build_model(n_estimators=100, max_depth=None, random_state=42):
    # Use class_weight='balanced' to help with class imbalance
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=random_state)
