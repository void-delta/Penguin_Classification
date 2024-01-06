from sklearn.preprocessing import LabelEncoder

def preprocess_input(features):
    label_encoder = LabelEncoder()

    # Assuming 'island' column is present in features
    features_encoded = features.copy()
    features_encoded['island'] = label_encoder.fit_transform(features['island'])
    
    # Add any additional preprocessing steps here if needed
    
    return features_encoded
