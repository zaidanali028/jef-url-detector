import pandas as pd
import numpy as np
import joblib
import validators
import tldextract
from urllib.parse import urlparse
import re

# Load the saved model and scaler
svm_classifier = joblib.load('phishing_detector_svm.joblib')  # Your trained SVM model
scaler = joblib.load('scaler.joblib')  # Scaler used for training
selected_features = joblib.load('selected_features.joblib')  # Selected features

def count_characters(text, characters):
    """Count the occurrences of specific characters in a given text."""
    return sum(text.count(char) for char in characters)

def extract_features(url):
    """Extract features from the URL for prediction."""
    features = {}
    
    # Ensure the URL is parsed correctly
    parsed = urlparse(url)
    
    # Extract domain using tldextract
    ext = tldextract.extract(url)
    domain = ext.domain + '.' + ext.suffix
    subdomain = ext.subdomain

    # 1. URL Level Features
    features['qty_dot_url'] = url.count('.')
    features['qty_hyphen_url'] = url.count('-')
    features['qty_underline_url'] = url.count('_')
    features['qty_slash_url'] = url.count('/')
    features['qty_questionmark_url'] = url.count('?')
    features['qty_equal_url'] = url.count('=')
    features['qty_at_url'] = url.count('@')
    features['qty_and_url'] = url.count('&')
    features['qty_exclamation_url'] = url.count('!')
    features['qty_space_url'] = url.count(' ')
    features['qty_tilde_url'] = url.count('~')
    features['qty_comma_url'] = url.count(',')
    features['qty_plus_url'] = url.count('+')
    features['qty_asterisk_url'] = url.count('*')
    features['qty_hashtag_url'] = url.count('#')
    features['qty_dollar_url'] = url.count('$')
    features['qty_percent_url'] = url.count('%')
    features['qty_vowels_domain'] = len(re.findall(r'[aeiouAEIOU]', domain))
    features['length_url'] = len(url)
    
    # 2. Domain Level Features
    features['qty_dot_domain'] = domain.count('.')
    features['qty_hyphen_domain'] = domain.count('-')
    features['qty_underline_domain'] = domain.count('_')
    features['qty_slash_domain'] = domain.count('/')
    features['qty_questionmark_domain'] = domain.count('?')
    features['qty_equal_domain'] = domain.count('=')
    features['qty_at_domain'] = domain.count('@')
    features['qty_and_domain'] = domain.count('&')
    features['qty_exclamation_domain'] = domain.count('!')
    features['qty_space_domain'] = domain.count(' ')
    features['qty_tilde_domain'] = domain.count('~')
    features['qty_comma_domain'] = domain.count(',')
    features['qty_plus_domain'] = domain.count('+')
    features['qty_asterisk_domain'] = domain.count('*')
    features['qty_hashtag_domain'] = domain.count('#')
    features['qty_dollar_domain'] = domain.count('$')
    features['qty_percent_domain'] = domain.count('%')
    features['domain_length'] = len(domain)
    features['domain_in_ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0  # Check if domain is an IP
    
    # 3. Directory Level Features
    directory = parsed.path
    features['qty_dot_directory'] = directory.count('.')
    features['qty_hyphen_directory'] = directory.count('-')
    features['qty_underline_directory'] = directory.count('_')
    features['qty_slash_directory'] = directory.count('/')
    features['qty_questionmark_directory'] = directory.count('?')
    features['qty_equal_directory'] = directory.count('=')
    features['qty_at_directory'] = directory.count('@')
    features['qty_and_directory'] = directory.count('&')
    features['qty_exclamation_directory'] = directory.count('!')
    features['qty_space_directory'] = directory.count(' ')
    features['qty_tilde_directory'] = directory.count('~')
    features['qty_comma_directory'] = directory.count(',')
    features['qty_plus_directory'] = directory.count('+')
    features['qty_asterisk_directory'] = directory.count('*')
    features['qty_hashtag_directory'] = directory.count('#')
    features['qty_dollar_directory'] = directory.count('$')
    features['qty_percent_directory'] = directory.count('%')
    features['directory_length'] = len(directory)
    
    # 4. File Level Features
    file = parsed.path.split('/')[-1]
    features['qty_dot_file'] = file.count('.')
    features['qty_hyphen_file'] = file.count('-')
    features['qty_underline_file'] = file.count('_')
    features['qty_slash_file'] = file.count('/')
    features['qty_questionmark_file'] = file.count('?')
    features['qty_equal_file'] = file.count('=')
    features['qty_at_file'] = file.count('@')
    features['qty_and_file'] = file.count('&')
    features['qty_exclamation_file'] = file.count('!')
    features['qty_space_file'] = file.count(' ')
    features['qty_tilde_file'] = file.count('~')
    features['qty_comma_file'] = file.count(',')
    features['qty_plus_file'] = file.count('+')
    features['qty_asterisk_file'] = file.count('*')
    features['qty_hashtag_file'] = file.count('#')
    features['qty_dollar_file'] = file.count('$')
    features['qty_percent_file'] = file.count('%')
    features['file_length'] = len(file)
    
    # 5. Parameters Level Features
    params = parsed.query
    features['qty_dot_params'] = params.count('.')
    features['qty_hyphen_params'] = params.count('-')
    features['qty_underline_params'] = params.count('_')
    features['qty_slash_params'] = params.count('/')
    features['qty_questionmark_params'] = params.count('?')
    features['qty_equal_params'] = params.count('=')
    features['qty_at_params'] = params.count('@')
    features['qty_and_params'] = params.count('&')
    features['qty_exclamation_params'] = params.count('!')
    features['qty_space_params'] = params.count(' ')
    features['qty_tilde_params'] = params.count('~')
    features['qty_comma_params'] = params.count(',')
    features['qty_plus_params'] = params.count('+')
    features['qty_asterisk_params'] = params.count('*')
    features['qty_hashtag_params'] = params.count('#')
    features['qty_dollar_params'] = params.count('$')
    features['qty_percent_params'] = params.count('%')
    features['params_length'] = len(params)
    features['tld_present_params'] = 1 if ext.suffix else 0  # Example feature
    
    # 6. Other Features (as per your dataset)
    # Add more feature extractions as needed based on your dataset and selected_features
    
    # Ensure all selected features are present
    for feature in selected_features:
        if feature not in features:
            features[feature] = 0  # or appropriate default value
    
    # Convert features to DataFrame
    feature_vector = pd.DataFrame([features])
    
    # Select only the required features
    return feature_vector[selected_features]

def predict_url(url):
    """Predict whether a URL is phishing or legitimate."""
    # Validate the URL format
    if not validators.url(url):
        return {"error": "Invalid URL format."}

    # Extract features from the URL
    features = extract_features(url)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = svm_classifier.predict(features_scaled)
    probability = svm_classifier.predict_proba(features_scaled)[:,1]

    return {
        'is_phishing': bool(prediction[0]),
        'confidence': float(probability[0])  # Confidence score for phishing
    }

# Example usage
if __name__ == "__main__":
    url_to_test = "http://example.com"  # Replace with the URL you want to test
    result = predict_url(url_to_test)
    if 'error' in result:
        print(f"URL: {url_to_test}, Error: {result['error']}")
    else:
        status = "Phishing" if result['is_phishing'] else "Legitimate"
        print(f"URL: {url_to_test}, Status: {status}, Confidence: {result['confidence']:.2f}")
