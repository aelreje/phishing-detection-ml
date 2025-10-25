import pandas as pd
from sklearn.model_selection import train_test_split
import re # (Stands for Regular Expressions, for finding URLs)
from bs4 import BeautifulSoup # (For parsing HTML emails)
from urllib.parse import urlparse # (For analyzing URLs)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib # For saving our model!

import numpy as np

import datetime # For creating timestamps
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay



# --- 1. Load the Dataset ---
print("Loading master_phishing_dataset.csv...")
try:
    # We use our clean, master dataset
    df = pd.read_csv('data-cleaning/master_phishing_dataset.csv')
except FileNotFoundError:
    print("Error: 'master_phishing_dataset.csv' not found.")
    print("Please make sure your 'main.py' is in the same folder as your dataset.")
    exit()

# Handle any rows that might be missing 'text' (just in case)
df = df.dropna(subset=['text'])

print(f"Loaded {len(df)} total emails.")

# --- 2. Separate Features (X) from Target (y) ---
X = df['text']  # The 'clues' (email text)
y = df['label'] # The 'answer' (0 for benign, 1 for phishing)

# --- 3. Create a Training and Testing Split ---
# We'll use 80% for training, 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% of data goes to testing
    random_state=42,  # A 'seed' to ensure we get the same 'random' split every time
    stratify=y        # This makes sure the 80/20 split has the same
                      # percentage of phishing/benign emails as the original
)

print(f"Training on {len(X_train)} emails.")
print(f"Testing on {len(X_test)} emails.")


# --- 4. Build Feature Extractors ---

# Rule 1: Suspicious Keywords
# (Professor's "Content features")
SUSPICIOUS_KEYWORDS = [
    # High-confidence phishing words
    'urgent', 'password', 'security alert', 'ssn', 'credit card', 
    'unusual activity', 'action required', 'suspend', 'locked',
    
    # Common but slightly less suspicious (we can keep a few)
    'bank', 'login', 'invoice', 'payment', 'verify your'
]

# --- 2. check_keywords(email_body) (IMPROVED) ---
def check_keywords(email_body):
    """Counts how many suspicious keywords are in the email."""
    count = 0
    # We add spaces to the beginning and end to catch keywords
    # at the very start or end of the email.
    email_body_lower = ' ' + str(email_body).lower() + ' '
    
    for keyword in SUSPICIOUS_KEYWORDS:
        # We also add spaces to the keyword
        if ' ' + keyword + ' ' in email_body_lower:
            count += 1
    return count



# --- Rule 2: URL & Link Features (DEBUGGING VERSION) ---
def extract_url_features(email_body):
    """
    Extracts features from URLs and <a> tags in an email body.
    This handles both plain text and HTML.
    """
    email_body = str(email_body) # Ensure it's a string
    url_count = 0
    ip_in_url_count = 0
    mismatch_count = 0
    # domain_in_text_count = 0 # <-- OUR NEW DEBUG COUNTER
    
    # Check for HTML content
    if '<html>' in email_body.lower():
        soup = BeautifulSoup(email_body, 'html.parser')
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not href:
                continue
            
            url_count += 1
            
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', href):
                ip_in_url_count += 1

            link_text = a.text.strip().lower()
            try:
                # Use a MORE GENERAL regex
                domain_match = re.search(r'([a-zA-Z0-9-]{2,}\.[a-zA-Z]{2,})', link_text)
                
                if domain_match:
                    # domain_in_text_count += 1 # <-- WE FOUND ONE!
                    
                    link_domain_text = domain_match.group(0) 
                    href_domain = urlparse(href).netloc 
                    
                    if href_domain and link_domain_text not in href_domain:
                        mismatch_count += 1
                
            except Exception:
                pass
    else:
        # Plain text email
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
        url_count = len(urls)
        for url in urls:
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
                ip_in_url_count += 1

    return {
        'url_count': url_count,
        'ip_in_url': 1 if ip_in_url_count > 0 else 0, 
        'link_mismatch': 1 if mismatch_count > 0 else 0,
        # 'domain_in_text_count': domain_in_text_count # <-- RETURN THE NEW COUNTER
    }






# # --- 5. Let's Test Our Functions ---
# print("\n--- Testing our feature extractors ---")

# # Test 1: Fake Phishing Email (HTML with mismatch)
# test_email_1 = """
# <html>
#   <body>
#     <p>Urgent action required! Please update your account password now.</p>
#     <a href="http://scam-site.com/login">Click here to verify your bank account</a>
#     <a href="http://192.168.4.1/phish">Login here: my-bank.com</a>
#   </body>
# </html>
# """

# # Test 2: Fake Benign Email (Plain text)
# test_email_2 = """
# Hey, here's the meeting agenda for tomorrow. 
# You can find the doc at http://docs.google.com/real-link.
# See you then!
# """

# # Test the keyword checker
# print(f"Keyword Score (Email 1): {check_keywords(test_email_1)}")
# print(f"Keyword Score (Email 2): {check_keywords(test_email_2)}")

# # Test the URL checker
# url_features_1 = extract_url_features(test_email_1)
# url_features_2 = extract_url_features(test_email_2)

# print(f"\nURL Features (Email 1): {url_features_1}")
# print(f"URL Features (Email 2): {url_features_2}")


# --- 6. Implement Rule-Based Scoring Function ---
def get_rule_based_score(features):
    """Calculates a simple weighted score from our rules."""
    score = 0
    explanation = [] # We'll store the reasons for the score

    # High-risk rules (add more points)
    if features['link_mismatch'] == 1:
        score += 3
        explanation.append("[HIGH RISK] Link text hides a different domain.")
    if features['ip_in_url'] == 1:
        score += 3
        explanation.append("[HIGH RISK] URL contains an IP address.")
    
    # Low-risk rules
    # We use features['keyword_count'] directly
    keyword_score = features['keyword_count']
    score += keyword_score
    if keyword_score > 0:
        explanation.append(f"[LOW RISK] Found {keyword_score} suspicious keyword(s).")
    
    return score, explanation



# --- 7. Create Master Feature Engineering Function (DEBUGGING VERSION) ---
def build_rule_features(data_series):
    """
    Applies all our feature extractors to a pandas Series (like X_train).
    Returns a new DataFrame with all our rule-based features.
    """
    print("Starting rule-based feature extraction...")
    
    features_df = pd.DataFrame()
    
    # 1. Apply keyword checker
    features_df['keyword_count'] = data_series.apply(check_keywords)
    
    # 2. Apply URL feature extractor
    print("Extracting URL features (this may take a moment)...")
    url_features = data_series.apply(extract_url_features)
    
    features_df['url_count'] = url_features.apply(lambda x: x['url_count'])
    features_df['ip_in_url'] = url_features.apply(lambda x: x['ip_in_url'])
    features_df['link_mismatch'] = url_features.apply(lambda x: x['link_mismatch'])
    
    # --- ADD THIS NEW LINE ---
    # .get() is safer, it returns 0 if the key doesn't exist
    # features_df['domain_in_text_count'] = url_features.apply(lambda x: x.get('domain_in_text_count', 0))
    # --- END OF NEW LINE ---
    
    print("Rule-based feature extraction complete!")
    return features_df



# --- 8. Build Features for Training and Testing Sets ---

# We can now delete our old test section.
# This is the real deal:
print("\n--- Building Features for TRAINING Data ---")
X_train_rules = build_rule_features(X_train)

print("\n--- Building Features for TESTING Data ---")
X_test_rules = build_rule_features(X_test)

# --- 9. Let's see what we built! ---
print("\n--- Example Features from Training Data (First 5 rows) ---")
print(X_train_rules.head())

print("\n--- Summary of Features ---")
# .describe() gives us a quick summary (avg, min, max, etc.)
print(X_train_rules.describe())





# --- 10. Train the "ML-lite" Model ---
print("\n--- Training the ML Model ---")

# --- Part A: Create the TF-IDF Features ---
print("Building TF-IDF features (this is the slowest step)...")
# We create the 'vectorizer' tool
# stop_words='english' removes common words (the, a, is)
# max_features=5000 tells it to only keep the 5,000 most important words
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# We 'fit' it on the training data (to learn the words)
# and 'transform' it into a number matrix
X_train_tfidf = vectorizer.fit_transform(X_train)

# We ONLY 'transform' the test data (to use the words it already learned)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF features built.")

# --- Part B: Combine TF-IDF with our Rule Features ---
# 'hstack' (horizontal stack) is a tool to "glue" our two feature
# tables together side-by-side.

# X_train_rules is our pandas DataFrame. .values gets the raw numbers.
X_train_combined = hstack([X_train_tfidf, X_train_rules.values])
X_test_combined = hstack([X_test_tfidf, X_test_rules.values])

print("Combined features ready.")

# --- Part C: Train the Logistic Regression Model ---
print("Fitting the Logistic Regression model...")

# We create the model.
# class_weight='balanced' helps a lot, as it auto-handles
# if we have more phishing than benign emails (or vice-versa).
ml_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# We 'fit' the model on our combined training data!
ml_model.fit(X_train_combined, y_train)

print("Model training complete!")

# --- 11. Evaluate the Model ---
print("\n--- MODEL EVALUATION ON TEST DATA ---")
# Now we use the 'test' data the model has NEVER seen
y_pred = ml_model.predict(X_test_combined)

# This report gives us Accuracy, Precision, and Recall
# '0' is Benign, '1' is Phishing
report = classification_report(y_test, y_pred, target_names=['Benign (0)', 'Phishing (1)'])
print(report)


# --- 11a. Show Top "Suspicious" Words (Generated by the Model) ---
print("\n--- Top 20 Most 'Phishing-like' Words (from ML Model) ---")

try:
    # Get the 5,000 words our vectorizer learned
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get the "weights" the model assigned to each word.
    # We look at [0] because it's a binary model.
    model_weights = ml_model.coef_[0]
    
    # The first 5,000 weights belong to the TF-IDF words
    tfidf_weights = model_weights[:5000]
    
    # Sort the words by their weight (most "phishing-like" first)
    top_indices = tfidf_weights.argsort()[::-1][:20]
    
    for i, index in enumerate(top_indices):
        word = feature_names[index]
        weight = tfidf_weights[index]
        print(f"{i+1: >2}. {word: <15} (Weight: {weight:.4f})")

except Exception as e:
    print(f"Could not print top features: {e}")





# --- 12. Save Your Model ---
# We save our trained model and the vectorizer to disk.
# This way, our 'demo' script can use them later!
joblib.dump(ml_model, 'models/phishing_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("\nModel and Vectorizer saved to 'phishing_model.pkl' and 'tfidf_vectorizer.pkl'")


# --- 13. Create and Save Visualizations (with Timestamps) ---
print("\n--- Generating and saving visualizations ---")

# Generate a unique timestamp for this training run
# This will look like: 20251025-113045
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"Using timestamp for filenames: {timestamp}")

# --- 1. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Model Performance: Confusion Matrix")
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, 
    ax=ax, 
    cmap='Blues', 
    display_labels=['Benign', 'Phishing']
)
# Add timestamp to the filename
filename_cm = f'visuals/confusion_matrix_{timestamp}.png'
plt.savefig(filename_cm)
print(f"Saved '{filename_cm}'")

# --- 2. ROC Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(
    ml_model, X_test_combined, y_test, 
    ax=ax, 
    name='Logistic Regression'
)
ax.set_title("ROC (Receiver Operating Characteristic) Curve")
# Add timestamp to the filename
filename_roc = f'visuals/roc_curve_{timestamp}.png'
plt.savefig(filename_roc)
print(f"Saved '{filename_roc}'")

# --- 3. Precision-Recall Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(
    ml_model, X_test_combined, y_test, 
    ax=ax, 
    name='Logistic Regression'
)
ax.set_title("Precision-Recall Curve")
# Add timestamp to the filename
filename_pr = f'visuals/precision_recall_curve_{timestamp}.png'
plt.savefig(filename_pr)
print(f"Saved '{filename_pr}'")

print("\nAll visualizations saved! Project training is complete.")