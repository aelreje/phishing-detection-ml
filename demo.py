import sys
import email 
from email.policy import default
import joblib 
import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scipy.sparse import hstack

# ---
# --- STEP 1: YOUR COPIED FUNCTIONS ---
# ---

# 1. SUSPICIOUS_KEYWORDS (the list)
# FINAL KEYWORD LIST (High-Precision)
SUSPICIOUS_KEYWORDS = [
    # High-confidence phishing words
    'urgent', 'security alert', 'ssn', 'credit card', 
    'unusual activity', 'action required', 'suspend', 'locked',
    'bank', 'invoice', 'payment', 'verify your'
]

# 2. check_keywords(email_body) (IMPROVED - Whole Word Check)
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

# 3. extract_url_features(email_body)
def extract_url_features(email_body):
    """
    Extracts features from URLs and <a> tags in an email body.
    This handles both plain text and HTML.
    """
    email_body = str(email_body) 
    url_count = 0
    ip_in_url_count = 0
    mismatch_count = 0
    
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
                domain_match = re.search(r'([a-zA-Z0-9-]{2,}\.[a-zA-Z]{2,})', link_text)
                
                if domain_match:
                    link_domain_text = domain_match.group(0) 
                    href_domain = urlparse(href).netloc 
                    
                    if href_domain and link_domain_text not in href_domain:
                        mismatch_count += 1
                
            except Exception:
                pass
    else:
        # --- THIS 'ELSE' BLOCK IS NOW FIXED ---
        # It no longer has the duplicated/broken loop
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
        url_count = len(urls)
        for url in urls:
            # This is the correct check
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
                ip_in_url_count += 1
        # --- END OF FIX ---

    return {
        'url_count': url_count,
        'ip_in_url': 1 if ip_in_url_count > 0 else 0, 
        'link_mismatch': 1 if mismatch_count > 0 else 0,
    }

# 4. get_rule_based_score(features)
def get_rule_based_score(features):
    """Calculates a simple weighted score from our rules."""
    score = 0
    explanation = [] 

    if features.get('link_mismatch') == 1:
        score += 3
        explanation.append("[HIGH RISK] Link text hides a different domain.")
    if features.get('ip_in_url') == 1:
        score += 3
        explanation.append("[HIGH RISK] URL contains an IP address.")
    
    keyword_score = features.get('keyword_count', 0)
    score += keyword_score
    if keyword_score > 0:
        explanation.append(f"[LOW RISK] Found {keyword_score} suspicious keyword(s).")
    
    return score, explanation

# ---
# --- STEP 2: FUNCTIONS FOR THE DEMO ---
# ---

def parse_eml(file_path):
    """
    Parses an .eml file and extracts headers and body.
    Returns (headers_dict, email_body)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f, policy=default)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    email_body = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            if ctype in ['text/plain', 'text/html'] and 'attachment' not in cdispo:
                try:
                    email_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                except Exception:
                    continue
                if ctype == 'text/html':
                    break
    else:
        try:
            email_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except Exception:
            email_body = ""

    return msg, email_body


def analyze_email(file_path, model, vectorizer):
    """
    Main function to analyze a single email file.
    """
    print(f"\nAnalyzing '{file_path}'...")
    
    headers, body = parse_eml(file_path)
    
    # Handle if file could not be read
    if body is None:
        return

    if not body:
        print("Could not extract a text or HTML body from this email.")
        return

    # 1. Extract Rule Features
    url_features = extract_url_features(body)
    all_rule_features = {
        'keyword_count': check_keywords(body),
        'url_count': url_features['url_count'],
        'ip_in_url': url_features['ip_in_url'],
        'link_mismatch': url_features['link_mismatch']
    }

    # 2. Get Rule-Based Score & Explanation
    rule_score, explanation = get_rule_based_score(all_rule_features)
    
    # 3. Prepare data for the ML model
    body_tfidf = vectorizer.transform([body])
    
    rule_vector = np.array([[
        all_rule_features['keyword_count'],
        all_rule_features['url_count'],
        all_rule_features['ip_in_url'],
        all_rule_features['link_mismatch']
    ]])
    
    combined_features_vector = hstack([body_tfidf, rule_vector])

    # 4. Get ML Model Prediction
    ml_prob = model.predict_proba(combined_features_vector)[0][1] # Get phishing prob
    ml_prediction = model.predict(combined_features_vector)[0]     # Get 0 or 1
    
    # 5. Show Final Report
    print("\n--- PHISHING DETECTION REPORT ---")
    print(f"File: {file_path}")
    print("---------------------------------")
    
    # Set a new, higher threshold to reduce false positives
    CONFIDENCE_THRESHOLD = 0.70  # 70%

    # We will make our decision based on the *probability* (ml_prob), not the 50/50 prediction
    final_decision = "🚨 PHISHING 🚨" if ml_prob >= CONFIDENCE_THRESHOLD else "✅ BENIGN ✅"
    
    print(f"Overall Classification: {final_decision}")
    print(f"ML Model Score (Probability of Phishing): {ml_prob * 100:.2f}%")
    print(f"Rule-Based Score: {rule_score}")
    
    print("\nExplanation (Key Features Found):")
    if not explanation:
        print(" - No suspicious rules triggered.")
    else:
        for line in explanation:
            print(f" - {line}")
    print("---------------------------------")


# ---
# --- STEP 3: MAIN EXECUTION ---
# ---
if __name__ == "__main__":
    # Load the trained model and vectorizer from the 'models' folder
    try:
        model = joblib.load('models/phishing_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("Error: Model files not found in 'models/' folder.")
        print("Please run 'main.py' first to train and save the models!")
        sys.exit(1)

    # Get the file path from the command line
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_email.eml>")
        print("Example: python demo.py my_test_email.eml")
        sys.exit(1)
        
    # --- THIS IS NEW ---
    # This lets you run multiple files at once
    # e.g.: python demo.py email1.eml email2.eml
    files_to_check = sys.argv[1:] # Get all arguments after the script name
    
    for file_path in files_to_check:
        analyze_email(file_path, model, vectorizer)