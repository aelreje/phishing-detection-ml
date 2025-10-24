import pandas as pd
from sklearn.model_selection import train_test_split
import re # (Stands for Regular Expressions, for finding URLs)
from bs4 import BeautifulSoup # (For parsing HTML emails)
from urllib.parse import urlparse # (For analyzing URLs)

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
    'urgent', 'verify', 'account', 'password', 'security', 'alert',
    'bank', 'login', 'confirm', 'ssn', 'credit card', 'update',
    'invoice', 'payment', 'unusual activity', 'action required'
]

def check_keywords(email_body):
    """Counts how many suspicious keywords are in the email."""
    count = 0
    # .lower() makes the search case-insensitive
    email_body_lower = str(email_body).lower() 
    
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in email_body_lower:
            count += 1
    return count

# --- Rule 2: URL & Link Features (IMPROVED) ---
def extract_url_features(email_body):
    """
    Extracts features from URLs and <a> tags in an email body.
    This handles both plain text and HTML.
    """
    email_body = str(email_body) # Ensure it's a string
    url_count = 0
    ip_in_url_count = 0
    mismatch_count = 0
    
    # Check for HTML content
    if '<html>' in email_body.lower():
        soup = BeautifulSoup(email_body, 'html.parser')
        
        # Find all <a> tags (links)
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not href:
                continue
            
            url_count += 1
            
            # Check for IP address in URL
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', href):
                ip_in_url_count += 1

            # --- START OF IMPROVED MISMATCH CHECK ---
            link_text = a.text.strip().lower() # The visible text
            try:
                # Use regex to find a domain-like string in the link text
                # This looks for patterns like 'something.com' or 'something.net'
                domain_match = re.search(r'([a-zA-Z0-9-]+\.(com|net|org|gov|edu))', link_text)
                
                if domain_match:
                    # We found a domain in the link text, e.g., "my-bank.com"
                    link_domain_text = domain_match.group(0) 
                    
                    # Get the domain from the actual link destination
                    href_domain = urlparse(href).netloc # e.g., "192.168.4.1"
                    
                    # Check if the visible domain is IN the destination domain
                    # e.g., "my-bank.com" in "login.my-bank.com" -> OK
                    # e.g., "my-bank.com" in "scam-site.com" -> MISMATCH
                    # e.g., "my-bank.com" in "192.168.4.1" -> MISMATCH
                    if href_domain and link_domain_text not in href_domain:
                        print(f"--- DEBUG: Mismatch found! Text domain '{link_domain_text}' not in href domain '{href_domain}'")
                        mismatch_count += 1
                
            except Exception:
                pass # Ignore malformed URLs
            # --- END OF IMPROVED MISMATCH CHECK ---
    else:
        # Plain text email: just find URLs using regex
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
        url_count = len(urls)
        for url in urls:
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
                ip_in_url_count += 1

    return {
        'url_count': url_count,
        'ip_in_url': 1 if ip_in_url_count > 0 else 0, # Return 1 (yes) or 0 (no)
        'link_mismatch': 1 if mismatch_count > 0 else 0
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



# --- 7. Create Master Feature Engineering Function ---
def build_rule_features(data_series):
    """
    Applies all our feature extractors to a pandas Series (like X_train).
    Returns a new DataFrame with all our rule-based features.
    """
    print("Starting rule-based feature extraction...")
    
    # Create an empty DataFrame to store our new features
    features_df = pd.DataFrame()
    
    # 1. Apply keyword checker
    # .apply() runs the 'check_keywords' function on every email
    features_df['keyword_count'] = data_series.apply(check_keywords)
    
    # 2. Apply URL feature extractor
    # This is more complex since 'extract_url_features' returns a dictionary.
    # We use a 'lambda' function to pull out each piece.
    print("Extracting URL features (this may take a moment)...")
    url_features = data_series.apply(extract_url_features)
    
    features_df['url_count'] = url_features.apply(lambda x: x['url_count'])
    features_df['ip_in_url'] = url_features.apply(lambda x: x['ip_in_url'])
    features_df['link_mismatch'] = url_features.apply(lambda x: x['link_mismatch'])
    
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