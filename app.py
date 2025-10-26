import streamlit as st
import joblib 
# --- Imports for Email Processing and Feature Extraction ---
import email 
from email.policy import default
import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scipy.sparse import hstack
# -----------------------------------------------------------

# --- 1. YOUR COPIED FUNCTIONS (Feature Extraction from demo.py) ---

# 1. SUSPICIOUS_KEYWORDS 
SUSPICIOUS_KEYWORDS = [
    'urgent', 'security alert', 'ssn', 'credit card', 
    'unusual activity', 'action required', 'suspend', 'locked',
    'bank', 'invoice', 'payment', 'verify your'
]

# 2. check_keywords(email_body)
def check_keywords(email_body):
    """Counts how many suspicious keywords are in the email."""
    count = 0
    email_body_lower = ' ' + str(email_body).lower() + ' '
    
    for keyword in SUSPICIOUS_KEYWORDS:
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
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
        url_count = len(urls)
        for url in urls:
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
                ip_in_url_count += 1

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

# --- 2. THE MODIFIED PARSER FUNCTION ---
def parse_eml(raw_bytes):
    """
    Parses raw .eml bytes (from Streamlit uploader) and extracts headers and body.
    Returns (msg_object, email_body)
    """
    email_body = ""
    
    try:
        # CRITICAL CHANGE: Use message_from_bytes instead of message_from_file
        msg = email.message_from_bytes(raw_bytes, policy=default) 
    except Exception as e:
        st.error(f"Error reading file bytes: {e}")
        return None, None

    # The rest of the logic extracts the body content
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

# --- 3. THE CACHED MODEL LOADER ---
@st.cache_resource
def load_model_assets():
    """Loads the trained ML model and TF-IDF vectorizer."""
    try:
        model = joblib.load('models/phishing_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model files not found in 'models/' folder.")
        st.error("Please run 'main.py' first to train and save the models!")
        return None, None 


# --- 4. MAIN STREAMLIT APPLICATION LOGIC ---
st.title("Phishing Email Detector")
st.write("Upload one or more .eml files to get a Phishing Score and Explanation.")

ml_model, tfidf_vectorizer = load_model_assets()

if ml_model and tfidf_vectorizer:
    st.success("Trained ML Model and Vectorizer loaded successfully!")
    
    # 5. FILE UPLOADER
    uploaded_files = st.file_uploader(
        "Upload one or more .eml files for analysis",
        type=['eml'],
        accept_multiple_files=True
    )
    
    # 6. START PROCESSING LOOP
    if uploaded_files:
        st.info(f"Received {len(uploaded_files)} file(s). Analyzing now...")
        
        for file in uploaded_files:
            
            with st.expander(f"Analysis Report for: {file.name}", expanded=True): # Expanded by default for testing
                st.markdown(f"#### Step 1: Parsing {file.name}...")
                
                # Get the raw bytes from the uploaded file object
                file_bytes = file.getvalue()
                
                # Call our modified parsing function
                msg_obj, body_text = parse_eml(file_bytes)
                
                if body_text is None:
                    continue
                
                if not body_text:
                    st.warning("Could not extract a readable text or HTML body from this email.")
                    continue

                st.success("Step 1 Complete: Email body successfully extracted!")
                st.markdown("---")
                st.markdown("##### Extracted Email Snippet:")
                # Show the first 500 characters of the body
                st.code(body_text[:500] + "...", language='text')

                # --- STEP 2: Feature Extraction ---
                st.markdown("---")
                st.markdown("#### Step 2: Extracting Rule-Based Features...")
                
                # 1. Extract Keyword Features
                keyword_score = check_keywords(body_text)
                
                # 2. Extract URL Features
                url_features = extract_url_features(body_text)
                
                # 3. Combine all features into one dictionary
                all_rule_features = {
                    'keyword_count': keyword_score,
                    'url_count': url_features['url_count'],
                    'ip_in_url': url_features['ip_in_url'],
                    'link_mismatch': url_features['link_mismatch']
                }

                # 4. Calculate the Rule-Based Score and Explanation
                rule_score, explanation = get_rule_based_score(all_rule_features)
                
                # --- Step 3: Display Results ---
                st.success("Step 2 Complete: Rule-Based Features Extracted!")
                st.markdown("---")
                st.markdown(f"#### Rule-Based Score: **{rule_score}**")
                
                st.markdown("##### Detailed Rule Features:")
                st.json(all_rule_features) # st.json displays a dictionary cleanly
                
                st.warning("Analysis paused here! If the feature extraction above looks correct, we'll implement the ML prediction next.")
                
                # --- STEP 3: ML Model Prediction ---
                st.markdown("---")
                st.markdown("#### Step 3: ML Model Prediction...")
                
                # 1. Prepare data for the ML model (TF-IDF)
                body_tfidf = tfidf_vectorizer.transform([body_text])
                
                # 2. Prepare the Rule-Based features for the ML model (NumPy Array)
                rule_vector = np.array([[
                    all_rule_features['keyword_count'],
                    all_rule_features['url_count'],
                    all_rule_features['ip_in_url'],
                    all_rule_features['link_mismatch']
                ]])
                
                # 3. Combine features
                combined_features_vector = hstack([body_tfidf, rule_vector])

                # 4. Get ML Model Prediction
                # [0][1] gives the probability for class 1 (phishing)
                ml_prob = ml_model.predict_proba(combined_features_vector)[0][1]
                ml_prediction = ml_model.predict(combined_features_vector)[0]
                
                # --- Step 4: Show Final Report ---
                CONFIDENCE_THRESHOLD = 0.70  # The threshold set in your demo.py
                
                final_decision_text = "🚨 PHISHING 🚨" if ml_prob >= CONFIDENCE_THRESHOLD else "✅ BENIGN ✅"
                
                if ml_prob >= CONFIDENCE_THRESHOLD:
                    st.error(f"FINAL DECISION: {final_decision_text}")
                else:
                    st.success(f"FINAL DECISION: {final_decision_text}")
                    
                st.markdown(f"**ML Model Score (Probability of Phishing):** `{ml_prob * 100:.2f}%`")
                st.markdown(f"**Rule-Based Score:** `{rule_score}`")
                
                st.markdown("---")
                st.markdown("##### Detailed Explanation (Rules Triggered):")
                if not explanation:
                    st.write("No suspicious rules triggered.")
                else:
                    for line in explanation:
                        st.write(f"- {line}")
                        
                st.info("Analysis complete for this file!")
else:
    pass