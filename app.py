
import streamlit as st
import joblib 
import email 
from email.policy import default
import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scipy.sparse import hstack

st.set_page_config(layout="wide")


# --- 1. FEATURE EXTRACTION HELPERS ---
SUSPICIOUS_KEYWORDS = [
    'urgent', 'security alert', 'ssn', 'credit card', 
    'unusual activity', 'action required', 'suspend', 'locked',
    'bank', 'invoice', 'payment', 'verify your'
]
def check_keywords(email_body):
    count = 0
    email_body_lower = ' ' + str(email_body).lower() + ' '
    for keyword in SUSPICIOUS_KEYWORDS:
        if ' ' + keyword + ' ' in email_body_lower:
            count += 1
    return count
def extract_url_features(email_body):
    email_body = str(email_body) 
    url_count = 0
    ip_in_url_count = 0
    mismatch_count = 0
    if '<html>' in email_body.lower():
        soup = BeautifulSoup(email_body, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href']
            if not href: continue
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
            except Exception: pass
    else:
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
        url_count = len(urls)
        for url in urls:
            if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url):
                ip_in_url_count += 1
    return {'url_count': url_count, 'ip_in_url': 1 if ip_in_url_count > 0 else 0, 'link_mismatch': 1 if mismatch_count > 0 else 0}
def get_rule_based_score(features):
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

# --- 2. EMAIL PARSING HELPER ---
# (This function stays exactly the same)
def parse_eml(raw_bytes):
    email_body = ""
    try:
        msg = email.message_from_bytes(raw_bytes, policy=default) 
    except Exception as e:
        st.error(f"Error reading file bytes: {e}")
        return None, None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype in ['text/plain', 'text/html'] and 'attachment' not in cdispo:
                try:
                    email_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                except Exception: continue
                if ctype == 'text/html': break
    else:
        try:
            email_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except Exception: email_body = ""
    return msg, email_body

# --- 3. UI HELPER FUNCTIONS ---
# (These two functions stay exactly the same)
def analyze_email_file(file, ml_model, tfidf_vectorizer):
    file_bytes = file.getvalue()
    msg_obj, body_text = parse_eml(file_bytes)
    if not body_text:
        return {'filename': file.name, 'status': 'Error', 'error_message': 'Could not extract a readable text or HTML body.'}
    keyword_score = check_keywords(body_text)
    url_features = extract_url_features(body_text)
    all_rule_features = {'keyword_count': keyword_score, 'url_count': url_features['url_count'], 'ip_in_url': url_features['ip_in_url'], 'link_mismatch': url_features['link_mismatch']}
    rule_score, explanation = get_rule_based_score(all_rule_features)
    body_tfidf = tfidf_vectorizer.transform([body_text])
    rule_vector = np.array([[all_rule_features['keyword_count'], all_rule_features['url_count'], all_rule_features['ip_in_url'], all_rule_features['link_mismatch']]])
    combined_features_vector = hstack([body_tfidf, rule_vector])
    ml_prob = ml_model.predict_proba(combined_features_vector)[0][1]
    return {'filename': file.name, 'status': 'Success', 'body_snippet': body_text[:500] + "...", 'all_rule_features': all_rule_features, 'rule_score': rule_score, 'explanation': explanation, 'ml_prob': ml_prob}

def display_report(result):
    if result['status'] == 'Error':
        st.error(f"Error analyzing {result['filename']}: {result['error_message']}")
        return
    # We remove the st.subheader, as the column already has a header
    st.markdown("---")
    st.markdown("##### Extracted Email Snippet:")
    st.code(result['body_snippet'], language='text')
    st.markdown("---")
    st.markdown("##### Rule-Based Features:")
    st.json(result['all_rule_features'])
    st.markdown("---")
    st.markdown("#### Final Decision & Scores")
    CONFIDENCE_THRESHOLD = 0.70
    ml_prob = result['ml_prob']
    rule_score = result['rule_score']
    explanation = result['explanation']
    final_decision_text = "🚨 PHISHING 🚨" if ml_prob >= CONFIDENCE_THRESHOLD else "✅ BENIGN ✅"
    if ml_prob >= CONFIDENCE_THRESHOLD: st.error(f"FINAL DECISION: {final_decision_text}")
    else: st.success(f"FINAL DECISION: {final_decision_text}")
    col1, col2 = st.columns(2)
    col1.metric("ML Model Score", f"{ml_prob * 100:.2f}%")
    col2.metric("Rule-Based Score", f"{rule_score}")
    st.markdown("---")
    st.markdown("##### Detailed Explanation (Rules Triggered):")
    if not explanation: st.write("No suspicious rules triggered.")
    else:
        for line in explanation: st.write(f"- {line}")

# --- 4. MODEL LOADER (CACHED) ---
# (This function stays exactly the same)
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('models/phishing_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model files not found in 'models/' folder.")
        st.error("Please run 'main.py' first to train and save the models!")
        return None, None 

# --- 5. MAIN STREAMLIT APPLICATION ---
st.title("Phishing Email Detector")

ml_model, tfidf_vectorizer = load_model_assets()

if ml_model and tfidf_vectorizer:
    
    # --- MODIFICATION 2: Create the two columns ---
    # We make the report column 2x as wide as the control column
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        uploaded_files = st.file_uploader(
            "Upload one or more .eml files for analysis",
            type=['eml'],
            accept_multiple_files=True,
            label_visibility="collapsed" # Hides the label, "Controls" is enough
        )

    with col2:
        st.subheader("Analysis Report")
        
        if uploaded_files:
            # --- Analysis loop (runs first) ---
            all_results = []
            with st.spinner(f"Analyzing {len(uploaded_files)} file(s)..."):
                for file in uploaded_files:
                    result = analyze_email_file(file, ml_model, tfidf_vectorizer)
                    all_results.append(result)
            
            # --- MODIFICATION 3: Move radio buttons into col1 ---
            # We re-open col1 to add the selector *after* analysis
            with col1:
                st.markdown("---")
                st.markdown("##### Select a report to view:")
                
                filenames = []
                for res in all_results:
                    icon = "🚨" if res.get('ml_prob', 0) >= 0.70 else "✅"
                    filenames.append(f"{icon} {res['filename']}")
                
                selected_filename_with_icon = st.radio(
                    "Analyzed Files",
                    filenames,
                    label_visibility="collapsed" # Hides the "Analyzed Files" label
                )

            # --- Report Display (inside col2) ---
            selected_filename = selected_filename_with_icon[2:]
            selected_result = None
            for res in all_results:
                if res['filename'] == selected_filename:
                    selected_result = res
                    break
            
            if selected_result:
                display_report(selected_result)
        
        else:
            # This shows in col2 before anything is uploaded
            st.info("Upload one or more files in the left panel to begin analysis.")
else:
    st.warning("Please wait for models to load or check errors above.")