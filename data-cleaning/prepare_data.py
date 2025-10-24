import pandas as pd
import glob
import os

# --- Data Cleaning and Combining Script ---

# Find all CSV files in the current directory.
print("Starting to search for CSV files...")
all_files = glob.glob('*.csv')

if not all_files:
    print("Error: No CSV files found in this folder.")
    print("Please make sure 'prepare_data.py' is in the SAME folder as your dataset CSVs.")
else:
    print(f"Found {len(all_files)} CSV files to process...")

all_dataframes = []

for filename in all_files:
    # Let's not re-process our own output file if we run this twice
    if filename == 'master_phishing_dataset.csv':
        continue
        
    print(f"--- Processing: {filename} ---")
    
    try:
        # Load the CSV file
        df = pd.read_csv(filename)
        
        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # We absolutely need a 'label' column
        if 'label' not in df.columns:
            print(f"Skipping '{filename}': No 'label' column found.")
            continue

        # Case 1: 'text' column already exists (like the all-phishing file)
        if 'text' in df.columns:
            # Just keep the 'text' and 'label' columns
            clean_df = df[['text', 'label']]
            
        # Case 2: 'subject' and 'body' columns exist (like the other files)
        elif 'subject' in df.columns and 'body' in df.columns:
            print(f"Found 'subject' and 'body'. Combining them into 'text'...")
            
            # Combine subject and body. 
            # .fillna('') handles any missing (NaN) values so we don't get errors
            df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
            
            # Just keep the new 'text' and 'label' columns
            clean_df = df[['text', 'label']]
            
        else:
            print(f"Skipping '{filename}': Could not find 'text' or ('subject' and 'body') columns.")
            continue
            
        # Add the cleaned data to our list
        all_dataframes.append(clean_df)
        print(f"Added {len(clean_df)} rows from '{filename}'.")

    except pd.errors.EmptyDataError:
        print(f"Skipping '{filename}': File is empty.")
    except Exception as e:
        print(f"Error processing '{filename}': {e}")

# --- Final Step: Combine all DataFrames ---

if all_dataframes:
    # Concatenate all the individual DataFrames into one big one
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Clean up any rows where the 'text' ended up being empty
    master_df = master_df.dropna(subset=['text'])
    
    # Save the master dataset to a new CSV
    master_df.to_csv('master_phishing_dataset.csv', index=False)
    
    print("\n--- Processing Complete ---")
    print(f"Successfully created 'master_phishing_dataset.csv'")
    
    # Show the results
    print("\n--- Master Dataset Info ---")
    master_df.info()
    
    print("\n--- Label Distribution ---")
    # This is important! Let's see the balance of 0s and 1s
    print(master_df['label'].value_counts())

else:
    print("\nNo data was processed. Please check your CSV files and column names.")