import pandas as pd
import os

# CONFIGURATION
INPUT_FILE = 'submission.csv'
OUTPUT_FILE = 'titanic_final_submission.csv'

def fix_submission():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Reading {INPUT_FILE}...")
    
    # Check if file has headers by reading first line
    with open(INPUT_FILE, 'r') as f:
        first_line = f.readline()
    
    has_header = "PassengerId" in first_line or "Survived" in first_line
    
    if has_header:
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE, header=None)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['PassengerId', 'Survived']

    # Ensure integer types
    df['PassengerId'] = df['PassengerId'].astype(int)
    df['Survived'] = df['Survived'].apply(lambda x: int(round(float(x))))
    df['Survived'] = df['Survived'].clip(0, 1)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"DONE: Fixed file saved as: {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    fix_submission()
