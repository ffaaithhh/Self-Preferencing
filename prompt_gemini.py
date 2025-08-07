import csv
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from environment variable
load_dotenv()  # Loads variables from .env
gemini_free_api_key = os.getenv("GEMINI_IMDA_API_KEY")

# Configure Gemini API
genai.configure(api_key=gemini_free_api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_experiment(input_csv, output_csv):
    # Define the exact output fieldnames we want
    output_fieldnames = ['Category', 'Prompt', 'Google', 'Competitor', 'Response']
    
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # Verify required columns exist in input
        required_columns = {'Category', 'Prompt', 'Google', 'Competitor'}
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        
        # Create writer with our desired output format
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            # Create new row with our desired structure
            new_row = {
                'Category': row['Category'],
                'Prompt': row['Prompt'],
                'Google': row['Google'],  # Maps Product1 to Google
                'Competitor': row['Competitor'],  # Maps Product2 to Competitor
                'Response': get_gemini_response(row['Prompt'])
            }
            
            writer.writerow(new_row)
            print(f"Processed {i+1} rows")
            time.sleep(1)  # Rate limit avoidance

# Run the experiment
run_experiment('small_google_prompt.csv', 'small_google_responses.csv')