import csv
import google.generativeai as genai  # Gemini API client
import time
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt


# Load API key from environment variable
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env
gemini_free_api_key = os.getenv("GEMINI_FREE_API_KEY")

# Configure Gemini API
genai.configure(api_key=gemini_free_api_key)
model = genai.GenerativeModel('gemini-pro')

def get_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

# Process CSV files
def run_experiment(input_csv, output_csv):
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ['Response'])
        writer.writeheader()

        for row in reader:
            prompt = row['Prompt']
            response = get_gemini_response(prompt)
            row['Response'] = response
            writer.writerow(row)
            time.sleep(1)  # Rate limit avoidance

# Run for both files
run_experiment('google_vs_competitors.csv', 'google_responses.csv')
run_experiment('control_prompts.csv', 'control_responses.csv')