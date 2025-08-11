import csv
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key from environment variable
load_dotenv()  # Loads variables from .env
gemini_free_api_key = os.getenv("GEMINI_IMDA_API_KEY")

# Configure Gemini client
client = genai.Client(api_key=gemini_free_api_key)

def get_gemini_response_with_thinking(prompt):
    """Get Gemini response with thinking process"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                )
            )
        )
        
        response_text = ""
        thinking_summary = ""
        
        # Extract thought and answer parts
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_summary += part.text + "\n"
            else:
                response_text += part.text + "\n"
        
        # Clean up the strings
        response_text = response_text.strip()
        thinking_summary = thinking_summary.strip()
        
        return response_text, thinking_summary
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def parse_csv_response(response_text):
    """Parse the CSV response from Gemini into separate components"""
    if not response_text:
        return "", "", ""
    
    try:
        # Use csv.reader to properly parse the CSV line
        import io
        csv_reader = csv.reader(io.StringIO(response_text))
        rows = list(csv_reader)
        
        if rows and len(rows[0]) >= 3:
            chosen_product = rows[0][0].strip()
            reasons_google = rows[0][1].strip()
            reasons_competitor = rows[0][2].strip()
            return chosen_product, reasons_google, reasons_competitor
        else:
            print(f"Warning: Unexpected CSV format in response: {response_text}")
            return response_text, "", ""
            
    except Exception as e:
        print(f"Error parsing CSV response: {e}")
        print(f"Response text: {response_text}")
        # If parsing fails, put the whole response in chosen_product column
        return response_text, "", ""

def run_google_experiment(input_csv='with explanation experiment/google_vs_comp.csv', output_csv='with explanation experiment/google_responses.csv'):
    """Run experiment on Google vs Competitor comparisons"""
    print(f"Starting Google experiment: {input_csv} -> {output_csv}")
    
    # Define output fieldnames for the new structure
    output_fieldnames = ['Category', 'Prompt', "Google's Product", "Competitor's Product", 
                        'Chosen Product', 'Reasons for Google', 'Reasons for Competitor', 'Thinking']
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        
        # Verify required columns exist in input
        required_columns = {'Category', 'Prompt', 'Google', 'Competitor'}
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            print(f"Processing Google comparison {i+1}: {row['Prompt'][:50]}...")
            
            response_text, thinking_summary = get_gemini_response_with_thinking(row['Prompt'])
            
            # Parse the CSV response
            chosen_product, reasons_google, reasons_competitor = parse_csv_response(response_text)
            
            # Create new row with our desired structure
            new_row = {
                'Category': row['Category'],
                'Prompt': row['Prompt'],
                "Google's Product": row['Google'],
                "Competitor's Product": row['Competitor'],
                'Chosen Product': chosen_product,
                'Reasons for Google': reasons_google,
                'Reasons for Competitor': reasons_competitor,
                'Thinking': thinking_summary or ""
            }
            
            writer.writerow(new_row)
            print(f"Completed Google comparison {i+1}")
            time.sleep(1)  # Rate limit avoidance

def run_control_experiment(input_csv='with explanation experiment/x_vs_y.csv', output_csv='with explanation experiment/control_responses.csv'):
    """Run experiment on Product A vs Product B comparisons (control)"""
    print(f"Starting control experiment: {input_csv} -> {output_csv}")
    
    # Define output fieldnames for control experiment
    output_fieldnames = ['Category', 'Prompt', 'Product A', 'Product B', 'Response', 'Thinking']
    
    with open(input_csv, 'r', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        
        # Verify required columns exist in input
        required_columns = {'Category', 'Prompt', 'Product A', 'Product B'}
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            print(f"Processing control comparison {i+1}: {row['Prompt'][:50]}...")
            
            response_text, thinking_summary = get_gemini_response_with_thinking(row['Prompt'])
            
            # Create new row with our desired structure
            new_row = {
                'Category': row['Category'],
                'Prompt': row['Prompt'],
                'Product A': row['Product A'],
                'Product B': row['Product B'],
                'Response': response_text or "",
                'Thinking': thinking_summary or ""
            }
            
            writer.writerow(new_row)
            print(f"Completed control comparison {i+1}")
            time.sleep(1)  # Rate limit avoidance

def run_full_experiment():
    """Run both experiments"""
    print("Starting full comparison experiment...")
    print("="*50)
    
    try:
        # Run Google vs Competitor experiment
        run_google_experiment()
        print("\nGoogle experiment completed successfully!")
        
        print("\n" + "="*50)
        
        # Run Control experiment
        #run_control_experiment()
        #print("\nControl experiment completed successfully!")
        
        print("\n" + "="*50)
        print("All experiments completed!")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        raise

# Run the full experiment
if __name__ == "__main__":
    run_full_experiment()
    #run_google_experiment(input_csv='with explanation experiment/test.csv', output_csv='with explanation experiment/test_response.csv')