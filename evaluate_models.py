import os
import random
import dotenv
import json
from openai import OpenAI
from utils.qa_benchmark.evaluator import BenchmarkRunner, read_json
from utils.qa_benchmark.create_qas import load_qa_items
# Example usage:
if __name__ == "__main__":
    # Load environment variables.
    dotenv.load_dotenv()

    # Define a list of QA items for evaluation.
    qa_items = load_qa_items("./chemrxiv_qas_v2_3_verified.json")
    # qa_items = random.sample(qa_items, 100)
    RECORDS_PATH = "./chemrxiv_qas_v2_final.json"
    RESPONSES_DIR = "responses"  # Directory to save intermediate responses
    RESULT_PATH = "results.csv"  # Save the final results as a CSV file
    with open(RECORDS_PATH, "w") as outfile:
        json.dump(qa_items, outfile)
    
    

   
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    records = read_json(RECORDS_PATH)

    benchmark_runner = BenchmarkRunner(
        records=records, responses_dir=RESPONSES_DIR, results_file=RESULT_PATH
    )

    benchmark_runner.run_all_benchmarks(skip_completed=True)