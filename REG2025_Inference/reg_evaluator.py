import pandas as pd
from eval import REG_Evaluator
import os

def evaluate_csv_direct(csv_path, embedding_model='dmis-lab/biobert-v1.1', spacy_model='en_core_sci_lg'):
    # Load CSV directly
    df = pd.read_csv(csv_path)
    
    eval_pairs = list(zip(df["Ground Truths"], df["Generated Reports"]))
    
    # Run REG evaluation
    evaluator = REG_Evaluator(embedding_model=embedding_model, spacy_model=spacy_model)
    score = evaluator.evaluate_dummy(eval_pairs)
    
    # Save result
    results_df = pd.DataFrame([{
        "Average Ranking Score": score
    }])
    out_path = os.path.join(os.path.dirname(csv_path), "results.csv")
    results_df.to_csv("results.csv", index=False)
    
    print(f"REG Score: {score:.4f} (n={len(eval_pairs)})")
    return score

# Change this path to load the respective gen_vs_gt.csv files 
csv_path = '/home/woody/iwi5/iwi5204h/HistGen/Data/TestResult_HistGen/seed43/17_seed43/Best31/gen_vs_gt.csv'

score = evaluate_csv_direct(csv_path)
