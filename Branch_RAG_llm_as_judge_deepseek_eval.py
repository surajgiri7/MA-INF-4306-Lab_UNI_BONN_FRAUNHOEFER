import pandas as pd
import re
from openai import OpenAI
from tqdm import tqdm  # For progress tracking
import os
from dotenv import load_dotenv
load_dotenv()

# Load CSV Data
csv_file_path = "updated_data.csv" 
data_df = pd.read_csv(csv_file_path)

# Verify Required Columns
required_columns = {"query", "subquery_1", "subquery_2", "llm_response_sub_query1", "llm_response_sub_query2", "ground_truth1", "ground_truth2", "sources_1", "sources_2", "source_text1", "source_text2"}
if not required_columns.issubset(data_df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Initialize DeepSeek API Client
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Function to Evaluate Using LLM-as-a-Judge
def evaluate_with_llm_judge(subquery, context, llm_response, truth_answer):
    prompt = f"""
    # Evaluation Task
    Assess the response based on five key criteria. Assign a **score (1-5)** for each, with a brief explanation.
    
    ## Scoring Criteria:
    1. **Relevance (Similarity to Expected Answer)**  
       - Does the response correctly answer the question?  
       - Does it mean the same as the truth answer, even if phrased differently?  
       **Score (1-5) + Explanation**  

    2. **Accuracy (Faithfulness to the Source)**  
       - Is the response factually correct based on the given source text?  
       - Does it avoid adding incorrect details?  
       **Score (1-5) + Explanation**  

    3. **Context Alignment**  
       - Does the response focus on the most relevant parts of the source?  
       - Does it stay on topic without unnecessary details?  
       **Score (1-5) + Explanation**  

    4. **Clarity & Precision**  
       - Is the response clear, concise, and well-structured?  
       - Does it avoid vague or off-topic information?  
       **Score (1-5) + Explanation**  

    5. **Completeness (Recall of Important Information)**  
       - Does the response capture all key details needed to answer the question?  
       - Is anything essential missing?  
       **Score (1-5) + Explanation**  

    ## Review:
    - **Subquery:** {subquery}  
    - **Source Text:** {context}  
    - **Truth Answer:** {truth_answer}  
    - **LLM Response:** {llm_response} 
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a strict but fair AI evaluator assessing the quality of responses based on the provided criteria."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Increased to allow for detailed explanations
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return None

# Function to Parse Scores from LLM Response
def extract_scores(evaluation):
    # Initialize the dictionary with default scores
    scores = {
        'answer relevancy': 0,
        'faithfulness': 0,
        'contextual relevancy': 0,
        'contextual precision': 0,
        'contextual recall': 0
    }
    
    # Use a regular expression to find all the scores in the format "Score: <number>"
    score_matches = re.findall(r"Score:\s*(\d+)", evaluation)
    
    # Map scores to corresponding categories in order of appearance
    categories = [
        'answer relevancy', 
        'faithfulness', 
        'contextual relevancy', 
        'contextual precision', 
        'contextual recall'
    ]
    
    # Assign the extracted scores to the dictionary
    for i, score in enumerate(score_matches):
        if i < len(categories):  # Ensure we don't go out of bounds
            scores[categories[i]] = int(score)
    
    return scores

# Initialize Storage for Scores
all_scores = {"answer relevancy": [], "faithfulness": [], "contextual relevancy": [], "contextual precision": [], "contextual recall": []}

# Process First 5 Rows to Save API Costs
evaluation_results = []
for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="🔍 Evaluating Responses", unit="row"):
    query = row["query"]
    subquery1, subquery2 = row["subquery_1"], row["subquery_2"]
    source_text1, source_text2 = row["source_text1"], row["source_text2"]
    truth_answer1, truth_answer2 = row["ground_truth1"], row["ground_truth2"]
    llm_response1, llm_response2 = row["llm_response_sub_query1"], row["llm_response_sub_query2"]
    
    # Evaluate Subquery 1
    eval1 = evaluate_with_llm_judge(subquery1, source_text1, llm_response1, truth_answer1)
    scores1 = extract_scores(eval1)  
    print("eval 1")
    print(eval1) 
    print(scores1)

    eval2 = evaluate_with_llm_judge(subquery2, source_text2, llm_response2, truth_answer2)
    scores2 = extract_scores(eval2) 
    print("eval 2") 
    print(eval2) 
    print(scores2)
    
    # Store Scores
    row_scores = {"query": query, "subquery1": subquery1, "subquery2": subquery2}
    for key in all_scores.keys():
        avg_score = (scores1[key] + scores2[key]) / 2  # Average of both subqueries
        row_scores[key] = avg_score
        all_scores[key].append(avg_score)
    
    evaluation_results.append(row_scores)

# Save Results to CSV
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv("evaluation_results.csv", index=False)
print("Evaluation results saved to evaluation_results.csv")

# Compute and Print Average Scores
average_scores = {criterion: sum(scores) / len(scores) for criterion, scores in all_scores.items() if scores}
print("\nOverall Evaluation Metrics:")
for criterion, score in average_scores.items():
    print(f"{criterion.capitalize()}: {score:.2f}")

# Compute Overall Average Score
overall_avg_score = sum(average_scores.values()) / len(average_scores)
print(f"\nOverall Average Score: {overall_avg_score:.2f}")