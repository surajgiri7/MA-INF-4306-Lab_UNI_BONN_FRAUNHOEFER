import pandas as pd
import re
from openai import OpenAI
from tqdm import tqdm  # For progress tracking
import os
from dotenv import load_dotenv
load_dotenv()

# Load CSV Data
csv_file_path = "eval_data_with_responses.csv"  # Ensure this file exists
data_df = pd.read_csv(csv_file_path)

# Verify Required Columns
required_columns = {"question", "answer", "id", "generated_response", "retrieved_context"}
if not required_columns.issubset(data_df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Initialize DeepSeek API Client
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Function to Evaluate Using LLM-as-a-Judge
def evaluate_with_llm_judge(question, context, llm_response, truth_answer):
    prompt = f"""
    # Evaluation Task
    Assess the response based on five key criteria. Assign a *score (1-5)* for each, with a brief explanation.
    
    ## Scoring Criteria:
    1. *Relevance (Similarity to Expected Answer)*  
       - Does the response correctly answer the question?  
       - Does it mean the same as the truth answer, even if phrased differently?  
       *Score (1-5) + Explanation*  

    2. *Accuracy (Faithfulness to the Source)*  
       - Is the response factually correct based on the given source text?  
       - Does it avoid adding incorrect details?  
       *Score (1-5) + Explanation*  

    3. *Context Alignment*  
       - Does the response focus on the most relevant parts of the source?  
       - Does it stay on topic without unnecessary details?  
       *Score (1-5) + Explanation*  

    4. *Clarity & Precision*  
       - Is the response clear, concise, and well-structured?  
       - Does it avoid vague or off-topic information?  
       *Score (1-5) + Explanation*  

    5. *Completeness (Recall of Important Information)*  
       - Does the response capture all key details needed to answer the question?  
       - Is anything essential missing?  
       *Score (1-5) + Explanation*  

    ## Review:
    - *Question:* {question}  
    - *Retrieved Context:* {context}  
    - *Truth Answer:* {truth_answer}  
    - *LLM Response:* {llm_response} 
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a strict but fair AI evaluator assessing the quality of responses based on the provided criteria."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return None

# Function to Parse Scores from LLM Response
def extract_scores(evaluation):
    scores = {"answer relevancy": 0, "faithfulness": 0, "contextual relevancy": 0, "contextual precision": 0, "contextual recall": 0}
    score_matches = re.findall(r"Score:\s*(\d+)", evaluation)
    categories = list(scores.keys())
    for i, score in enumerate(score_matches):
        if i < len(categories):
            scores[categories[i]] = int(score)
    return scores

# Initialize Storage for Scores
all_scores = {"answer relevancy": [], "faithfulness": [], "contextual relevancy": [], "contextual precision": [], "contextual recall": []}

evaluation_results = []
for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="🔍 Evaluating Responses", unit="row"):
    question = row["question"]
    retrieved_context = row["retrieved_context"]
    truth_answer = row["answer"]
    llm_response = row["generated_response"]
    
    eval_result = evaluate_with_llm_judge(question, retrieved_context, llm_response, truth_answer)
    scores = extract_scores(eval_result)  
    print("Evaluation Result:")
    print(eval_result) 
    print(scores)
    
    row_scores = {"question": question}
    for key in all_scores.keys():
        row_scores[key] = scores[key]
        all_scores[key].append(scores[key])
    
    evaluation_results.append(row_scores)

# Save Results to CSV
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv("naive_rag_evaluation_results.csv", index=False)
print("Evaluation results saved to naive_rag_evaluation_results.csv")

# Compute and Print Average Scores
average_scores = {criterion: sum(scores) / len(scores) for criterion, scores in all_scores.items() if scores}
print("\nOverall Evaluation Metrics:")
for criterion, score in average_scores.items():
    print(f"{criterion.capitalize()}: {score:.2f}")

# Compute Overall Average Score
overall_avg_score = sum(average_scores.values()) / len(average_scores)
print(f"\nOverall Average Score: {overall_avg_score:.2f}")
