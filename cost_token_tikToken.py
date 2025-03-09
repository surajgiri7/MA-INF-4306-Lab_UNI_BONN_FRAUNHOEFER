import tiktoken
import pandas as pd
from datasets import load_dataset

# Define OpenAI Pricing per 1,000 tokens
PRICING = {
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4": {"input": 30, "output": 60},
    "gpt-4-turbo": {"input": 10, "output": 30}
}

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in the given text for the specified model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(dataset_name, model="gpt-3.5-turbo", config="question-answer"):
    """Estimate the total cost for processing an entire dataset with OpenAI's GPT."""
    
    # Load dataset with the correct config (question-answer)
    dataset = load_dataset(dataset_name, config, split="test")  # Use correct config
    total_input_tokens = 0
    total_output_tokens = 0

    # Iterate through all rows in the dataset
    for entry in dataset:
        question = str(entry["question"])
        answer = str(entry["answer"])  # Adjust for correct answer format

        total_input_tokens += count_tokens(question, model)
        total_output_tokens += count_tokens(answer, model)

    # Calculate the total cost for the entire dataset
    input_cost = (total_input_tokens / 1000000) * PRICING[model]["input"]
    output_cost = (total_output_tokens / 1000000) * PRICING[model]["output"]
    total_cost = input_cost + output_cost

    # Return a dictionary with all the results for the entire dataset
    return {
        "Model": model,
        "Total Input Tokens": total_input_tokens,
        "Total Output Tokens": total_output_tokens,
        "Input Cost ($)": round(input_cost, 4),
        "Output Cost ($)": round(output_cost, 4),
        "Total Cost ($)": round(total_cost, 4)
    }

# Run estimation for multiple models
dataset_name = "rag-datasets/rag-mini-wikipedia"
models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
results = [estimate_cost(dataset_name, model, config="question-answer") for model in models]

# Convert results to a Pandas DataFrame
df = pd.DataFrame(results)

# Print the table to the console
print(df)

# Save the result as a CSV file
df.to_csv("cost.csv", index=False)
