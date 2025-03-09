import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss 
import re
import json
from openai import OpenAI
from datasets import load_dataset
import random
import os
from dotenv import load_dotenv
load_dotenv()

# Load the dataset
ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
ds_eval = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

# Filter out queries that do NOT contain "and" or "or" (to get non-branching ones)
def filter_non_branching_queries(dataset):
    return [example for example in dataset if not ("and" in example['question'].lower() or "or" in example['question'].lower())]

non_branching_queries = filter_non_branching_queries(ds_eval['test'])

# Combine two queries with "and" or "or" and pair their answers
def combine_queries_with_answers(dataset):
    combined_queries_with_answers = []
    random.shuffle(dataset)
    for i in range(0, len(dataset) - 1, 2):
        query1 = dataset[i]['question']
        answer1 = dataset[i]['answer']
        query2 = dataset[i+1]['question']
        answer2 = dataset[i+1]['answer']

        conjunction = random.choice(["and", "or"])
        combined_query = f"{query1} {conjunction} {query2}"
        combined_answers = f"Answer 1: {answer1}\nAnswer 2: {answer2}"

        combined_queries_with_answers.append({
            "combined_query": combined_query,
            "subquery_1": query1,
            "subquery_2": query2,
            "combined_answers": combined_answers,
            "ground_truth1": answer1,  # Separate ground truth for subquery 1
            "ground_truth2": answer2,  # Separate ground truth for subquery 2
        })
    return combined_queries_with_answers

combined_queries = combine_queries_with_answers(non_branching_queries)
print(combined_queries[0])

# Extracting passages and ids
documents = ds["passages"]["passage"]
ids = ds["passages"]["id"]

# Create and store knowledge base
def create_knowledge_base(documents, ids, index_path, metadata_path, embedding_model):
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)
    metadata = [{"id": ids[i], "text": documents[i]} for i in range(len(documents))]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print("Knowledge Base Created & Stored Successfully!")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create and store knowledge base
create_knowledge_base(documents, ids, "knowledge_base.index", "kb_metadata.json", embedding_model)

# Load FAISS Knowledge Base
class DocumentRetriever:
    def __init__(self, faiss_index_path, metadata_path, embedding_model):
        self.embedding_model = embedding_model
        self.index = faiss.read_index(faiss_index_path)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def retrieve(self, query, top_k=10):
        query_embedding = self.embedding_model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, k=top_k)
        relevant_docs = [self.metadata[i] for i in I[0] if i != -1]
        return relevant_docs

# Generate responses with DeepSeek
def generate_with_deepseek(subquery1, subquery2, context_subquery1, context_subquery2, doc_ids_subquery1, doc_ids_subquery2):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    # Format the prompt to include the context for both subqueries and their associated document texts and IDs
    prompt = f"""
    You are a helpful assistant. Answer each subquery separately using ONLY the provided context. Use the following format for your response:

    Answer 1: [Your answer to Subquery 1]
    Answer 2: [Your answer to Subquery 2]

    Do not deviate from this format. Do not include any additional text or explanations.

    Context for Subquery 1 (Source documents: {', '.join([f'ID {doc_id}: {doc_text}...' for doc_id, doc_text in zip(doc_ids_subquery1, context_subquery1)])}):
    {context_subquery1}
    Subquery 1: {subquery1}

    Context for Subquery 2 (Source documents: {', '.join([f'ID {doc_id}: {doc_text}...' for doc_id, doc_text in zip(doc_ids_subquery2, context_subquery2)])}):
    {context_subquery2}
    Subquery 2: {subquery2}
    """

    # Call the DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer subqueries separately with clear distinctions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0,
        stream=False
    )

    # Return the generated response
    return response.choices[0].message.content

# Function to extract answers from LLM response
def extract_answers(llm_response):
    # Use regex to find answers for Answer 1 and Answer 2, even if there are extra characters like **
    matches = re.findall(r"Answer\s\d+:?\s*([^\n]+)", llm_response, re.DOTALL)
    
    # Ensure there are exactly two matches, otherwise return empty answers
    return matches[:2] if len(matches) >= 2 else ["", ""]  # Extract only the first two answers

# Branching RAG Pipeline
class BranchingRAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, combined_query, subquery1, subquery2, i):
        # Retrieve documents for both subqueries independently
        retrieved_docs_subquery1 = self.retriever.retrieve(subquery1)
        retrieved_docs_subquery2 = self.retriever.retrieve(subquery2)

        if not retrieved_docs_subquery1 or not retrieved_docs_subquery2:
            return pd.DataFrame([{
                'query': combined_query,
                'subquery_1': subquery1,
                'subquery_2': subquery2,
                'llm_response': 'No relevant documents found.',
                'llm_response_sub_query1': '',
                'llm_response_sub_query2': '',
                'sources_1': [],
                'sources_2': [],
                'ground_truth1': combined_queries[i]['ground_truth1'],
                'ground_truth2': combined_queries[i]['ground_truth2'],
            }])

        # Extract context and doc IDs for each subquery
        context_subquery1 = [doc["text"] for doc in retrieved_docs_subquery1]
        context_subquery2 = [doc["text"] for doc in retrieved_docs_subquery2]
        doc_ids_subquery1 = [doc["id"] for doc in retrieved_docs_subquery1]
        doc_ids_subquery2 = [doc["id"] for doc in retrieved_docs_subquery2]

        # Now generate the answers using the context for each subquery
        response = generate_with_deepseek(subquery1, subquery2, "\n".join(context_subquery1), "\n".join(context_subquery2),
                                          doc_ids_subquery1, doc_ids_subquery2)

        # Extract answers for each subquery
        answer1, answer2 = extract_answers(response)

        # Return the DataFrame with updated fields
        return pd.DataFrame([{
            'query': combined_query,
            'subquery_1': subquery1,
            'subquery_2': subquery2,
            'llm_response': response,
            'llm_response_sub_query1': answer1,  # Extracted Answer 1
            'llm_response_sub_query2': answer2,  # Extracted Answer 2
            'truth_answers': combined_queries[i]['combined_answers'],
            'sources_1': [str(doc_id) for doc_id in doc_ids_subquery1],
            'sources_2': [str(doc_id) for doc_id in doc_ids_subquery2],
            'ground_truth1': combined_queries[i]['ground_truth1'],
            'ground_truth2': combined_queries[i]['ground_truth2'],
        }])

# Initialize pipeline
retriever = DocumentRetriever("knowledge_base.index", "kb_metadata.json", embedding_model)
branching_rag_pipeline = BranchingRAGPipeline(retriever)

# Process the first 5 queries
data = []
for i, query_data in enumerate(combined_queries):  # Only the first 5 queries
    print(f"Processing query {i+1}/{len(combined_queries)}")
    
    combined_query = query_data['combined_query']
    subquery1 = query_data['subquery_1']
    subquery2 = query_data['subquery_2']
    
    df = branching_rag_pipeline.run(combined_query, subquery1, subquery2, i)
    data.append(df)
    

# Concatenate all DataFrames
final_df = pd.concat(data, ignore_index=True)

# Save results
final_df.to_csv("rag_output.csv", index=False)
print("Data saved to rag_output.csv")

# Display first 5 rows
print(final_df.head())
