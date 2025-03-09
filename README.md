# Lab_Fraunhoefer_UNI_Bonn
RAG.ipynb - Naive RAG with the ChatGPT model
Naive_RAG_deepseek_Final.ipynb - Naive RAG with the Deepseek model
Agentic_and_Vanilla_RAG_eval.ipynb - Agentic and Vanilla RAG with the ChatGPT model

Wikipedia dataset
text-corpus - knowledge base
question-answering - evaluation

For the Naive RAG, the main RAG code is in the notebook: Naive_RAG_deepseek_Final_Eval.ipynb. For the evaluation of Naive RAG, the notebook generates the csv file with the responses for the evaluation dataset, then that csv file is used in the python script Naive_RAG_deepseek_Final_Eval.py. The file Naive_RAG_eval_OpenAI.ipynb is the file for evaluation of the naive RAG with the OpenAI model as the LLM model.

For the Branch RAG, the main Branch RAG code is in the file Branch_RAG_mit_deepseek.py for the RAG with the LLM model Deepseek and in the file Branch_RAG_mit_OpenAI.py for the RAG with the LLM model ChatGPT. the evaluation for each of the versions is done in the files Branch_RAG_llm_as_judge_deepseek_eval.py and Branch_RAG_llm_as_judge_OpenAI_eval.py respectively. For the evaluation, same as for naive RAG, the corresponding evaluation dataframes are saved into the respective csv files and then used in the respective evaluation scripts.

For Agentic RAG,, the notebook Agentic_RAG_eval_Deepseek.ipynb is the notebook for the Agentic RAG model and evaluation with Deepseek model as llm and the notebook Agentic_RAG_eval_OpenAI.ipynb is the notebook for the Agentic RAG model and evaluation with OpenAI model as llm. 
