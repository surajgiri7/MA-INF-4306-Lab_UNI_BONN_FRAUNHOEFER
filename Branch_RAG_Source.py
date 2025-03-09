import pandas as pd
import json
import ast  # To safely parse string lists like "[101,103,104]" 

# Load metadata from JSON file
with open("kb_metadata.json", "r") as f:
    metadata = json.load(f)

# Convert metadata into a dictionary {id: text}
metadata_dict = {str(entry["id"]): entry["text"] for entry in metadata}

# Load the CSV file (which contains "source1" and "source2" columns with lists of IDs)
data_df = pd.read_csv("rag_output.csv")  # Replace with actual filename

# Function to map a list of IDs to text passages and return a list
def map_ids_to_text(id_list_str):
    try:
        # Convert the string list (e.g., "[101,103,104]") to an actual Python list
        id_list = ast.literal_eval(id_list_str)  
        if not isinstance(id_list, list):  # Ensure it's a list
            return ["Invalid ID format"]

        # Map each ID to its corresponding text
        text_list = [metadata_dict.get(str(id), "Not Found") for id in id_list]
        return text_list  # Return as a list
    
    except Exception:
        return ["Error parsing IDs"]

# Apply function to the "source1" and "source2" columns separately
data_df["source_text1"] = data_df["sources_1"].apply(map_ids_to_text)
data_df["source_text2"] = data_df["sources_2"].apply(map_ids_to_text)

# Save updated CSV with source_text1 and source_text2 stored as JSON-like strings (for proper CSV storage)
data_df["source_text1"] = data_df["source_text1"].apply(json.dumps)
data_df["source_text2"] = data_df["source_text2"].apply(json.dumps)

# Save updated CSV
data_df.to_csv("updated_data.csv", index=False)

print("Updated CSV file saved as 'updated_data.csv'")
