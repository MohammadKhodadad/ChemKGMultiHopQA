import json

def load_qa_items(json_path):
    """
    Loads QA items from a JSON file and creates a list of dictionaries for evaluation.
    
    The JSON file is expected to contain a list of objects. Each object should include:
      - "q": the question.
      - "a": the expected answer.
      - "path": a list of dictionaries, where each dictionary has "text" and "meta1" fields.
      
    This function aggregates the "text" and "meta1" from each element in "path" as the context.
    
    Args:
        json_path (str): The path to the JSON file.
        
    Returns:
        list of dict: A list where each dictionary has the following keys:
                      - "question": the question text.
                      - "expected": the expected answer.
                      - "context": the aggregated context from "text" and "meta1" in the path.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    qa_items = []
    for entry in data:
        question = entry.get("q", "").strip()
        expected = entry.get("a", "").strip()
        
        # Build the context by concatenating "text" and "meta1" from each element in the "path" list.
        context_parts = []
        for path_item in entry.get("path", []):
            text = path_item.get("text", "").strip()
            meta1 = path_item.get("meta1", "").strip()
            # Combine text and meta1 if available.
            combined = " ".join(part for part in [text, meta1] if part)
            if combined:
                context_parts.append(combined)
        context = "\n".join(context_parts)
        length=len(entry.get("path", []))
        qa_items.append({
            "question": question,
            "expected": expected,
            "context": context,
            "length":length
        })
    
    return qa_items

# Example usage:
if __name__ == "__main__":
    # Replace 'qa_data.json' with the actual path to your JSON file.
    json_file_path = "/media/torontoai/GraphRAG/GraphRAG/data_loader/data/chemrxiv_qas.json"
    qa_items = load_qa_items(json_file_path)
    # for item in qa_items:
    #     print("Question:", item["question"])
    #     print("Expected Answer:", item["expected"])
    #     print("Context:", item["context"])
    #     print("-" * 50)
