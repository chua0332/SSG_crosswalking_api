import numpy as np
import pandas as pd
from sentence_transformers import util
import torch

def get_embeddings(texts, model, to_tensor=False, batch_size=64):
    """
    Compute embeddings for a list/Series of text entries.
    """
    try:
        # Ensure input is Python list of strings
        if not isinstance(texts, list):
            texts = texts.astype(str).tolist()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=to_tensor,  # True → torch tensors
            show_progress_bar=True
        )

        return embeddings
    
    except Exception as e:
        print("Error computing embeddings")
        print("Error message:", e)
        return None
    

def find_best_skill_match(title: str, description: str, model, skill_embeddings, ssg_skills_df: pd.DataFrame):
    """
    Computes best matching skill for a given input skill (title + description).
    """

    # Ensure formatting matches how embeddings were created
    query = f"{title} - {description}".strip()

    # Encode query into embedding
    query_emb = model.encode([query], convert_to_tensor=True)

    # Compute cosine similarity against stored embeddings
    scores = util.cos_sim(query_emb, skill_embeddings)[0]

    # Top match
    top_idx = torch.argmax(scores).item()
    
    # Extract combined skill string
    best_match_combined = ssg_skills_df['skill_description_combined'][top_idx]

    # Split title & description
    mapped_title, mapped_desc = best_match_combined.split("-", 1)
    
    # Retrieve skill_id too
    skill_id = ssg_skills_df['skill_id'][top_idx]
    
    # Extracted the score value so i can create the duplicate flag based on 0.6695 threshold
    score_value = float(scores[top_idx])
    
    # Applying the threshold value
    is_duplicate = score_value >= 0.6695

    return {
        "input_skill_title": title.strip(),
        "input_skill_description": description.strip(),
        "output_skill_id": skill_id,
        "output_skill_title": mapped_title.strip(),
        "output_skill_description": mapped_desc.strip(),
        "score": score_value,
        "isDuplicate": is_duplicate
    }

