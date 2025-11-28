from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pandas as pd 
import torch 
from sentence_transformers import SentenceTransformer
from src.utils.functions import find_best_skill_match

app = FastAPI() 

# Global resources
model = None
skill_embeddings = None
skills_df = None 

class SkillInput(BaseModel): 
    title: str
    description: str 
    
@app.on_event("startup")
def load_resources():
    """
    This loads the mpnet-base-v2 model, skill_embeddings, and ssg_skills_df
    """
    global model, skill_embeddings, skills_df
    print("Loading model and resources...")
    
    # Load mpnet-base-v2 model
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Loading precomputed skill embeddings (torch tensor)
    skill_embeddings = torch.load("./data/ssg_skill_embeddings.pt", map_location=torch.device("cpu"))
    
    # Loading the ssg skills dataframe from CSV
    skills_df = pd.read_csv('./data/ssg_skills.csv')
    
    print("Model, skill embeddings and skills dataframe are all loaded")
    

@app.get("/")
def root():
    return {"status": "OK", "message": "Skill Matching API online"}

@app.get("/match-skill")
def match_skill(input: SkillInput=Depends()):
    """
    Given an input skill title + description, return the best matching SSG skill
    """
    global model, skill_embeddings, skills_df 
    
    result = find_best_skill_match(
        title = input.title,
        description= input.description,
        model = model,
        skill_embeddings=skill_embeddings,
        ssg_skills_df=skills_df,
    )
    
    return result