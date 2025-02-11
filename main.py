from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
#import openai
import google.generativeai as genai
from typing import Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Recipe Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Configure OpenAI
#openai.api_key = os.getenv("OPENAI_API_KEY")
#if not openai.api_key:
#    raise ValueError("OPENAI_API_KEY environment variable is not set")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=google_api_key)
class RecipeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The recipe to generate")
    diet_preference: Optional[str] = Field(None, description="Dietary preference (e.g., vegetarian, vegan)")
    cuisine_type: Optional[str] = Field(None, description="Type of cuisine (e.g., Italian, Mexican)")
    calorie_range: Optional[str] = Field(None, description="Calorie range (e.g., 100-200, 200-300)")

    class Config:
        schema_extra = {
            "example": {
                "query": "chocolate chip cookies",
                "diet_preference": "vegetarian",
                "cuisine_type": "italian",
                "calorie_range": "100-200"
            }
        }

class LearningResource(BaseModel):
    title: str
    url: str
    type: str

class RecipeResponse(BaseModel):
    recipe: str
    image_url: str
    learning_resources: List[LearningResource]

def generate_recipe(query: str, diet_preference: Optional[str] = None, cuisine_type: Optional[str] = None, calorie_range: Optional[str] = None) -> dict:
    logger.info(f"Generating recipe for query: {query}, diet: {diet_preference}, cuisine: {cuisine_type}, calorie range: {calorie_range}")
    
    if not query:
        raise HTTPException(status_code=400, detail="Recipe query is required")

    # Create a detailed prompt for the recipe
    prompt = f"""Create a detailed recipe for {query}"""
    if diet_preference:
        prompt += f" that is {diet_preference}"
    if cuisine_type:
        prompt += f" in {cuisine_type} style"
    if calorie_range:
        prompt += f" with approximately {calorie_range} calories"
    
    prompt += """\n\nFormat the recipe in markdown with the following sections:
    1. Brief Description
    2. Ingredients (as a bulleted list)
    3. Instructions (as numbered steps)
    4. Tips (as a bulleted list)
    5. Nutritional Information (as a bulleted list)
    
    Use markdown formatting like:
    - Headers (###)
    - Bold text (**)
    - Lists (- and 1.)
    - Sections (>)
    """

    try:
        logger.info(f"Sending prompt to Google PaLM: {prompt}")
        
        # Generate recipe text using Google PaLM
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        recipe_text = response.text
        logger.info("Successfully generated recipe text")

        # Generate recipe image (Note: Google PaLM does not support image generation)
        # Use a placeholder image URL for now
        image_url = "https://via.placeholder.com/1024x1024"  # Placeholder image URL
        logger.info("Using placeholder image URL")

        # Get learning resources
        learning_resources = get_learning_resources(query)
        logger.info("Successfully generated learning resources")

        response_data = {
            "recipe": recipe_text,
            "image_url": image_url,
            "learning_resources": learning_resources
        }
        
        return response_data
    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_learning_resources(recipe_name: str) -> list:
    return [
        {
            "title": f"Master the Art of {recipe_name}",
            "url": f"https://cooking-school.example.com/learn/{recipe_name.lower().replace(' ', '-')}",
            "type": "video"
        },
        {
            "title": f"Tips and Tricks for Perfect {recipe_name}",
            "url": f"https://recipes.example.com/tips/{recipe_name.lower().replace(' ', '-')}",
            "type": "article"
        }
    ]

@app.post("/recipe", response_model=RecipeResponse)
async def get_recipe(request: RecipeRequest):
    logger.info(f"Received recipe request: {request}")
    try:
        result = generate_recipe(request.query, request.diet_preference, request.cuisine_type, request.calorie_range)
        logger.info("Successfully generated recipe response")
        return result
    except Exception as e:
        logger.error(f"Error processing recipe request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
