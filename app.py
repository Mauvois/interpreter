from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Ensure that the API key is set correctly
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterpretationRequest(BaseModel):
    question: str
    iching_response: str


@app.post("/interpret")
async def interpret(request: InterpretationRequest):
    question = request.question
    iching_response = request.iching_response

    if not question or not iching_response:
        logger.error("Both question and I Ching response are required.")
        raise HTTPException(
            status_code=400, detail="Both question and I Ching response are required."
        )

    prompt = (
        f"L'utilisateur a demandé: {question}\n"
        f"I Ching a répondu: {iching_response}\n"
        f"Fournis une interpretation en 2 courts paragraphes"
    )

    try:
        logger.info(f"Sending request to OpenAI API with prompt: {prompt}")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        interpretation = completion.choices[0].message.content.strip()
        return {"interpretation": interpretation}
    except Exception as e:  # Catching generic exception to log error properly
        logger.error(f"Error during OpenAI API call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"},
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
