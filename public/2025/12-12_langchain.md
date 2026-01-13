
- RAG
- MCP
- LLM
- [langchain docs](https://python.langchain.com/docs/introduction/)
- [openai platform](https://platform.openai.com/docs/models)
- [Gemini apikey](https://aistudio.google.com/apikey)
- [courses intro-to-langgraph](https://academy.langchain.com/courses/intro-to-langgraph)
```py
pip install python-dotenv langchain pinecone-client

export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."

export OPENAI_API_KEY="..."
export PINECONE_ENV="..."
export PINECONE_API_KEY="..."
```
```py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# True [TODO]

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
llm("Explain llm in one sentence")
```
```py
# Gemini api key
export GOOGLE_API_KEY="..."
export LANGSMITH_API_KEY="..."
```
```py
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model.invoke("Hello, world!")


```
```py
from google import genai
api_key = os.environ.get("GOOGLE_API_KEY")
gemini_version = "gemini-2.5-pro"
client = genai.Client(api_key=api_key)
print("All models:", [ii.name for ii in client.models.list()])

uploaded_file = client.files.upload(file=audio_path)

# Generate content using the uploaded file
classes = ["Bird", "Cat", "Cow", "Dog", "Donkey", "Frog", "Lion", "Maymun", "Sheep", "Tavuk"]
response = client.models.generate_content(
    model=gemini_version,
    contents=[
        {"text": "detail description of this animal sound. It could be one of a " + ", ".join(classes)},
        uploaded_file,
    ],
)
json.loads(response.model_dump_json())['candidates'][0]['content']['parts'][0]['text']
```
```py
from google import genai
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter

reader = PdfReader(pdf_path)
```
