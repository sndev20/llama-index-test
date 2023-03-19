import sys
from llama_index import GPTSimpleVectorIndex, LLMPredictor
from langchain import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import openai
import os

# path
root = str(Path(__file__).parents[1].resolve())

# api key
root = str(Path(__file__).parents[1].resolve())
load_dotenv(root + '/.env')
# dotenvだけだと読み込んでくれないので、os.environで読み込む
openai.api_key = os.environ['OPENAI_API_KEY']

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=400))
index = GPTSimpleVectorIndex.load_from_disk(root + "/src/data/p-index.json")

args = sys.argv
question = args[1]
print("Q: "+ question)
output = index.query(question+"100文字以内で要約して教えて下さい。", llm_predictor=llm_predictor)
print("A: %s" % (output))