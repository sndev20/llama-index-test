import csv
import os
from dotenv import load_dotenv
import openai
from llama_index import GPTSimpleVectorIndex, SimpleWebPageReader
from llama_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor
from pathlib import Path

root = str(Path(__file__).parents[1].resolve())
load_dotenv(root + '/.env')
# dotenvだけだと読み込んでくれないので、os.environで読み込む
openai.api_key = os.environ['OPENAI_API_KEY']

urls = []
with open(root + '/src/assets/personal.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        urls.append(row[0])

documents = SimpleWebPageReader().load_data(urls)
index = GPTSimpleVectorIndex(documents=documents, llm_predictor=ChatGPTLLMPredictor()
)
index.save_to_disk(root + '/src/data/p-index.json')