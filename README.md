# Requirements

Install dependencies in `requirements.txt` and ollama with `bge-m3` and `llama3.1:8b`

# How to launch

~~~
ollama &> ollama.log &
uvicorn first:app --host 0.0.0.0 --port 8000 &> uvicorn.log &
~~~
