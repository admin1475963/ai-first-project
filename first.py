from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from typing import Iterator
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
import getpass
import os

# load documents
cards_loader = CSVLoader("./data/cards.csv")
doc1_loader = PyPDFLoader("./data/doc1.pdf")
doc2_loader = PyPDFLoader("./data/doc2.pdf")
deepseekpdf_loader = PyPDFLoader("./data/DeepSeek_V3.pdf")
cards = cards_loader.load()
doc1 = doc1_loader.load()
doc2 = doc2_loader.load()
doc_deepseek = deepseekpdf_loader.load()

# split text
csv_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, add_start_index=True
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, add_start_index=True
)
card_splits = csv_splitter.split_documents(cards)
doc1_splits = text_splitter.split_documents(doc1)
doc2_splits = text_splitter.split_documents(doc2)
deepseek_splits = text_splitter.split_documents(doc_deepseek)

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# embed documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(embedding_function=embeddings, persist_directory="./vdb")
_ = vector_store.add_documents(documents=card_splits)
print("Cards done")
_ = vector_store.add_documents(documents=doc1_splits)
print("doc1 done")
_ = vector_store.add_documents(documents=doc2_splits)
print("doc2 done")
_ = vector_store.add_documents(documents=deepseek_splits)
print("deepseek done")


# create prompt template
prompt = PromptTemplate.from_template(
    """
    Ты помощник-ассистент, который отвечает на вопросы клиента.
    Отвечай используя только следующий контекст.
    Если ты не знаешь или не нашел ответа, так и скажи.
    Если ты нашел несколько ответов, то расскажи о каждой из них.
    Вопрос: {question} 
    Контекст: {context} 
    Ответ:
    """
)

llm = ChatOpenAI(model="gpt-4o-mini")
# define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    for doc in retrieved_docs:
        print(f"{doc.metadata}\n")
        print(f"{doc.page_content}\n")
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# initialize FastAPI app
app = FastAPI()

# define request body
class ChatRequest(BaseModel):
    message: str

@app.post("/message")
async def chat(request: ChatRequest):
    output = graph.invoke({"question": request.message})
    return {"response": output["answer"]}

