from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4, shutil, os
from langchain_community.document_loaders import WebBaseLoader

def scrape_webpage(url):
    bs4_strainer = bs4.SoupStrainer(class_="content-area")
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={"parse_only": bs4_strainer},
    )
    return loader.load()

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=100, add_start_index=True
    )
    return text_splitter.split_documents(docs)

def store_embeddings(text_chunks):
    print("Initializing embeddings...")
    try:
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        print("Embeddings initialized.")

        print("Creating vectorstore...")
        vectorstore = Chroma.from_documents(text_chunks, embedding, persist_directory="./chroma_db")
        vectorstore.persist()

        print("Vectorstore created!")

        return vectorstore
    except Exception as e:
        print(f"Error in embeddings: {e}")
        return None

if __name__ == "__main__":
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # Step 1: Scrape webpage
    url = "https://pythonology.eu/a-rag-web-scraper-with-langchain-ollama-and-chroma/"
    docs = scrape_webpage(url)
    print("Step 1 done")

    # Step 2: Split text into chunks
    text_chunks = split_text(docs)
    print("Step 2 done")

    # Step 3: Generate embeddings and store them
    vectorstore = store_embeddings(text_chunks)
    print("Step 3 done")

    # Step 4: Set up Q&A system
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "alpha": 0.7})
    print("Step 4 done")

    # Step 5: Ask a question
    query = "Can you do a step by step summary of the article?"
    retrieved_docs = retriever.invoke(query)
    context = ' '.join([doc.page_content for doc in retrieved_docs])

    print("Step 5 done")

    llm = OllamaLLM(model="llama3.1:latest")

    # Answer
    response = llm.invoke(f"""Answer the question according to the context given:
            Question: {query}
            Context: {context}
    """)

    print("\nGenerated Response:", response)
    print(f"\n ----- Context retrieved from Database w/ embeddings: -----\n\n {context}")
