from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
import os

def load_pdf_files(pdf_directory):
    """
    Loads PDF files from the specified directory and handles any errors during loading.
    Skips files that cannot be loaded.
    """
    documents = []

    # Walk through each directory and file in the given directory
    for root, dirs, files in os.walk(pdf_directory):
        for file in files:
            if file.endswith('.pdf'):
                # Construct the full path of the PDF file
                file_path = os.path.join(root, file)
                
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    documents.extend(pages)
                    print(f"Successfully loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue  # Skip this file and move to the next one
    return documents

# Single pdf
# loader = PyPDFLoader("docs/file.pdf")
# pages = loader.load_and_split()

# Pdfs in folder
pdf_directory = "test_pdfs"
pages = load_pdf_files(pdf_directory)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
faiss_index = FAISS.from_documents(pages, embeddings)
# faiss_index.save_local("faiss_index")
# faiss_index = FAISS.load_local("faiss_index", embeddings)

index = faiss_index

query = "Do you have information about a flood incident that occurred in 2004?"

found_docs = index.similarity_search(query, k=15)
# found_docs = index.max_marginal_relevance_search(query, k=2)

# for doc in found_docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

model="deepseek-r1:32b"
llm = OllamaLLM(model=model)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.as_retriever(search_type="similarity", search_kwargs={"k":15}), 
    return_source_documents=True)

result = qa({"query": query})
print(result['result'])
