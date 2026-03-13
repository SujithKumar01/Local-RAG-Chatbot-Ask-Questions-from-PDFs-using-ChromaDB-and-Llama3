# Local RAG System with ChromaDB and Llama3

This project implements a **Local Retrieval-Augmented Generation (RAG) system** that allows you to ask questions about a PDF document and get accurate answers based on its content.

Instead of relying only on a language model’s training data, the system retrieves relevant information from your documents and uses it as context when generating answers.

The entire pipeline runs **locally**, using open-source models and tools.

---

## Project Overview

This system performs the following steps:

1. Load a PDF document
2. Split the document into smaller chunks
3. Convert text chunks into embeddings
4. Store embeddings in a vector database
5. Retrieve the most relevant chunks for a question
6. Send the retrieved context to a local LLM
7. Generate a concise answer

This approach improves accuracy and reduces hallucinations because answers are grounded in the document content.

---

## Technologies Used

* Python
* LangChain
* ChromaDB (Vector Database)
* HuggingFace Embeddings
* Ollama
* Llama3 (Local LLM)

---

## System Architecture

PDF Document
↓
Document Loader
↓
Text Splitter
↓
Text Chunks
↓
Embedding Model
↓
Vector Database (ChromaDB)
↓
Retriever
↓
Prompt Template
↓
Local LLM (Llama3 via Ollama)
↓
Final Answer

---

## Project Structure

```
Local_RAG/
│
├── youtube_guidelines.pdf
├── chroma_db/                # Stored vector database
├── local_rag_with_langchainollama.ipynb            # Main RAG pipeline
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/local-rag-system.git](https://github.com/SujithKumar01/Local-RAG-Chatbot-Ask-Questions-from-PDFs-using-ChromaDB-and-Llama3.git
cd Local-RAG-Chatbot-Ask-Questions-from-PDFs-using-ChromaDB-and-Llama3
```

Create a virtual environment:

```
python -m venv agent_ai
source agent_ai/bin/activate
```

Install dependencies:

```
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-huggingface
pip install langchain-chroma
pip install chromadb
pip install pypdf
pip install ollama
```
or

```
pip install -r requirements.txt
```

---

## Install and Run Ollama

Install Ollama from the official website and download the Llama3 model:

```
ollama pull llama3
```

Run the Ollama server:

```
ollama run llama3
```

---

## How to Run the Project

Run the RAG system:
```
Run the Code Blocks in the .ipynb file 
```

Example interaction:

```
Ask a Question: What are the YouTube community guidelines?

Answer:
YouTube prohibits harmful or misleading content and requires users to follow community safety policies.
```

Exit the program using:

```
exit or quit
```

---

## Key Components Explained

### Document Loader

Loads the PDF and converts it into text documents.

### Text Splitter

Breaks large documents into smaller chunks for efficient processing.

### Embeddings

Converts text into numerical vectors representing semantic meaning.

### Vector Database

Stores embeddings and enables fast similarity search.

### Retriever

Finds the most relevant chunks related to the user’s query.

### Local LLM

Generates the final answer using retrieved context.

---

## Why Use RAG?

Traditional language models may produce incorrect answers.
RAG improves reliability by grounding responses in real documents.

Benefits include:

* More accurate responses
* Reduced hallucinations
* Ability to use private data
* Fully local and privacy-friendly

---

## Example Use Cases

* Chat with PDFs
* Internal company knowledge bases
* Legal or policy document assistants
* Research paper exploration
* Technical documentation assistants

---

## Future Improvements

Possible extensions for this project:

* Multi-document RAG
* Web search integration
* Agentic AI workflows
* Hybrid search (keyword + semantic)
* Streamlit web interface
* Chat history memory

---

## Author

Sujith Kumar Thangella

Machine Learning Engineer | Data Science Enthusiast | DevOps Learner

---

## License

This project is open-source and available under the MIT License.
