# Retrieval-Augmented Cooking Assistant

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system designed to serve as an intelligent cooking assistant. By combining a local large language model with a retrieval system that indexes recipes from a cookbook PDF, the system is able to generate detailed, context-aware responses to cooking-related questions. The assistant not only generates fluent text but also grounds its responses in the actual content of the cookbook.

## Dataset & Source Material
**Source Cookbook:**  
The recipes are extracted from a digital version of Yotam Ottolenghi's *Ottolenghi Simple* (2018). The PDF is processed to extract both the textual content and the table of contents, which is used to associate pages with individual recipes.

## Recipe Extraction and Vector Store Creation

### **1. Extracting the Recipes**
- **PDF Parsing:**  
  The cookbook PDF is parsed using `pdfplumber` to extract text on a page-by-page basis, and `PyMuPDF` (via `fitz`) to extract the table of contents.
- **Data Organization:**  
  The extracted pages are converted into a pandas DataFrame. The table of contents is used to identify recipe pages (specifically, level 2 entries) and to associate each page with its corresponding recipe title.
- **Merging Recipe Data:**  
  An as-of merge assigns each page of text to the most recent recipe title, after which pages are grouped to form complete recipe descriptions.

### **2. Converting Recipes into Documents**
- **Text Splitting:**  
  Each recipe description is split into chunks of up to 1000 tokens (with a 20% overlap) using LangChain’s `RecursiveCharacterTextSplitter`, ensuring that each document is a self-contained unit.
- **Document Creation:**  
  The chunks are converted into LangChain `Document` objects, each tagged with the corresponding recipe metadata.

### **3. Building the Vector Store**
- **Embedding Generation:**  
  The recipe documents are embedded using a HuggingFace model (`all-MiniLM-L6-v2`).
- **Vector Store Construction:**  
  The embeddings are stored in a Chroma vector store, enabling fast and efficient retrieval of relevant recipe data during question-answering.

## Retrieval-Augmented Generation Pipeline

### **1. Local LLM Setup**
- **Model Initialization:**  
  A local LLM (meta-llama/Llama-3.1-8B-Instruct) is loaded and wrapped using HuggingFace’s `pipeline` to generate text.
- **LangChain Integration:**  
  The model is integrated with LangChain via the `HuggingFacePipeline` wrapper, making it compatible with custom chain operations.

### **2. Custom Prompt Engineering**
A custom prompt is defined to instruct the model to act as an expert chef. The prompt template requires the generated response to include:
- A brief introduction to the recipe or topic.
- A clear, structured list of ingredients with quantities.
- Detailed, step-by-step cooking instructions.
- Useful tips or variations for the dish.

### **3. Constructing the RAG Chain**
- **With Retrieval:**  
  The chain retrieves context from the Chroma vector store, processes the retrieved documents, and then feeds them to the LLM along with the user’s question. This ensures that the response is accurate and references the cookbook.
- **Without Retrieval:**  
  For comparison, a simple chain is also implemented where no additional context is provided to the model.

## Evaluation
The experiment compared the performance of the LLM with and without the retrieval step:
- **Without Retrieval:**  
  Although the model generates an answer that appears acceptable at first glance, it lacks specific references to the cookbook, leading to generic and potentially inaccurate instructions.
- **With Retrieval:**  
  Incorporating a retrieval step allows the model to access the specific recipes in the cookbook. As a result, the generated answers are correct, complete, and directly reference the source material.
