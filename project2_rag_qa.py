# rag_document_qa.py
# RAG-Based Document Q&A System
# Uses: LangChain, HuggingFace, ChromaDB, FastAPI

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from PyPDF2 import PdfReader
import docx

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict
    chunk_id: str

class RAGDocumentSystem:
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "microsoft/DialoGPT-medium",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):

        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        self.vector_store = None
        self.qa_chain = None
        self._init_llm()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.prompt_template = "You are a helpful AI assistant. Use the following context to answer the question.\nIf you don't know the answer, say I don't have enough information to answer this.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def _init_llm(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float32,
                device_map='cpu'
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("LLM loaded successfully")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None

    def extract_text_from_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return self._clean_text(text)

    def extract_text_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._clean_text(text)

    def extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return self._clean_text(f.read())

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        return text.strip()

    def chunk_document(self, text: str, source: str) -> List[DocumentChunk]:
        chunks = self.text_splitter.split_text(text)
        document_chunks = []
        for i, chunk in enumerate(chunks):
            doc_chunk = DocumentChunk(
                content=chunk,
                metadata={'source': source, 'chunk_index': i, 'total_chunks': len(chunks)},
                chunk_id=f"{source}_chunk_{i}"
            )
            document_chunks.append(doc_chunk)
        return document_chunks

    def add_documents(self, file_paths: List[str], collection_name: str = "default"):
        all_chunks = []
        for file_path in file_paths:
            print(f"Processing: {file_path}")
            if file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = self.extract_text_from_docx(file_path)
            elif file_path.endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            chunks = self.chunk_document(text, os.path.basename(file_path))
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")

        print(f"\nCreating vector store with {len(all_chunks)} chunks...")
        texts = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]

        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=collection_name
        )

        if self.llm:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": self.prompt}
            )

        print("Vector store ready!")
        return len(all_chunks)

    def query(self, question: str, top_k: int = 3) -> Dict:
        if not self.vector_store:
            return {'answer': "No documents indexed. Please add documents first.", 'sources': []}

        relevant_docs = self.vector_store.similarity_search(question, k=top_k)

        if self.qa_chain:
            try:
                answer = self.qa_chain.run(question)
            except:
                context = "\n".join([doc.page_content for doc in relevant_docs])
                answer = self._generate_simple_answer(question, context)
        else:
            context = "\n".join([doc.page_content for doc in relevant_docs])
            answer = self._generate_simple_answer(question, context)

        sources = [
            {
                'content': doc.page_content[:200] + "...",
                'metadata': doc.metadata,
                'relevance_score': 1.0
            }
            for doc in relevant_docs
        ]

        return {'answer': answer, 'sources': sources, 'question': question}

    def _generate_simple_answer(self, question: str, context: str) -> str:
        sentences = context.split('.')
        question_words = set(question.lower().split())
        scored_sentences = []
        for sent in sentences:
            score = len(set(sent.lower().split()) & question_words)
            if score > 0:
                scored_sentences.append((score, sent.strip()))
        scored_sentences.sort(reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:3]]
        if top_sentences:
            return "Based on the documents: " + ". ".join(top_sentences) + "."
        return "I found relevant information but couldn't generate a specific answer."

# FastAPI Application
app = FastAPI(title="RAG Document Q&A System", version="1.0")

class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict]
    question: str

rag_system = None

@app.on_event("startup")
async def init_rag():
    global rag_system
    rag_system = RAGDocumentSystem()
    print("RAG system initialized")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        num_chunks = rag_system.add_documents([file_path])
        return {
            "message": "Document indexed successfully",
            "filename": file.filename,
            "chunks_created": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = rag_system.query(request.question, request.top_k)
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents_indexed": rag_system.vector_store is not None
    }

if __name__ == "__main__":
    print("=" * 60)
    print("RAG DOCUMENT Q&A SYSTEM")
    print("=" * 60)
    rag = RAGDocumentSystem()

    sample_text = "Company Policy Document\n\n1. Leave Policy:\nEmployees are entitled to 20 days of paid annual leave per year.\nSick leave is unlimited with proper medical documentation.\nMaternity leave is 26 weeks as per government regulations.\n\n2. Remote Work Policy:\nEmployees can work remotely up to 3 days per week.\nCore collaboration hours are 10 AM to 4 PM IST.\nRemote work requests must be approved by managers.\n\n3. Performance Review:\nPerformance reviews happen twice a year: June and December.\nReview criteria include: project delivery, code quality, collaboration, and innovation.\nTop performers receive bonuses up to 20% of annual salary."

    os.makedirs("sample_docs", exist_ok=True)
    with open("sample_docs/policy.txt", "w") as f:
        f.write(sample_text)

    rag.add_documents(["sample_docs/policy.txt"])

    questions = [
        "How many days of leave do I get?",
        "What are the performance review criteria?",
        "Can I work from home?"
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {len(result['sources'])} chunks found")

    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
