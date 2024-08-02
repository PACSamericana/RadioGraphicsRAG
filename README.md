- Repository for "Enhancing Large Language Models for Radiologists with Retrieval-Augmented Generation: A Radiology-Specific Approach"
- vector_db.py: code for upserting text chunks into Qdrant vector database. Assumes the folder contains individual JSON files in format {"citation": "(citation)", "text": "(text)"} where (citation) is the citation of the journal article, and (text) is the text chunk.
- RAG_pipeline.py: formatted for use as a backend for a Streamlit app, but can be adapted to running from command line.
- question_benchmark folder contains multiple-choice questions used to evaluate the RAG pipeline. "ABR CORE Exam Study Guide 2014" has questions beginning on page 130.
