import os
import logging
import sys
import qdrant_client
from llama_index.core import download_loader, SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext 
from llama_index.core import PromptTemplate, QueryBundle
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import StorageContext
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import set_global_service_context, load_index_from_storage
from llama_index.core.node_parser import JSONNodeParser
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from typing import Optional
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import BaseRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from cohere import Client
from typing import Any, List, Optional
import matplotlib.pyplot as plt
from llama_index.llms.groq import Groq
from llama_index.llms.cohere import Cohere
from llama_index.llms.gemini import Gemini
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI

def get_rag_answer(question, llm_choice, context, top_k, rerank, similarity):

    # Initialize Ollama model and set up Service Context
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
    )
    
    if llm_choice == "GPT4-turbo":
        llm = OpenAI(model="gpt-4-0125-preview", context_window=context, temperature=0)
    elif llm_choice == "Command R Plus":
        llm = Cohere(model="command-r-plus", api_key=os.environ.get("COHERE_API_KEY"), temperature=0)
    elif llm_choice == "Claude Opus":
        llm = Anthropic(model="claude-3-opus-20240229", api_key=os.environ.get("ANTHROPIC_API_KEY"), temperature=0)
    elif llm_choice == "Gemini 1.5":
        llm = Gemini(model="models/gemini-1.5-pro-latest", api_key=os.environ.get("GOOGLE_API_KEY"), context_window=context, temperature=0)
    elif llm_choice == "Mixtral 8x22B":
        llm = MistralAI(model="open-mixtral-8x22b", api_key=os.environ.get("MISTRAL_API_KEY"), temperature=0)
    elif llm_choice == "Llama 3 70B":
        llm = Groq(model="llama3-70b-8192", api_key=os.environ.get("MISTRAL_API_KEY"), temperature=0)
    else:
        llm = Groq(model="mixtral-8x7b-32768", api_key=os.environ.get("GROQ_API_KEY"), context_window=context, generate_kwargs={"temperature":0.0})
        
   
    embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")

    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    set_global_service_context(service_context)  # Set global service context

    QDRANT_MAIN_URL = "[QDRANT_URL]"
    QDRANT_NODES = (
        "[QDRANT_NODE]"
    )
    QDRANT_API_KEY = "QDRANT_API_KEY"

    from qdrant_client import QdrantClient

    client = QdrantClient(QDRANT_MAIN_URL, api_key=QDRANT_API_KEY)

    vector_store = QdrantVectorStore(client=client, collection_name="[QDRANT_COLLECTION_NAME]")

    qa_prompt = PromptTemplate(
            "You are a helpful medical expert, and your task is to answer a multi-choice medical question using the provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Given this information and not prior knowledge, please think carefully and then choose the answer from the provided options. Your responses will be used for research purposes only, so please have a definite answer: {query_str}\n"
    )

    query_str = question

    query_embedding = embed_model.get_query_embedding(query_str)

    query_mode = "default"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=top_k, mode=query_mode
    )
    query_result = vector_store.query(vector_store_query)

    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    for node in nodes_with_scores:
        display_source_node(node, source_length=250)

    class CustomRetriever(BaseRetriever):
        def __init__(
            self,
            vector_store,
            embed_model,
            query_mode: str = "default",
            similarity_top_k: int = 10,
        ) -> None:
            """Init params."""
            self._vector_store = vector_store
            self._embed_model = embed_model
            self._query_mode = query_mode
            self._similarity_top_k = similarity_top_k
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            """Retrieve."""
            query_embedding = embed_model.get_query_embedding(query_str)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._similarity_top_k,
                mode=self._query_mode,
            )
            query_result = vector_store.query(vector_store_query)
            print(f"Number of nodes returned by vector store query: {len(query_result.nodes)}")

            nodes_with_scores = []
            for index, node in enumerate(query_result.nodes):
                score: Optional[float] = None
                if query_result.similarities is not None:
                    score = query_result.similarities[index]
                nodes_with_scores.append(NodeWithScore(node=node, score=score))
            print(f"Number of nodes in nodes_with_scores: {len(nodes_with_scores)}")
            return nodes_with_scores

    retriever = CustomRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=top_k
    )

    retrieved_nodes = retriever.retrieve(query_str)

    class CustomNodePostprocessor:
        def __init__(self, similarity_cutoff: float, rerank_model: str, top_n: int, api_key: str):
            self.similarity_processor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
            self.rerank_processor = CohereRerank(api_key=api_key, top_n=top_n)

        def postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]) -> List[NodeWithScore]:
            # Apply similarity cutoff
            filtered_nodes = self.similarity_processor.postprocess_nodes(nodes, query_bundle)

            # Apply reranking
            reranked_nodes = self.rerank_processor.postprocess_nodes(filtered_nodes, query_bundle)

            return reranked_nodes

    # Example usage
    custom_processor = CustomNodePostprocessor(rerank_model="rerank-english-v3.0",
        similarity_cutoff=similarity,
        top_n=rerank,
        api_key = os.environ.get("COHERE_API_KEY")                                      
    )

    retrieved_nodes = retriever.retrieve(query_str)

    # Create a QueryBundle
    query_bundle = QueryBundle(query_str=query_str)

    # Postprocess nodes
    postprocessed_nodes = custom_processor.postprocess_nodes(retrieved_nodes, query_bundle)

    def generate_response(postprocessed_nodes, query_str, qa_prompt, llm):
        # Generate context string from postprocessed nodes
        context_str = "\n\n".join([node.node.get_content() for node in postprocessed_nodes])
        fmt_qa_prompt = qa_prompt.format(
            context_str=context_str, query_str=query_str
        )
        response = llm.complete(fmt_qa_prompt)
        return str(response), fmt_qa_prompt, postprocessed_nodes

    # Generate response using postprocessed nodes
    response, fmt_qa_prompt, postprocessed_nodes = generate_response(
        postprocessed_nodes, query_str, qa_prompt, llm
    )

    unique_citations = set()  # To track and eliminate duplicates
    formatted_links = []  # To store formatted citations as clickable links

    # Combine both node lists for processing while keeping track of their source
    combined_nodes = [('postprocessed', node) for node in postprocessed_nodes] + \
                     [('retrieved', node) for node in retrieved_nodes if node not in postprocessed_nodes]

    for source, node in combined_nodes:
        citation = node.node.metadata.get('citation', '')
        doi = citation.split('doi: ')[-1].strip()
        if citation not in unique_citations:
            unique_citations.add(citation)
            # Append <br> for HTML line breaks in Streamlit
            formatted_link = f"[{citation}](https://doi.org/{doi})<br>"
            formatted_links.append((source, formatted_link))

    # Separate the formatted links based on their source
    post_links = [link for source, link in formatted_links if source == 'postprocessed']
    ret_links = [link for source, link in formatted_links if source == 'retrieved']

    # Use '<br><br>' for double line breaks in the response body for HTML content in Streamlit
    response_body = f"<br>{response}<br><br>Citations:<br>" + \
            '<br>'.join(post_links) + \
            "<br>Additional sources that may be relevant:<br>" + \
            '<br>'.join(ret_links) + \
            f"<br><br>Number of postprocessed nodes: {len(postprocessed_nodes)} | Number of retrieved nodes: {len(retrieved_nodes)}"
    return response_body
