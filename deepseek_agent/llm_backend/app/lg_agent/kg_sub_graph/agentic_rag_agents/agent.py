"""This file is for LangGraph Studio testing."""

import os

from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

from app.services.dashscope_embeddings import DashScopeSyncEmbeddings
from app.services.dashscope_langchain import create_agent_model

from ps_genai_agents.retrievers.cypher_examples import (
    Neo4jVectorSearchCypherExampleRetriever,
)

# from ps_genai_agents.workflows.single_agent import create_text2cypher_agent
from ps_genai_agents.workflows.multi_agent import (
    create_text2cypher_with_visualization_workflow,
)

neo4j_graph = Neo4jGraph(enhanced_schema=True)
llm = create_agent_model(["langgraph-studio", "agentic-rag"])
embedder = DashScopeSyncEmbeddings()

neo4j_driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI", ""),
    auth=(os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", "")),
)
vector_index_name = "cypher_query_vector_index"

cypher_example_retriever = Neo4jVectorSearchCypherExampleRetriever(
    neo4j_driver=neo4j_driver, vector_index_name=vector_index_name, embedder=embedder
)

# Create the graph to be found by LangGraph Studio
graph = create_text2cypher_with_visualization_workflow(
    llm=llm,
    cypher_example_retriever=cypher_example_retriever,
    llm_cypher_validation=False,
    graph=neo4j_graph,
)
