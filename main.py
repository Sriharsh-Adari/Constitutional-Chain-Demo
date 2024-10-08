#-----------------------------------------------------------------
# LangGraph - Constitutional AI
#-----------------------------------------------------------------

import boto3
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import ChatBedrock

from typing import List, Optional, Tuple
from typing_extensions import Annotated, TypedDict

from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.constitutional_ai.prompts import (
    CRITIQUE_PROMPT,
    REVISION_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

import streamlit as st

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

llm = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

#-----------------------------------------------------------------
# LangChain Constitutional chain migration to LangGraph

class Critique(TypedDict):
    """Generate a critique, if needed."""

    critique_needed: Annotated[bool, ..., "Whether or not a critique is needed."]
    critique: Annotated[str, ..., "If needed, the critique."]


critique_prompt = ChatPromptTemplate.from_template(
    "Critique this response according to the critique request. "
    "If no critique is needed, specify that.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Critique request: {critique_request}"
)

revision_prompt = ChatPromptTemplate.from_template(
    "Revise this response according to the critique and reivsion request.\n\n"
    "Query: {query}\n\n"
    "Response: {response}\n\n"
    "Critique request: {critique_request}\n\n"
    "Critique: {critique}\n\n"
    "If the critique does not identify anything worth changing, ignore the "
    "revision request and return 'No revisions needed'. If the critique "
    "does identify something worth changing, revise the response based on "
    "the revision request.\n\n"
    "Revision Request: {revision_request}"
)

chain = llm | StrOutputParser()
critique_chain = critique_prompt | llm.with_structured_output(Critique)
revision_chain = revision_prompt | llm | StrOutputParser()

#-----------------------------------------------------------------
# LangGraph State

class State(TypedDict):
    query: str
    constitutional_principles: List[ConstitutionalPrinciple]
    initial_response: str
    critiques_and_revisions: List[Tuple[str, str]]
    response: str
    
#-----------------------------------------------------------------
# Amazon Bedrock KnowledgeBase

from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="W3NMIJXLUE", # 👈 Change it to your Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
)

#-----------------------------------------------------------------
# LangGraph: RAG node

# LangChain - RAG chain with citations
def retrieval_augmented_generation(state: State):
    """Generate RAG response using Amazon KnowledgeBase. """
    
    template = '''Answer the question based only on the following context:
    {context}
    
    Question: {question}'''
    
    prompt = ChatPromptTemplate.from_template(template)
        
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(response = prompt | llm | StrOutputParser())
        .pick(["response", "context"])
    )
    
    response = chain.invoke(state["query"])
    
    return {"response": response['response'], "initial_response": response['response']}

#-----------------------------------------------------------------
# LangGraph: critique_and_revise Node

def critique_and_revise(state: State):
    """Critique and revise response according to principles."""
    critiques_and_revisions = []
    response = state["initial_response"]
    for principle in state["constitutional_principles"]:
        critique = critique_chain.invoke(
            {
                "query": state["query"],
                "response": response,
                "critique_request": principle.critique_request,
            }
        )
        if critique["critique_needed"]:
            revision = revision_chain.invoke(
                {
                    "query": state["query"],
                    "response": response,
                    "critique_request": principle.critique_request,
                    "critique": critique["critique"],
                    "revision_request": principle.revision_request,
                }
            )
            response = revision
            critiques_and_revisions.append((critique["critique"], revision))
        else:
            critiques_and_revisions.append((critique["critique"], ""))
    return {
        "critiques_and_revisions": critiques_and_revisions,
        "response": response,
    }

#-----------------------------------------------------------------
# Build Graph

# LangGraph - workflow orchestration 
@st.cache_resource
def create_graph():
    
    graph = StateGraph(State)
    graph.add_node("retrieval_augmented_generation", retrieval_augmented_generation)
    graph.add_node("critique_and_revise", critique_and_revise)
    
    graph.add_edge(START, "retrieval_augmented_generation")
    graph.add_edge("retrieval_augmented_generation", "critique_and_revise")
    graph.add_edge("critique_and_revise", END)
    
    # Compile
    app = graph.compile()

    return app

#-----------------------------------------------------------------
# This is for the graph workflow visualization on the sidebar
@st.cache_data
def create_graph_image():
    return create_graph().get_graph().draw_mermaid_png()

#-----------------------------------------------------------------
# Custom ConstitutionalPrinciple - DEI Principle

constitutional_principles = [
    ConstitutionalPrinciple(
        name="DEI Principle",
        critique_request="Analyze the content for any lack of diversity, equity, or inclusion. Identify specific instances where the text could be more inclusive or representative of diverse perspectives.",
        revision_request="Rewrite the content by incorporating critiques to be more diverse, equitable, and inclusive. Ensure representation of various perspectives and use inclusive language throughout."
    )
]

# ------------------------------------------------------------------------
# Streamlit App

# Clear Chat History fuction
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.subheader('Constitutional AI Demo')
    st.image(create_graph_image())
    st.text("""
    ConstitutionalPrinciple(
        name="DEI Principle",
        critique_request="Analyze the content for any lack of diversity, equity, or inclusion. Identify specific instances where the text could be more inclusive or representative of diverse perspectives.",
        revision_request="Rewrite the content by incorporating critiques to be more diverse, equitable, and inclusive. Ensure representation of various perspectives and use inclusive language throughout."
    )
    """)
    st.button('Clear Screen', on_click=clear_screen)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner(f"Generating..."):
        
        app = create_graph()
        
        start_time = time.time()
        
        generation = {}
        for chunk in app.stream(
            {"query": prompt, "constitutional_principles": constitutional_principles}
        ):
            for node, value in chunk.items():
                st.info(f"**{node}**", icon="🤖")
                with st.expander("message"):
                    st.write(value)
                for k,v in value.items():
                    generation[k] = v

    end_time = time.time()
    st.markdown(f"**Run Time**: {end_time - start_time:.2f}s\n")

    with st.chat_message("assistant"):
        st.markdown("**[initial response]**")
        st.write(generation['initial_response'])
        st.session_state.messages.append({"role": "assistant", "content": "[initial response] " + generation['initial_response']})
    
        st.markdown("**[critiques]**")
        st.write(generation['critiques_and_revisions'][0][0])
        st.session_state.messages.append({"role": "assistant", "content": "[critiques] " + generation['critiques_and_revisions'][0][0]})
    
        st.markdown("**[revised response]**")
        st.write(generation['response'])
        st.session_state.messages.append({"role": "assistant", "content": "[revised response] " + generation['response']})
                

