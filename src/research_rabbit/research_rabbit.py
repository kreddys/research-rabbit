import json

from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from openai import OpenAI

from research_rabbit.configuration import Configuration
from research_rabbit.utils import deduplicate_and_format_sources, tavily_search, format_sources
from research_rabbit.state import SummaryState, SummaryStateInput, SummaryStateOutput
from research_rabbit.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions

# Initialize OpenAI client with DeepSeek configuration
config = Configuration()
client = OpenAI(
    api_key=config.deepseek_api_key,
    base_url=config.deepseek_base_url
)

def get_completion(messages, json_mode=False):
    """Helper function to get completions from DeepSeek API"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message

# Nodes   
def generate_query(state: SummaryState):
    """Generate a query for web search"""
    
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    messages = [
        {"role": "system", "content": query_writer_instructions_formatted},
        {"role": "user", "content": "Generate a query for web search. Return ONLY a JSON object with a 'query' field containing the search query."}
    ]
    
    result = get_completion(messages, json_mode=True)
    
    try:
        query = json.loads(result.content)
        return {"search_query": query['query']}
    except json.JSONDecodeError:
        # Fallback: if the response isn't valid JSON, use the raw content as the query
        return {"search_query": result.content.strip()}

def web_research(state: SummaryState):
    """Gather information from the vector store instead of web"""
    
    # Get relevant documents from vector store
    results = rag_manager.retrieve_relevant_context(state.search_query, k=1)
    
    # Format the results similar to web search results
    formatted_results = []
    for doc in results:
        formatted_result = {
            "title": "Document Chunk",
            "url": "local://vector-store",
            "content": doc.page_content
        }
        formatted_results.append(formatted_result)
    
    search_results = {"results": formatted_results}
    
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000)
    return {
        "sources_gathered": [format_sources(search_results)], 
        "research_loop_count": state.research_loop_count + 1, 
        "web_research_results": [search_str]
    }

def summarize_sources(state: SummaryState):
    """Summarize the gathered sources"""
    
    existing_summary = state.running_summary
    most_recent_web_research = state.web_research_results[-1]

    if existing_summary:
        human_message_content = (
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )
    else:
        human_message_content = (
            f"Generate a summary of these search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )

    messages = [
        {"role": "system", "content": summarizer_instructions},
        {"role": "user", "content": human_message_content}
    ]
    
    result = get_completion(messages)
    return {"running_summary": result.content}

def reflect_on_summary(state: SummaryState):
    """Reflect on the summary and generate a follow-up query"""

    messages = [
        {"role": "system", "content": reflection_instructions.format(research_topic=state.research_topic)},
        {"role": "user", "content": f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}"}
    ]
    
    result = get_completion(messages)
    follow_up_query = json.loads(result.content)

    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """Finalize the summary"""
    
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """Route the research based on the follow-up query"""

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary" 
    
# Add nodes and edges 
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()