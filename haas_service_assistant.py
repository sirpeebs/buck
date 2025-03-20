import os
import streamlit as st
import re
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any, Optional
from metadata_extractor import MetadataExtractor
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"  # Using GPT-4 Turbo for better performance

# Configure Pinecone using the new API syntax
PINECONE_API_KEY = "pcsk_rRbje_AfPzubDtVzTSUfGrSH47fnL1arEvGqU46rozwobQdN7qVhiuzAXdPFBt8hkUUjc"
PINECONE_INDEX = "masterdata"

# Initialize Pinecone with the new API syntax
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Initialize metadata extractor
extractor = MetadataExtractor()

def get_unique_models() -> List[str]:
    """
    Fetch unique models from the Pinecone database.
    
    Returns:
        List of unique model names
    """
    try:
        # Query Pinecone with a dummy vector to get all records
        # We only need the metadata, so we can use any vector
        dummy_vector = [0.0] * 1536  # OpenAI embedding dimension
        results = index.query(
            vector=dummy_vector,
            top_k=10000,  # Get a large number of results
            include_metadata=True
        )
        
        # Extract unique models from metadata
        models = set()
        for match in results.matches:
            if "Model" in match.metadata:
                models.add(match.metadata["Model"])
        
        # Sort models alphabetically
        return sorted(list(models))
    except Exception as e:
        st.error(f"Error fetching models from database: {str(e)}")
        return []

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a given text using OpenAI's embedding model."""
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def query_pinecone(
    query_embedding: List[float],
    top_k: int = 3,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> List[Dict]:
    """
    Query Pinecone with the embedding and optional metadata filters.
    
    Args:
        query_embedding: The embedding vector to query with
        top_k: Number of top results to return
        metadata_filters: Filters for metadata fields (Model, Serial, etc.)
        
    Returns:
        List of matching records with their metadata
    """
    # Prepare the filter if metadata filters are provided
    filter_dict = {}
    if metadata_filters:
        for key, value in metadata_filters.items():
            # Skip the Keywords field as it's not a direct filter
            if key != "Keywords" and value:
                filter_dict[key] = {"$eq": value}
    
    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True
    }
    
    if filter_dict:
        query_params["filter"] = filter_dict
    
    # Query Pinecone
    results = index.query(**query_params)
    
    return results.matches

def extract_parts_info(text: str) -> List[Dict[str, str]]:
    """
    Extract part numbers and descriptions from a service text.
    
    Args:
        text: A service description text
        
    Returns:
        List of dictionaries with part number and description
    """
    parts = []
    
    # Enhanced pattern for part numbers: digit-digit format typical for Haas parts
    # Also captures part numbers with optional letters (e.g., 123-4567A)
    part_matches = re.findall(r'\b(\d+-\d+[A-Za-z]*)\b([^.]*)', text)
    
    for match in part_matches:
        part_num = match[0].strip()
        # Try to extract a description if available
        description = match[1].strip()
        
        # Clean up the description
        if description:
            # Remove common prefixes and clean up
            description = re.sub(r'^(part|component|assembly|unit|module)\s+', '', description, flags=re.IGNORECASE)
            description = description.strip('., ')
            
            # If description is too short or generic, try to get more context
            if len(description.split()) < 2:
                # Look for more context in the surrounding text
                context_match = re.search(rf'{re.escape(part_num)}\s*[^.]*?([^.]*?)(?=\d+-\d+[A-Za-z]*|$)', text)
                if context_match:
                    additional_context = context_match.group(1).strip()
                    if additional_context:
                        description = additional_context
        
        if description:
            parts.append({"number": part_num, "description": description})
        else:
            parts.append({"number": part_num, "description": "Component"})
    
    return parts

def format_response(query: str, results: List[Dict], metadata_filters: Dict[str, Any]) -> str:
    """
    Format the retrieved results into a helpful response for the technician.
    
    Args:
        query: The original query
        results: List of matching records with their metadata
        metadata_filters: The metadata filters applied to the search
        
    Returns:
        Formatted response with troubleshooting guidance
    """
    if not results:
        # If no results with filters, suggest broadening the search
        if metadata_filters and any(key in metadata_filters for key in ["Model", "Alarm", "Serial"]):
            filter_terms = []
            if "Model" in metadata_filters:
                filter_terms.append(f"model {metadata_filters['Model']}")
            if "Alarm" in metadata_filters:
                filter_terms.append(f"Alarm {metadata_filters['Alarm']}")
            if "Serial" in metadata_filters:
                filter_terms.append(f"serial {metadata_filters['Serial']}")
                
            filters_desc = " and ".join(filter_terms)
            return f"I couldn't find any records for {filters_desc}. Would you like me to search more broadly or provide additional details about the issue?"
        
        return "I couldn't find any relevant information for your query. Could you provide more details about the issue you're experiencing?"
    
    # Group information by metadata for coherent response
    models = set()
    alarms = set()
    common_issues = []
    troubleshooting_steps = []
    parts_info = []
    verification_tests = []
    
    # Extract keywords from the query for better part relevance
    query_keywords = set(word.lower() for word in query.split())
    
    # Add common technical terms that might be relevant
    technical_terms = {
        "spindle", "tool", "axis", "motor", "sensor", "board", "cable", "bearing",
        "drive", "control", "power", "coolant", "chip", "program", "alarm",
        "error", "fault", "malfunction", "repair", "replace", "service"
    }
    query_keywords.update(technical_terms)
    
    for result in results:
        metadata = result.get("metadata", {})
        score = result.get("score", 0)
        
        if "Model" in metadata:
            models.add(metadata["Model"])
        
        if "Alarm" in metadata:
            alarms.add(metadata["Alarm"])
            
        if "WorkRequired" in metadata and metadata["WorkRequired"]:
            common_issues.append({
                "issue": metadata["WorkRequired"],
                "score": score
            })
            
        if "ServicePerformed" in metadata and metadata["ServicePerformed"]:
            service_text = metadata["ServicePerformed"]
            steps = service_text.split('. ')
            for step in steps:
                step = step.strip()
                if step:
                    troubleshooting_steps.append({
                        "step": step,
                        "score": score
                    })
            
            # Extract parts information with enhanced relevance scoring
            extracted_parts = extract_parts_info(service_text)
            for part in extracted_parts:
                # Calculate part relevance score based on multiple factors
                part_relevance = score
                part_context = part["description"].lower()
                
                # Boost score based on various factors
                for keyword in query_keywords:
                    if keyword in part_context:
                        part_relevance *= 1.5
                
                # Additional boost if part is mentioned in troubleshooting steps
                if any(part["number"] in step["step"] for step in troubleshooting_steps):
                    part_relevance *= 1.3
                
                # Boost if part is mentioned in common issues
                if any(part["number"] in issue["issue"] for issue in common_issues):
                    part_relevance *= 1.3
                
                # Store the full context for better relevance
                parts_info.append({
                    "part": part,
                    "score": part_relevance,
                    "context": service_text.lower(),
                    "mentioned_in_steps": any(part["number"] in step["step"] for step in troubleshooting_steps),
                    "mentioned_in_issues": any(part["number"] in issue["issue"] for issue in common_issues)
                })
                
        if "VerificationTest" in metadata and metadata["VerificationTest"]:
            verification_tests.append({
                "test": metadata["VerificationTest"],
                "score": score
            })
    
    # Sort and deduplicate information by relevance (score)
    def deduplicate_and_sort(items, key_field):
        unique_items = {}
        for item in items:
            text = item[key_field].lower()
            if text not in unique_items or unique_items[text]["score"] < item["score"]:
                unique_items[text] = item
        return sorted(unique_items.values(), key=lambda x: x["score"], reverse=True)
    
    sorted_issues = deduplicate_and_sort(common_issues, "issue")
    sorted_steps = deduplicate_and_sort(troubleshooting_steps, "step")
    
    # For parts, deduplicate by part number and sort by relevance
    unique_parts = {}
    for part_item in parts_info:
        part_num = part_item["part"]["number"]
        if part_num not in unique_parts or unique_parts[part_num]["score"] < part_item["score"]:
            unique_parts[part_num] = part_item
    sorted_parts = sorted(unique_parts.values(), key=lambda x: x["score"], reverse=True)
    
    sorted_tests = deduplicate_and_sort(verification_tests, "test")
    
    # Construct response
    response = []
    
    # Introduction based on available information
    intro = "Based on similar service records,"
    if alarms:
        alarm_list = ", ".join(alarms)
        intro += f" for Alarm {alarm_list}"
    if models:
        model_list = ", ".join(models)
        intro += f" on {model_list} models,"
    
    response.append(f"{intro} here's what I found:")
    
    # Common issues
    if sorted_issues:
        response.append("\n**Common issues identified:**")
        for i, issue in enumerate(sorted_issues[:3], 1):  # Top 3 issues
            response.append(f"{i}. {issue['issue']}")
    
    # Troubleshooting steps
    if sorted_steps:
        response.append("\n**Recommended troubleshooting steps:**")
        for i, step in enumerate(sorted_steps[:5], 1):  # Top 5 steps
            response.append(f"{i}. {step['step']}")
    
    # Parts information with enhanced context
    if sorted_parts:
        response.append("\n**Relevant parts based on your issue:**")
        for i, part_item in enumerate(sorted_parts[:5], 1):  # Top 5 parts
            part = part_item["part"]
            context = part_item["context"]
            
            # Extract a relevant snippet from the context
            context_words = context.split()
            if len(context_words) > 20:
                context = "..." + " ".join(context_words[:20]) + "..."
            
            # Build part description with relevance indicators
            part_desc = f"{i}. Part #{part['number']}"
            if part["description"] and part["description"] != "Component":
                part_desc += f" - {part['description']}"
            
            # Add relevance indicators
            relevance_indicators = []
            if part_item["mentioned_in_steps"]:
                relevance_indicators.append("mentioned in troubleshooting steps")
            if part_item["mentioned_in_issues"]:
                relevance_indicators.append("related to common issues")
            
            if relevance_indicators:
                part_desc += f" ({', '.join(relevance_indicators)})"
            
            response.append(part_desc)
            response.append(f"   Context: {context}")
    
    # Verification tests
    if sorted_tests:
        response.append("\n**Recommended verification tests:**")
        for i, test in enumerate(sorted_tests[:3], 1):  # Top 3 tests
            response.append(f"{i}. {test['test']}")
    
    # Closing
    response.append("\nWould you like more specific information about any of these steps or parts?")
    
    return "\n".join(response)

def process_technician_query(query: str) -> str:
    """
    Process a technician's query and generate a helpful response.
    
    Args:
        query: The technician's query about a Haas CNC issue
        
    Returns:
        Formatted response with troubleshooting guidance
    """
    # Extract metadata filters from the query
    metadata_filters = extractor.extract_from_query(query)
    
    # Enhance the query with extracted metadata for better semantic search
    enhanced_query = extractor.enhance_query(query, metadata_filters)
    
    # Generate embedding for the enhanced query
    query_embedding = generate_embedding(enhanced_query)
    
    # Query Pinecone with the embedding and metadata filters
    results = query_pinecone(
        query_embedding=query_embedding,
        top_k=5,  # Get top 5 results for better coverage
        metadata_filters=metadata_filters
    )
    
    # Format the results into a structured format for GPT
    formatted_results = format_response(query, results, metadata_filters)
    
    # Prepare the prompt for GPT
    system_prompt = """You are a Haas CNC service expert. Your task is to analyze the provided service records and generate a comprehensive, helpful response for the technician. 
    Focus on providing clear, actionable steps while maintaining technical accuracy. If the information is insufficient, ask for more details.
    
    Important guidelines:
    1. Do not include any email-style signatures or closings
    2. Keep responses focused and technical
    3. Prioritize the most relevant information first
    4. If part numbers are mentioned, explain their relevance to the issue
    5. Maintain a professional but direct tone"""
    
    user_prompt = f"""Original Query: {query}

Service Records Analysis:
{formatted_results}

Please provide a detailed, helpful response that:
1. Directly addresses the technician's query
2. Incorporates relevant information from the service records
3. Provides clear, actionable steps
4. Maintains a professional, technical tone
5. Asks for clarification if needed"""

    # Get GPT's enhanced response
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting GPT response: {str(e)}")
        return formatted_results  # Fallback to original formatted response if GPT fails

# Streamlit UI
st.set_page_config(page_title="Haas Service Assistant", page_icon="🔧")

st.title("Haas Service Assistant")
st.subheader("CNC Troubleshooting Support for Field Technicians")

st.write("""
This assistant helps Haas CNC field service technicians troubleshoot issues by retrieving 
relevant information from previous service records.
""")

# Add sidebar with common options
st.sidebar.title("Filters")
st.sidebar.write("The assistant will automatically extract these from your query, but you can specify them explicitly here.")

# Fetch unique models from Pinecone
available_models = get_unique_models()
models = [""] + available_models  # Add empty option for no filter
selected_model = st.sidebar.selectbox("CNC Model", options=models)

# Alarm number input
alarm_number = st.sidebar.text_input("Alarm Number", "")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Describe the CNC issue you're troubleshooting..."):
    # Append any explicitly selected filters to the prompt
    filter_context = ""
    if selected_model:
        filter_context += f" for {selected_model} model"
    if alarm_number:
        filter_context += f" with Alarm {alarm_number}"
    
    enhanced_prompt = prompt
    if filter_context:
        enhanced_prompt += filter_context
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching service records..."):
            response = process_technician_query(enhanced_prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    # This allows running the script directly for testing
    if not st.runtime.exists():
        test_query = "I'm troubleshooting Alarm 108 on a VF-3 model. What should I do first?"
        print(process_technician_query(test_query)) 