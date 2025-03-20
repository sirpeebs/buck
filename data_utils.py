import os
import csv
import json
import time
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"  # Updated to match haas_service_assistant.py

# Configure Pinecone using the new API syntax
PINECONE_API_KEY = "pcsk_rRbje_AfPzubDtVzTSUfGrSH47fnL1arEvGqU46rozwobQdN7qVhiuzAXdPFBt8hkUUjc"
PINECONE_INDEX = "masterdata"

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a given text using OpenAI's embedding model.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a zero vector as fallback (not ideal but prevents crashing)
        return [0.0] * 1536  # OpenAI ada embedding dimension

def prepare_service_record_for_embedding(record: Dict[str, Any]) -> str:
    """
    Prepare a service record for embedding by combining relevant fields.
    
    Args:
        record: Dictionary containing service record data
        
    Returns:
        Formatted text ready for embedding
    """
    text_parts = []
    
    # Add model information if available
    if "Model" in record and record["Model"]:
        text_parts.append(f"Model: {record['Model']}")
    
    # Add alarm information if available
    if "Alarm" in record and record["Alarm"]:
        text_parts.append(f"Alarm: {record['Alarm']}")
    
    # Add work required information
    if "WorkRequired" in record and record["WorkRequired"]:
        text_parts.append(f"Issue: {record['WorkRequired']}")
    
    # Add service performed information
    if "ServicePerformed" in record and record["ServicePerformed"]:
        text_parts.append(f"Service: {record['ServicePerformed']}")
    
    # Add verification test information
    if "VerificationTest" in record and record["VerificationTest"]:
        text_parts.append(f"Verification: {record['VerificationTest']}")
    
    # Combine all parts with newlines
    return "\n".join(text_parts)

def load_data_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load service records from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries containing service records
    """
    records = []
    
    try:
        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        print(f"Loaded {len(records)} records from {file_path}")
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
    
    return records

def batch_upsert_to_pinecone(
    records: List[Dict[str, Any]],
    batch_size: int = 100,
    namespace: str = ""
) -> None:
    """
    Batch upsert service records to Pinecone.
    
    Args:
        records: List of service record dictionaries
        batch_size: Size of batches for upserting
        namespace: Pinecone namespace to use
    """
    # Initialize Pinecone with the new API syntax
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    
    total_records = len(records)
    print(f"Preparing to upsert {total_records} records to Pinecone index {PINECONE_INDEX}")
    
    # Process in batches
    for i in tqdm(range(0, total_records, batch_size)):
        batch = records[i:i + batch_size]
        items_to_upsert = []
        
        for j, record in enumerate(batch):
            # Create a record ID
            record_id = f"record_{i + j}"
            if "RecordID" in record:
                record_id = str(record["RecordID"])
            elif "Id" in record:
                record_id = str(record["Id"])
            
            # Prepare text for embedding
            text_for_embedding = prepare_service_record_for_embedding(record)
            
            # Generate embedding
            embedding = generate_embedding(text_for_embedding)
            
            # Prepare metadata (exclude large text fields from metadata)
            metadata = {k: v for k, v in record.items() if k not in ["embedding", "vector"]}
            
            # Add to upsert batch
            items_to_upsert.append({
                "id": record_id,
                "values": embedding,
                "metadata": metadata
            })
            
            # Avoid rate limiting on OpenAI API
            time.sleep(0.1)
        
        try:
            # Upsert the batch
            upsert_response = index.upsert(vectors=items_to_upsert, namespace=namespace)
            print(f"Upserted batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size}, " 
                  f"vectors: {upsert_response.upserted_count}")  # Changed to use object attribute
        except Exception as e:
            print(f"Error upserting batch to Pinecone: {e}")
    
    print(f"Completed upserting {total_records} records to Pinecone")

def test_search(query: str, top_k: int = 3) -> None:
    """
    Test search functionality by querying the Pinecone index.
    
    Args:
        query: The query text
        top_k: Number of top results to return
    """
    # Initialize Pinecone with the new API syntax
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    
    print(f"Testing search with query: {query}")
    
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    
    try:
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"\nFound {len(results.matches)} matches:")
        for i, match in enumerate(results.matches):
            print(f"\nResult {i+1} (Score: {match.score:.4f}):")
            for key, value in match.metadata.items():
                if key in ["Model", "Alarm", "WorkRequired", "ServicePerformed"]:
                    print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error querying Pinecone: {e}")

def initialize_pinecone_index() -> None:
    """Initialize a new Pinecone index if it doesn't exist."""
    # Initialize Pinecone with the new API syntax
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index already exists
    existing_indexes = pc.list_indexes()
    
    if PINECONE_INDEX not in [index.name for index in existing_indexes]:
        print(f"Creating new Pinecone index: {PINECONE_INDEX}")
        
        # Create the index - dimensions for text-embedding-3-small model is 1536
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp")
        )
        print(f"Created new index: {PINECONE_INDEX}")
    else:
        print(f"Index {PINECONE_INDEX} already exists")

def main():
    parser = argparse.ArgumentParser(description="Haas Service Assistant Data Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Initialize index command
    init_parser = subparsers.add_parser("init", help="Initialize Pinecone index")
    
    # Import data command
    import_parser = subparsers.add_parser("import", help="Import data from CSV to Pinecone")
    import_parser.add_argument("--file", required=True, help="Path to CSV file with service records")
    import_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for upserts")
    import_parser.add_argument("--namespace", default="", help="Pinecone namespace")
    
    # Test search command
    test_parser = subparsers.add_parser("test", help="Test search functionality")
    test_parser.add_argument("--query", required=True, help="Query text to search for")
    test_parser.add_argument("--top-k", type=int, default=3, help="Number of top results to return")
    
    args = parser.parse_args()
    
    if args.command == "init":
        initialize_pinecone_index()
    
    elif args.command == "import":
        records = load_data_from_csv(args.file)
        if records:
            batch_upsert_to_pinecone(records, args.batch_size, args.namespace)
    
    elif args.command == "test":
        test_search(args.query, args.top_k)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 