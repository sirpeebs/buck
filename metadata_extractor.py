import re
from typing import Dict, Any, List, Optional

class MetadataExtractor:
    """Utility class to extract metadata from technician queries for filtering Pinecone searches."""
    
    # Common Haas CNC models
    MODELS = [
        # VF Series
        "VF-1", "VF-2", "VF-3", "VF-4", "VF-5", "VF-6", "VF-7", "VF-8", "VF-9", "VF-10", "VF-11", 
        "VF-12", "VF-APC", "VF-TR", "VF-SS",
        # ST Series
        "ST-10", "ST-15", "ST-20", "ST-25", "ST-30", "ST-35", "ST-40",
        # DM Series
        "DM-1", "DM-2",
        # UMC Series
        "UMC-500", "UMC-750",
        # Other common models
        "EC-300", "EC-400", "EC-500", "EC-550",
        "DT-1", "DT-2",
        "GR-408", "GR-510", "GR-712"
    ]
    
    # Common serial number patterns
    SERIAL_PATTERNS = [
        r'\b\d{6,9}\b',  # Standard Haas serial with 6-9 digits
        r'\b[A-Z]{1,2}\d{5,8}\b'  # Series code followed by digits
    ]
    
    # Common work tasks/issues
    COMMON_TASKS = [
        "servo", "spindle", "coolant", "overload", "temperature", "axis", "tool changer",
        "alignment", "lubrication", "oil leak", "hydraulic", "pneumatic", "electrical",
        "control", "motor", "drive", "limit switch", "encoder", "bearing", "brake"
    ]
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.model_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(model) for model in self.MODELS) + r')\b', 
            re.IGNORECASE
        )
        self.serial_patterns = [re.compile(pattern) for pattern in self.SERIAL_PATTERNS]
        self.alarm_pattern = re.compile(r'\balarm\s*(\d+)\b', re.IGNORECASE)
        self.task_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(task) for task in self.COMMON_TASKS) + r')\b', 
            re.IGNORECASE
        )
    
    def extract_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata from a technician's query.
        
        Args:
            query: The natural language query from a technician
            
        Returns:
            Dictionary of metadata filters to apply to Pinecone query
        """
        metadata = {}
        
        # Extract model
        model_match = self.model_pattern.search(query)
        if model_match:
            metadata["Model"] = model_match.group(1).upper()
        
        # Extract serial number
        for pattern in self.serial_patterns:
            serial_match = pattern.search(query)
            if serial_match:
                metadata["Serial"] = serial_match.group(0).upper()
                break
        
        # Extract alarm number
        alarm_match = self.alarm_pattern.search(query)
        if alarm_match:
            metadata["Alarm"] = alarm_match.group(1)
        
        # Extract possible work required terms (for semantic search enhancement)
        work_matches = self.task_pattern.findall(query)
        if work_matches:
            # Note: We don't add these directly to metadata filters but use them 
            # to enhance the semantic query
            metadata["Keywords"] = work_matches
        
        return metadata
    
    def enhance_query(self, original_query: str, metadata: Dict[str, Any]) -> str:
        """
        Enhance the original query with extracted metadata for better semantic search.
        
        Args:
            original_query: The original technician query
            metadata: Extracted metadata
            
        Returns:
            Enhanced query for semantic embedding generation
        """
        enhanced_query = original_query
        
        # Add specific terms for better semantic search
        if "Model" in metadata:
            if metadata["Model"] not in original_query:
                enhanced_query += f" {metadata['Model']}"
        
        if "Alarm" in metadata:
            if f"Alarm {metadata['Alarm']}" not in original_query.lower():
                enhanced_query += f" Alarm {metadata['Alarm']}"
        
        if "Keywords" in metadata:
            for kw in metadata["Keywords"]:
                if kw.lower() not in original_query.lower():
                    enhanced_query += f" {kw}"
        
        # Add common troubleshooting terms relevant to CNC machines
        enhanced_query += " troubleshooting repair service maintenance"
        
        return enhanced_query

# Example usage
if __name__ == "__main__":
    extractor = MetadataExtractor()
    
    # Example queries
    test_queries = [
        "I'm troubleshooting Alarm 108 on a VF-3 model. What should I do first?",
        "My ST-20 serial 1234567 has a spindle overload issue",
        "Getting a tool changer error on UMC-750"
    ]
    
    for query in test_queries:
        metadata = extractor.extract_from_query(query)
        enhanced = extractor.enhance_query(query, metadata)
        
        print(f"Original: {query}")
        print(f"Metadata: {metadata}")
        print(f"Enhanced: {enhanced}")
        print("-" * 50) 