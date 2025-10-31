"""Chunk analysis tool for content understanding."""
from typing import List, Dict, Any
import re

class ChunkAnalysisTool:
    """Tool for analyzing text chunks."""
    
    def analyze_chunk(self, chunk_text: str) -> Dict[str, Any]:
        """Analyze a text chunk to extract key information.
        
        Args:
            chunk_text: Text content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Extract sentences across English and Bengali punctuation
        sentences = re.split(r'(?<=[.!?।])\s+', chunk_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract potential entities leveraging Latin capitalization and Bengali script
        entity_pattern = re.compile(r'\b(?:[A-Z][a-zA-Z]+|[\u0980-\u09FF]{2,})\b')
        entities = {
            match.group(0).strip('.,!?;:।')
            for match in entity_pattern.finditer(chunk_text)
        }
        
        # Extract numbers (potential facts)
        numbers = re.findall(r'\d+', chunk_text)
        
        # Identify question indicators
        question_words = [
            'what', 'when', 'where', 'who', 'why', 'how',
            'কী', 'কি', 'কখন', 'কোথায়', 'কোথায়', 'কার', 'কারা', 'কেন', 'কিভাবে', 'কে'
        ]
        potential_questions = sum(1 for word in question_words 
                                 if word in chunk_text.lower())
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(chunk_text.split()),
            'entities': list(entities)[:10],  # Top 10
            'has_numbers': len(numbers) > 0,
            'number_count': len(numbers),
            'potential_topics': len(entities),
            'question_potential': potential_questions > 0
        }
    
    def suggest_question_types(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest appropriate question types based on chunk analysis.
        
        Args:
            analysis: Output from analyze_chunk
            
        Returns:
            List of suggested question types
        """
        suggestions = []
        
        # Factual questions if entities present
        if analysis['potential_topics'] > 2:
            suggestions.append('factual')
        
        # Numerical questions if numbers present
        if analysis['has_numbers']:
            suggestions.append('quantitative')
        
        # Conceptual questions if substantial content
        if analysis['sentence_count'] >= 5:
            suggestions.append('conceptual')
        
        # Analytical questions if complex content
        if analysis['word_count'] > 100:
            suggestions.append('analytical')
        
        return suggestions if suggestions else ['factual']
