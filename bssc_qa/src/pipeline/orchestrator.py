"""Pipeline Orchestrator for BSSC_QA."""
from typing import List, Dict, Any
from tqdm import tqdm
import json

class QAPipelineOrchestrator:
    """Orchestrates the full QA generation pipeline."""
    
    def __init__(self, generator_agent, synthesis_agent, evaluator_agent,
                 vector_store_manager, config):
        self.generator = generator_agent
        self.synthesizer = synthesis_agent
        self.evaluator = evaluator_agent
        self.vs_manager = vector_store_manager
        self.config = config
    
    def generate_qa_from_chunks(self, num_chunks: int = 10, 
                               questions_per_chunk: int = 3) -> Dict[str, Any]:
        """Generate QA pairs from stored chunks.
        
        Args:
            num_chunks: Number of chunks to process
            questions_per_chunk: Questions to generate per chunk
            
        Returns:
            Dictionary with results and statistics
        """
        results = {
            'total_chunks': num_chunks,
            'total_questions_attempted': 0,
            'total_qa_pairs': 0,
            'passed_qa_pairs': 0,
            'failed_qa_pairs': 0,
            'qa_pairs': [],
            'statistics': {}
        }
        
        # Get chunks from vector store
        print(f"Retrieving {num_chunks} chunks...")
        chunks = self._get_sample_chunks(num_chunks)
        
        print(f"\nProcessing {len(chunks)} chunks...")
        
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Generating QA")):
            try:
                # Step 1: Generate questions
                questions = self.generator.generate_questions(
                    chunk.page_content,
                    count=questions_per_chunk
                )
                
                results['total_questions_attempted'] += len(questions)
                
                # Step 2: Synthesize answers
                for question in questions:
                    qa_pair = self.synthesizer.synthesize_answer(
                        question['question'],
                        question['question_type']
                    )
                    
                    # Step 3: Evaluate quality
                    evaluated_qa = self.evaluator.evaluate_qa(qa_pair)
                    
                    # Store results
                    results['total_qa_pairs'] += 1
                    if evaluated_qa['passed']:
                        results['passed_qa_pairs'] += 1
                        results['qa_pairs'].append(evaluated_qa)
                    else:
                        results['failed_qa_pairs'] += 1
                    
            except Exception as e:
                print(f"\nError processing chunk {chunk_idx}: {e}")
                continue
        
        # Calculate statistics
        results['statistics'] = self._calculate_statistics(results)
        
        return results
    
    def _get_sample_chunks(self, num_chunks: int) -> List:
        """Get sample chunks from vector store."""
        # Use diverse queries to get varied chunks
        queries = [
            "travel destination",
            "historical place",
            "tourist attraction",
            "cultural site",
            "natural beauty"
        ]
        
        chunks = []
        chunks_per_query = max(num_chunks // len(queries), 1)
        
        for query in queries:
            results = self.vs_manager.similarity_search(query, k=chunks_per_query)
            chunks.extend(results)
            if len(chunks) >= num_chunks:
                break
        
        return chunks[:num_chunks]
    
    def _calculate_statistics(self, results: Dict) -> Dict[str, Any]:
        """Calculate pipeline statistics."""
        stats = {}
        
        if results['total_qa_pairs'] > 0:
            stats['pass_rate'] = results['passed_qa_pairs'] / results['total_qa_pairs']
            
            # Average scores
            if results['qa_pairs']:
                avg_score = sum(qa['overall_score'] for qa in results['qa_pairs']) / len(results['qa_pairs'])
                stats['average_score'] = avg_score
                
                # Question type distribution
                type_dist = {}
                for qa in results['qa_pairs']:
                    qtype = qa.get('question_type', 'unknown')
                    type_dist[qtype] = type_dist.get(qtype, 0) + 1
                stats['question_type_distribution'] = type_dist
        
        return stats
    
    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export QA pairs to JSON file."""
        export_data = {
            'metadata': {
                'total_qa_pairs': results['passed_qa_pairs'],
                'statistics': results['statistics']
            },
            'qa_pairs': [
                {
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'evidence': qa['evidence_spans'],
                    'scores': qa['scores'],
                    'overall_score': qa['overall_score']
                }
                for qa in results['qa_pairs']
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results exported to: {output_path}")
