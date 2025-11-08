"""Main runner script for BSSC_QA pipeline."""
import sys
from pathlib import Path

# Add src to path
base_path = Path(__file__).parent
sys.path.append(str(base_path / 'bssc_qa' / 'src'))

from core.config import load_config
from core.llm_factory import create_llm
from core.vector_store import VectorStoreManager
from agents.generator_agent import GeneratorAgent
from agents.synthesis_agent import SynthesisAgent
from agents.evaluator_agent import EvaluatorAgent
from tools.retrieval_tool import RetrievalTool
from tools.validation_tool import ValidationTool
from tools.chunk_tool import ChunkAnalysisTool
from pipeline.orchestrator import QAPipelineOrchestrator
from datetime import datetime
from utils.prompt_loader import PromptManager

def main():
    """Run the full BSSC_QA pipeline."""
    print("=" * 70)
    print("BSSC_QA Pipeline - Starting...")
    print("=" * 70)
    
    # Load configuration
    config_path = base_path / 'config.json'
    cfg = load_config(config_path)
    print(f"\n✅ Configuration loaded from {config_path}")

    prompt_manager = PromptManager(
        prompt_path=cfg.prompts.path,
        base_path=base_path
    )
    
    # Initialize vector store
    vs_manager = VectorStoreManager(
        persist_directory=str(base_path / cfg.vector_store.persist_directory),
        collection_name=cfg.vector_store.collection_name,
        embedding_model=cfg.vector_store.embedding_model
    )
    print(f"✅ Vector store initialized ({vs_manager.get_collection_count()} chunks)")
    
    # Initialize tools
    chunk_analyzer = ChunkAnalysisTool()
    retrieval_tool = RetrievalTool(vs_manager)
    validator = ValidationTool(cfg.agents['evaluator']['quality_threshold'])
    print("✅ Tools initialized")
    
    # Create LLMs for each agent
    generator_cfg = cfg.agents['generator']
    generator_provider = generator_cfg['provider']
    generator_provider_cfg = cfg.llm.providers[generator_provider]
    generator_llm = create_llm(
        provider=generator_provider,
        api_key=generator_provider_cfg.resolve_api_key(generator_provider),
        model=generator_provider_cfg.model,
        temperature=generator_provider_cfg.temperature
    )
    
    synthesis_cfg = cfg.agents['synthesis']
    synthesis_provider = synthesis_cfg['provider']
    synthesis_provider_cfg = cfg.llm.providers[synthesis_provider]
    synthesis_llm = create_llm(
        provider=synthesis_provider,
        api_key=synthesis_provider_cfg.resolve_api_key(synthesis_provider),
        model=synthesis_provider_cfg.model,
        temperature=synthesis_provider_cfg.temperature
    )
    
    evaluator_cfg = cfg.agents['evaluator']
    evaluator_provider = evaluator_cfg['provider']
    evaluator_provider_cfg = cfg.llm.providers[evaluator_provider]
    evaluator_llm = create_llm(
        provider=evaluator_provider,
        api_key=evaluator_provider_cfg.resolve_api_key(evaluator_provider),
        model=evaluator_provider_cfg.model,
        temperature=evaluator_provider_cfg.temperature
    )
    
    # Create agents
    generator = GeneratorAgent(generator_llm, retrieval_tool, chunk_analyzer, prompt_manager)
    synthesizer = SynthesisAgent(
        synthesis_llm,
        vs_manager,
        synthesis_cfg['max_evidence_spans'],
        prompt_manager
    )
    evaluator = EvaluatorAgent(
        evaluator_llm,
        validator,
        evaluator_cfg['quality_threshold'],
        prompt_manager
    )
    print("✅ Agents initialized")
    
    # Create orchestrator
    orchestrator = QAPipelineOrchestrator(
        generator, synthesizer, evaluator, vs_manager, cfg
    )
    print("✅ Orchestrator ready")
    
    # Run pipeline
    print("\n" + "=" * 70)
    print("Starting QA Generation...")
    print("=" * 70)
    
    results = orchestrator.generate_qa_from_chunks(
        num_chunks=20,
        questions_per_chunk=3
    )
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = base_path / 'data' / 'output' / f'qa_dataset_{timestamp}.json'
    orchestrator.export_results(results, str(output_file))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Total QA pairs: {results['total_qa_pairs']}")
    print(f"  Passed: {results['passed_qa_pairs']}")
    print(f"  Pass rate: {results['statistics'].get('pass_rate', 0):.1%}")
    print(f"  Average score: {results['statistics'].get('average_score', 0):.2f}")
    print(f"\nOutput: {output_file}")

if __name__ == "__main__":
    main()
