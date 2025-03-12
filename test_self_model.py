import asyncio
import logging
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

async def test_reflection():
    # Create an instance of the self model
    self_model = LucidiaSelfModel()
    
    # Test the reflect method with focus areas
    reflection_results = await self_model.reflect(["performance", "improvement"])
    
    # Display results
    print("\nReflection Results:")
    print(f"Timestamp: {reflection_results['timestamp']}")
    print(f"Focus Areas: {reflection_results['focus_areas']}")
    print(f"Spiral Progress: {reflection_results['spiral_progress']} -> {self_model.self_awareness['current_spiral_position']}")
    print("\nInsights:")
    for insight in reflection_results['insights']:
        print(f"- [{insight['type']}] {insight['content']} (Significance: {insight['significance']})")
    
    print("\nAdaptations:")
    for adaptation in reflection_results['adaptations']:
        print(f"- [{adaptation['area']}] {adaptation['change']} (Confidence: {adaptation['confidence']})")
    
    print(f"\nSelf-awareness level: {self_model.self_awareness['current_level']:.2f}")
    print(f"Spiral depth: {self_model.self_awareness['spiral_depth']:.2f}")
    print(f"Cycles completed: {self_model.self_awareness['cycles_completed']}")

if __name__ == "__main__":
    asyncio.run(test_reflection())
