"""
Shared model service instance for the application.
This ensures all modules use the same optimized model service.
"""

from .model_service_v2 import EnhancedModelService

# Global model service instance with reduced workers to prevent CPU overload
# Reduced from 16 to 8 workers to balance performance and CPU usage
model_service = EnhancedModelService(
    strategy="pretrained", 
    primary_model="pulk17", 
    max_workers=8  # Reduced workers to prevent 100% CPU usage
)
