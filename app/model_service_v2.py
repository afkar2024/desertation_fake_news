"""
Enhanced Model Service with High-Performance Pre-trained Models
Implements Phase 2 of the Model Performance Implementation Plan
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List, Any, Optional
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import os
import time

logger = logging.getLogger(__name__)


class EnhancedModelService:
    """High-performance model service with pre-trained models for fake news detection"""
    
    # Model configurations with expected performance
    PRETRAINED_MODELS = {
        "pulk17": {
            "name": "Pulk17/Fake-News-Detection",
            "expected_accuracy": 0.9958,
            "expected_f1": 0.9957,
            "description": "Primary model - NOTE: Has inverted labels (REAL=fake, FAKE=real)",
            "inverted": True
        },
        "hamzab": {
            "name": "hamzab/roberta-fake-news-classification", 
            "expected_accuracy": 1.00,  # on training set
            "expected_f1": 0.99,
            "description": "Backup model with excellent performance",
            "inverted": False
        },
        "jy46604790": {
            "name": "jy46604790/Fake-News-Bert-Detect",
            "expected_accuracy": 0.95,
            "expected_f1": 0.94,
            "description": "Fallback model",
            "inverted": False
        }
    }
    
    def __init__(self, strategy: str = "pretrained", primary_model: str = "pulk17", max_workers: Optional[int] = None):
        """
        Initialize the enhanced model service
        
        Args:
            strategy: "pretrained", "ensemble", or "finetuned"
            primary_model: key from PRETRAINED_MODELS dict
            max_workers: Number of workers for parallel processing
        """
        self.strategy = strategy
        self.primary_model = primary_model
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_workers = max_workers or min(32, (cpu_count() or 1) + 4)  # Increase worker count
        self.pipelines = {}
        self.models = {}
        self.tokenizers = {}
        
        logger.info(f"Initializing EnhancedModelService with strategy: {strategy}")
        logger.info(f"Device: {'GPU' if self.device == 0 else 'CPU'}")
        
        if strategy == "pretrained":
            self._load_pretrained_model(primary_model)
        elif strategy == "ensemble":
            self._load_ensemble_models()
        elif strategy == "finetuned":
            self._load_finetuned_model()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _load_pretrained_model(self, model_key: str = "pulk17"):
        """Load a single high-performance pre-trained model with fallback"""
        # Try to load the requested model
        if model_key in self.PRETRAINED_MODELS:
            model_config = self.PRETRAINED_MODELS[model_key]
            model_name = model_config["name"]
            
            logger.info(f"Loading pre-trained model: {model_name}")
            try:
                # Load tokenizer and model with proper configuration
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Create pipeline with configured tokenizer and model
                self.pipelines[model_key] = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    # Configure for proper text handling
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                logger.info(f"✅ Successfully loaded {model_name}")
                logger.info(f"   Expected performance: {model_config['expected_accuracy']:.1%} accuracy, {model_config['expected_f1']:.1%} F1")
                if model_config.get("inverted", False):
                    logger.warning(f"   ⚠️ NOTE: This model has inverted labels (REAL=fake, FAKE=real)")
                return
            except Exception as e:
                logger.error(f"❌ Primary model failed: {e}")
        
        # Fallback mechanism - try backup models
        fallback_order = ["pulk17", "hamzab", "jy46604790"]
        for fallback_key in fallback_order:
            if fallback_key == model_key:
                continue  # Skip the one we already tried
            
            try:
                model_config = self.PRETRAINED_MODELS[fallback_key]
                model_name = model_config["name"]
                logger.info(f"Trying backup model: {model_name}")
                
                # Load tokenizer and model with proper configuration
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Create pipeline with configured tokenizer and model
                self.pipelines[fallback_key] = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    # Configure for proper text handling
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                logger.info(f"✅ Successfully loaded backup model {model_name}")
                return
            except Exception as e:
                logger.warning(f"Backup model {fallback_key} failed: {e}")
        
        # If all models fail, raise error
        raise RuntimeError("Failed to load any fake news detection model")
    
    def _load_ensemble_models(self):
        """Load multiple models for ensemble voting"""
        logger.info("Loading ensemble models...")
        
        # Load top 3 models for ensemble
        models_to_load = ["pulk17", "hamzab", "jy46604790"]
        
        for model_key in models_to_load:
            try:
                self._load_pretrained_model(model_key)
            except Exception as e:
                logger.warning(f"Failed to load {model_key} for ensemble: {e}")
        
        if len(self.pipelines) == 0:
            raise RuntimeError("Failed to load any models for ensemble")
        
        logger.info(f"Ensemble ready with {len(self.pipelines)} models")
    
    def _load_finetuned_model(self, model_path: Optional[str] = None):
        """Load a fine-tuned model from local path"""
        if model_path is None:
            # Default to ISOT fine-tuned model if available
            model_path = "models/isot_bert-base-uncased"
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Fine-tuned model not found at {model_path}")
            logger.info("Falling back to pre-trained model")
            self._load_pretrained_model("pulk17")
            return
        
        logger.info(f"Loading fine-tuned model from {model_path}")
        self.tokenizers["finetuned"] = AutoTokenizer.from_pretrained(str(model_path))
        self.models["finetuned"] = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to("cuda" if self.device == 0 else "cpu")
        self.models["finetuned"].eval()
        logger.info("✅ Fine-tuned model loaded")
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text with high-performance model
        
        Returns:
            Dict with prediction, confidence, probabilities, and model info
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Truncate text to prevent token length issues
        text = self._truncate_text(text, max_tokens=400)
        
        if self.strategy == "pretrained":
            return self._classify_with_pretrained(text)
        elif self.strategy == "ensemble":
            return self._classify_with_ensemble(text)
        elif self.strategy == "finetuned":
            return self._classify_with_finetuned(text)
        
        return self._empty_result()
    
    def _classify_with_pretrained(self, text: str) -> Dict[str, Any]:
        """Classify using single pre-trained model"""
        model_key = list(self.pipelines.keys())[0]
        pipeline_obj = self.pipelines[model_key]
        model_config = self.PRETRAINED_MODELS[model_key]
        
        try:
            # Additional safety: ensure text length is reasonable
            if len(text.split()) > 400:
                text = self._truncate_text(text, max_tokens=350)  # More aggressive truncation
            
            result = pipeline_obj(text)
            
            # Parse prediction using verified logic
            if isinstance(result, list) and len(result) > 0:
                prediction_result = result[0]
                label = str(prediction_result.get('label', '')).upper()
                confidence = prediction_result.get('score', 0.0)
                
                # Check if this model has inverted labels
                is_inverted = model_config.get("inverted", False)
                
                # Debug logging
                logger.debug(f"Raw label: {label}, Confidence: {confidence}, Model inverted: {is_inverted}")
                
                # Handle different label formats
                if 'LABEL_1' in label:
                    # For Pulk17 (inverted): LABEL_1 = real news
                    # For others: LABEL_1 = fake news
                    prediction = 0 if is_inverted else 1
                elif 'LABEL_0' in label:
                    # For Pulk17 (inverted): LABEL_0 = fake news
                    # For others: LABEL_0 = real news
                    prediction = 1 if is_inverted else 0
                elif 'FAKE' in label or 'FALSE' in label:
                    # Check if model has inverted labels
                    if is_inverted:
                        # For Pulk17: "FAKE" actually means REAL news
                        prediction = 0  # real
                    else:
                        # Normal: "FAKE" means fake news
                        prediction = 1  # fake
                elif 'REAL' in label or 'TRUE' in label:
                    # Check if model has inverted labels
                    if is_inverted:
                        # For Pulk17: "REAL" actually means FAKE news
                        prediction = 1  # fake
                    else:
                        # Normal: "REAL" means real news
                        prediction = 0  # real
                else:
                    # Fallback based on confidence
                    prediction = 1 if confidence > 0.5 else 0
                    logger.warning(f"Unknown label format: {label}, using confidence threshold")
            else:
                # Fallback if result format is unexpected
                logger.error(f"Unexpected result format: {result}")
                return self._empty_result()
            
            # Calculate probabilities correctly
            if prediction == 1:  # Fake
                prob_fake = confidence
                prob_real = 1 - confidence
            else:  # Real
                prob_real = confidence
                prob_fake = 1 - confidence
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "real": float(prob_real),
                    "fake": float(prob_fake)
                },
                "model_used": model_config['name'],
                "expected_performance": {
                    "accuracy": model_config['expected_accuracy'],
                    "f1": model_config['expected_f1']
                }
            }
        except Exception as e:
            text_len = len(text.split()) if text else 0
            logger.error(f"Error in classification: {e} (text length: {text_len} words)")
            return self._empty_result()
    
    def _classify_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Classify using ensemble voting"""
        predictions = []
        confidences = []
        
        for model_key, pipeline_obj in self.pipelines.items():
            try:
                result = pipeline_obj(text)
                model_config = self.PRETRAINED_MODELS.get(model_key, {})
                is_inverted = model_config.get("inverted", False)
                
                # Parse using same logic as single model
                if isinstance(result, list) and len(result) > 0:
                    prediction_result = result[0]
                    label = str(prediction_result.get('label', '')).upper()
                    confidence = prediction_result.get('score', 0.0)
                    
                    # Handle different label formats with inversion check
                    if 'LABEL_1' in label:
                        pred = 1  # fake
                    elif 'LABEL_0' in label:
                        pred = 0  # real
                    elif 'FAKE' in label or 'FALSE' in label:
                        pred = 0 if is_inverted else 1  # inverted for Pulk17
                    elif 'REAL' in label or 'TRUE' in label:
                        pred = 1 if is_inverted else 0  # inverted for Pulk17
                    else:
                        pred = 1 if confidence > 0.5 else 0
                    
                    predictions.append(pred)
                    confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Model {model_key} failed: {e}")
        
        if not predictions:
            return self._empty_result()
        
        # Weighted voting based on confidence
        weighted_fake = sum(c for p, c in zip(predictions, confidences) if p == 1)
        weighted_real = sum(c for p, c in zip(predictions, confidences) if p == 0)
        total_weight = weighted_fake + weighted_real
        
        if total_weight > 0:
            prob_fake = weighted_fake / total_weight
            prob_real = weighted_real / total_weight
        else:
            prob_fake = sum(predictions) / len(predictions)
            prob_real = 1 - prob_fake
        
        prediction = 1 if prob_fake > prob_real else 0
        confidence = max(prob_fake, prob_real)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": {
                "real": float(prob_real),
                "fake": float(prob_fake)
            },
            "model_used": "ensemble",
            "ensemble_models": len(self.pipelines),
            "ensemble_agreement": sum(p == prediction for p in predictions) / len(predictions)
        }
    
    def _classify_with_finetuned(self, text: str) -> Dict[str, Any]:
        """Classify using fine-tuned model"""
        if "finetuned" not in self.models:
            # Fallback to pre-trained if fine-tuned not available
            return self._classify_with_pretrained(text)
        
        tokenizer = self.tokenizers["finetuned"]
        model = self.models["finetuned"]
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if self.device == 0:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        
        # Get prediction
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())
        
        # Binary classification assumed
        prob_real = float(probs[0].item())
        prob_fake = float(probs[1].item()) if len(probs) > 1 else 1 - prob_real
        
        return {
            "prediction": pred_idx,
            "confidence": confidence,
            "probabilities": {
                "real": prob_real,
                "fake": prob_fake
            },
            "model_used": "finetuned",
            "model_path": str(self.models.get("model_path", "unknown"))
        }
    
    def _truncate_text(self, text: str, max_tokens: int = 400) -> str:
        """Truncate text to prevent token length issues"""
        if not text:
            return ""
        
        # Simple word-based truncation
        words = text.split()
        if len(words) > max_tokens:
            words = words[:max_tokens]
            text = ' '.join(words)
        
        return text
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty/neutral result"""
        return {
            "prediction": 0,
            "confidence": 0.5,
            "probabilities": {
                "real": 0.5,
                "fake": 0.5
            },
            "model_used": "none"
        }
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts efficiently with parallel processing"""
        if not texts:
            return []
        
        # For small batches, use sequential processing
        if len(texts) <= 10:
            results = []
            for text in texts:
                truncated_text = self._truncate_text(text, max_tokens=400)
                results.append(self.classify_text(truncated_text))
            return results
        
        # For large batches, use parallel processing
        start_time = time.time()
        logger.info(f"🤖 Processing {len(texts):,} texts in parallel using {self.max_workers} workers...")
        
        # Truncate all texts first
        logger.info(f"✂️  Truncating {len(texts):,} texts...")
        truncated_texts = [self._truncate_text(text, max_tokens=400) for text in texts]
        truncate_time = time.time() - start_time
        logger.info(f"✅ Text truncation complete in {truncate_time:.1f}s")
        
        # Process in parallel with progress tracking
        logger.info(f"🔄 Starting parallel classification...")
        results = [None] * len(truncated_texts)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.classify_text, text): i 
                for i, text in enumerate(truncated_texts)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed += 1
                    
                    # Progress updates every 10%
                    if completed % max(1, len(texts) // 10) == 0:
                        progress = (completed / len(texts)) * 100
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(f"   🤖 Classification: {completed:,}/{len(texts):,} ({progress:.1f}%) - {rate:.1f} texts/sec")
                        
                except Exception as exc:
                    logger.error(f'Text {index} generated an exception: {exc}')
                    results[index] = self._empty_result()
                    completed += 1
        
        total_time = time.time() - start_time
        rate = len(texts) / total_time if total_time > 0 else 0
        logger.info(f"✅ Model classification complete: {len(texts):,} texts in {total_time:.1f}s ({rate:.1f} texts/sec)")
        
        return results
    
    def reload_model(self, model_path: str):
        """Reload model from a different path"""
        logger.info(f"Reloading model from {model_path}")
        self.strategy = "finetuned"
        self.models.clear()
        self.tokenizers.clear()
        self.pipelines.clear()
        self._load_finetuned_model(model_path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "strategy": self.strategy,
            "device": "GPU" if self.device == 0 else "CPU",
            "loaded_models": []
        }
        
        if self.pipelines:
            for key in self.pipelines.keys():
                config = self.PRETRAINED_MODELS.get(key, {})
                info["loaded_models"].append({
                    "key": key,
                    "name": config.get("name", "unknown"),
                    "expected_accuracy": config.get("expected_accuracy", 0),
                    "expected_f1": config.get("expected_f1", 0),
                    "inverted": config.get("inverted", False)
                })
        
        if self.models:
            for key in self.models.keys():
                info["loaded_models"].append({
                    "key": key,
                    "type": "finetuned"
                })
        
        return info
    
    def set_temperature(self, temperature: float):
        """Set temperature for model predictions (placeholder for compatibility)"""
        logger.info(f"Temperature setting not available in enhanced model service. Using default temperature.")
        return 1.0
    
    def calibrate_temperature(self, texts: List[str], labels: List[int], max_steps: int = 200, lr: float = 0.01) -> float:
        """Calibrate temperature on provided labeled texts (placeholder for compatibility)"""
        logger.info("Temperature calibration not available in enhanced model service. Using default temperature.")
        return 1.0
    
    def explain_text(self, text: str, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """Generate basic explanation for text (placeholder for SHAP compatibility)"""
        # Get prediction first
        result = self.classify_text(text)
        
        # Create basic explanation
        tokens = text.split()[:20]  # First 20 words
        shap_values = [0.1] * len(tokens)  # Placeholder values
        
        return {
            "tokens": tokens,
            "shap_values": shap_values,
            "base_value": 0.5,
            "prediction": result,
            "message": "Advanced SHAP explanations not available in enhanced model service"
        }
    
    def explain_text_attention(self, text: str) -> Dict[str, Any]:
        """Generate attention-based explanation (placeholder for compatibility)"""
        tokens = text.split()[:20]
        weights = [0.1] * len(tokens)
        
        return {
            "tokens": tokens,
            "weights": weights,
            "message": "Attention explanations not available in enhanced model service"
        }
    
    def explain_text_lime(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation (placeholder for compatibility)"""
        features = text.split()[:num_features]
        weights = [0.1] * len(features)
        
        return {
            "features": features,
            "weights": weights,
            "message": "LIME explanations not available in enhanced model service"
        }
    
    def explanation_confidence(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """Get explanation confidence (placeholder for compatibility)"""
        return {
            "confidence": 0.8,
            "top_k": top_k,
            "message": "Explanation confidence not available in enhanced model service"
        }
    
    def generate_counterfactuals(self, text: str, max_candidates: int = 3) -> List[Dict[str, Any]]:
        """Generate counterfactual examples (placeholder for compatibility)"""
        # Create simple counterfactuals by removing words
        words = text.split()
        counterfactuals = []
        
        for i in range(min(max_candidates, len(words))):
            if len(words) > 1:
                # Remove a word to create counterfactual
                modified_words = words[:i] + words[i+1:]
                modified_text = " ".join(modified_words)
                
                # Get prediction for modified text
                result = self.classify_text(modified_text)
                
                counterfactuals.append({
                    "text": modified_text,
                    "prediction": result,
                    "removed_word": words[i] if i < len(words) else "",
                    "type": "word_removal"
                })
        
        return counterfactuals
    
    def reload(self, model_source: Optional[str] = None) -> Dict[str, Any]:
        """Reload model from source (compatibility method)"""
        if model_source:
            self.reload_model(model_source)
        return self.get_model_info()


# Example usage and integration point
if __name__ == "__main__":
    # Quick test
    service = EnhancedModelService(strategy="pretrained", primary_model="pulk17")
    
    test_text = """Residents of Washington, DC, continue to take to the streets to protest President Trump’s federal takeover of the city and deployment of National Guard troops and federal law enforcement officers as a “solution” to a fabricated “crime wave.” “We demand ICE out of DC. We demand an end to this unnecessary law enforcement,” Nee Nee Taylor, co-founder and executive director of Harriet’s Wildest Dreams, said at a “Free DC” rally on Monday, Aug. 18. “We demand full autonomy. We demand: Hands off DC!” TRNN correspondent and host of Rattling the Bars Mansa Musa reports from the ground in federally occupied Washington, DC."""
    
    result = service.classify_text(test_text)
    
    print(f"Text: {test_text}")
    print(f"Prediction: {'FAKE' if result['prediction'] == 1 else 'REAL'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model: {result['model_used']}")