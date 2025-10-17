import logging
import os
from PIL import Image

import numpy as np

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.utils.torch import ClassifierOutputProcessor

import torch
import torch.nn.functional as F

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)


class ColModernVBertConfig(fout.TorchImageModelConfig):
    """
    Config class for ColModernVBert.
    
    ColModernVBert is a multi-vector retrieval model that generates variable-length
    ColBERT-style embeddings (Nx128) for both images and text, enabling fine-grained
    visual document retrieval and zero-shot classification.
    
    Args:
        model_path (str): HuggingFace model ID. Default: "ModernVBERT/colmodernvbert"
        
        text_prompt (str): Optional baseline text prompt for classification. Default: ""
        
        pooling_strategy (str): Final pooling strategy for multi-vector to 1D conversion.
            Options: "mean" (default) or "max".
            - "mean": Average pooling, good for holistic document matching
            - "max": Max pooling, good for specific content/keyword matching
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        # Processor handles preprocessing, so use raw inputs
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        
        # Only set up output processor if classes provided (for classification)
        if "classes" in d and d["classes"] is not None and len(d["classes"]) > 0:
            if "output_processor_cls" not in d:
                d["output_processor_cls"] = "fiftyone.utils.torch.ClassifierOutputProcessor"
        
        super().__init__(d)
        
        # ColModernVBert-specific configuration
        self.model_path = self.parse_string(d, "model_path", default="ModernVBERT/colmodernvbert")
        self.text_prompt = self.parse_string(d, "text_prompt", default="")
        self.pooling_strategy = self.parse_string(d, "pooling_strategy", default="mean")
        
        # Validate pooling strategy
        if self.pooling_strategy not in ["mean", "max"]:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'max', got '{self.pooling_strategy}'"
            )


class ColModernVBertModel(fout.TorchImageModel, fom.PromptMixin):
    """
    ColModernVBert model for document understanding and retrieval.
    
    This model supports two workflows:
    
    1. **Retrieval/Similarity Search**: Returns 128-dim pooled embeddings
       - Multi-vectors → final pooling (mean/max) → 128-dim
       - Use with compute_embeddings() and compute_similarity()
       - Efficient for large-scale search
       
    2. **Zero-Shot Classification**: Uses variable-length multi-vector embeddings
       - Use with apply_model()
       - MaxSim scoring for fine-grained classification
       - Higher accuracy than pooled embeddings
    
    Unlike ColPali, ColModernVBert outputs pre-compressed 128-dim vectors,
    so no token pooling is required.
    
    The model extends TorchImageModel for image processing and PromptMixin for text embedding.
    """
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A ColModernVBertConfig instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Storage for cached data
        self._text_features = None  # Cached multi-vector text features for classification
        self._last_computed_embeddings = None  # Last computed 128-dim pooled embeddings
        

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings."""
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts."""
        return True
    
    @property
    def classes(self):
        """The list of class labels for the model."""
        return self._classes

    @classes.setter
    def classes(self, value):
        """Set new classes and invalidate cached text features."""
        self._classes = value
        self._text_features = None  # Invalidate cache
        
        # Rebuild output processor if classes are provided
        if value is not None and len(value) > 0:
            self._output_processor = ClassifierOutputProcessor(classes=value)
        else:
            self._output_processor = None
    
    @property
    def text_prompt(self):
        """The text prompt prefix for classification."""
        return self.config.text_prompt

    @text_prompt.setter  
    def text_prompt(self, value):
        """Set new text prompt and invalidate cached text features."""
        self.config.text_prompt = value
        self._text_features = None  # Invalidate cache
    
    def _load_model(self, config):
        """Load ColModernVBert model and processor from HuggingFace.
        
        Args:
            config: ColModernVBertConfig instance containing model parameters

        Returns:
            model: The loaded model
        """
        
        
        logger.info(f"Loading ColModernVBert model from {config.model_path}")
        
        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self._device)
            
            # Use bfloat16 for Ampere or newer GPUs (capability >= 8.0)
            if capability[0] >= 8:
                model_kwargs["dtype"] = torch.bfloat16
            else:
                model_kwargs["dtype"] = torch.float16

        # Enable flash attention if available
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        # Load processor
        self.processor = ColModernVBertProcessor.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = ColModernVBert.from_pretrained(
            config.model_path,
            trust_remote_code=True,

        )
        
        self.model.to(self._device)
        self.model.eval()
        
        return self.model

    def _prepare_images_for_processor(self, imgs):
        """Convert images to PIL format (processor's expected input).
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            List of PIL Images
        """
        pil_images = []
        
        for img in imgs:
            if isinstance(img, Image.Image):
                # Already PIL Image
                pil_images.append(img)
            elif isinstance(img, torch.Tensor):
                # Tensor (CHW) → PIL Image
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    # Assume normalized [0, 1] or [-1, 1]
                    if img_np.min() < 0:
                        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                # Numpy array (HWC) → PIL Image
                if img.dtype != np.uint8:
                    # Assume normalized [0, 1] or [0, 255]
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        return pil_images

    def _apply_final_pooling(self, multi_vector_embeddings):
        """Apply final pooling to multi-vector embeddings to get fixed 1D vectors.
        
        Args:
            multi_vector_embeddings: Tensor of shape (batch, num_vectors, 128)
            
        Returns:
            Tensor of shape (batch, 128)
        """
        if self.config.pooling_strategy == "mean":
            # Mean pooling across vectors
            pooled = multi_vector_embeddings.mean(dim=1)
        elif self.config.pooling_strategy == "max":
            # Max pooling across vectors
            pooled = multi_vector_embeddings.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
        
        return pooled

    def _get_text_features(self):
        """Get or compute multi-vector text features for classification.
        
        Creates embeddings for each class by combining text_prompt with class names.
        
        Returns:
            List of multi-vector text embeddings [(num_vectors, 128), ...] - one per class
        """
        if self._text_features is None:
            # Create prompts for each class
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            # Compute and cache multi-vector text features
            self._text_features = self._embed_prompts_multivector(prompts)
        
        return self._text_features
    
    def _embed_prompts_multivector(self, prompts):
        """Embed text prompts using processor and model (returns multi-vectors).
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            List of multi-vector embeddings [(num_vectors, 128), ...]
        """
        # Process texts through processor
        text_inputs = self.processor.process_texts(prompts)
        
        # Move to device
        text_inputs = {k: v.to(self._device) for k, v in text_inputs.items()}
        
        # Get multi-vector embeddings
        with torch.no_grad():
            embeddings = self.model(**text_inputs)
            # embeddings shape: (batch, num_vectors, 128)
        
        # Convert to list of tensors (one per prompt)
        embeddings_list = [embeddings[i] for i in range(embeddings.shape[0])]
        
        return embeddings_list

    def embed_prompt(self, prompt):
        """Embed a single text prompt to 128-dim vector for retrieval.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: 128-dim embedding vector
        """
        embeddings = self._embed_prompts_multivector([prompt])
        # embeddings is list of [(num_vectors, 128)]
        
        # Apply final pooling to get 1D vector
        multi_vector = embeddings[0].unsqueeze(0)  # (1, num_vectors, 128)
        pooled = self._apply_final_pooling(multi_vector)  # (1, 128)
        
        result = pooled[0].cpu().numpy()
        return result

    def embed_prompts(self, prompts):
        """Embed multiple text prompts to 128-dim vectors for retrieval.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: 128-dim embeddings with shape (batch, 128)
        """
        embeddings_list = self._embed_prompts_multivector(prompts)
        
        # Stack into tensor for pooling
        # Note: Assumes all prompts have same number of vectors (padded)
        multi_vectors = torch.stack(embeddings_list)  # (batch, num_vectors, 128)
        
        # Apply final pooling
        pooled = self._apply_final_pooling(multi_vectors)  # (batch, 128)
        
        result = pooled.cpu().numpy()
        return result

    def embed_images(self, imgs):
        """Embed images to 128-dim vectors for retrieval/similarity search.
        
        Uses multi-vector embeddings with final pooling to get fixed-dimension vectors.
        
        Args:
            imgs: List of images (PIL, numpy arrays, or tensors)
            
        Returns:
            numpy array: 128-dim embeddings with shape (batch, 128)
        """
        # Convert to PIL images
        pil_images = self._prepare_images_for_processor(imgs)
        
        # Process images through processor
        image_inputs = self.processor.process_images(pil_images)
        
        # Move to device
        image_inputs = {k: v.to(self._device) for k, v in image_inputs.items()}
        
        # Get multi-vector embeddings
        with torch.no_grad():
            multi_vector_embeddings = self.model(**image_inputs)
            # Shape: (batch, num_vectors, 128)
            
            # Apply final pooling to get 1D vectors
            pooled_embeddings = self._apply_final_pooling(multi_vector_embeddings)
            # Shape: (batch, 128)
            
            # Cache for get_embeddings()
            self._last_computed_embeddings = pooled_embeddings
        
        return pooled_embeddings.cpu().numpy()
    
    def embed(self, img):
        """Embed a single image.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: 128-dim embedding
        """
        embeddings = self.embed_images([img])
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed
            
        Returns:
            numpy array: 128-dim embeddings
        """
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed 128-dim pooled embeddings.
        
        Returns:
            numpy array: The last computed embeddings with shape (batch, 128)
        """
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
        
        result = self._last_computed_embeddings.cpu().numpy()
        return result

    def _get_class_logits(self, text_features, image_features):
        """Calculate multi-vector similarity scores using MaxSim.
        
        Uses the processor's built-in MaxSim scoring for efficient computation.
        
        Args:
            text_features: List of multi-vector text embeddings
                [(num_text_vectors, 128), ...] - one per class
            image_features: List of multi-vector image embeddings
                [(num_image_vectors, 128), ...] - one per image
            
        Returns:
            tuple: (logits_per_image, logits_per_text)
                - logits_per_image: shape (num_images, num_classes)
                - logits_per_text: shape (num_classes, num_images)
        """
        with torch.no_grad():
            # Use processor's built-in MaxSim scoring
            logits_per_text = self.processor.score(
                text_features,  # List of (num_vectors, 128) tensors
                image_features,  # List of (num_vectors, 128) tensors
                device=self._device
            )
            # Returns: (num_classes, num_images)
            
            logits_per_image = logits_per_text.t()
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run zero-shot classification on a batch of images.
        
        Uses multi-vector similarity with MaxSim scoring between image and class text embeddings.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            Classification predictions processed by output processor
        """
        # Check if classification is supported
        if self.classes is None or len(self.classes) == 0:
            raise ValueError(
                "Cannot perform classification without classes. "
                "Set classes when loading: foz.load_zoo_model(..., classes=['class1', 'class2'])"
            )
        
        if self._output_processor is None:
            raise ValueError(
                "No output processor configured for classification."
            )
        
        # Convert to PIL images
        pil_images = self._prepare_images_for_processor(imgs)
        
        # Process images through processor
        image_inputs = self.processor.process_images(pil_images)
        
        # Move to device
        image_inputs = {k: v.to(self._device) for k, v in image_inputs.items()}
        
        # Get multi-vector image embeddings
        with torch.no_grad():
            image_embeddings = self.model(**image_inputs)
            # Shape: (batch, num_vectors, 128)
        
        # Convert to list of tensors for MaxSim scoring
        image_features = [image_embeddings[i] for i in range(image_embeddings.shape[0])]
        
        # Get cached multi-vector text features for classes
        text_features = self._get_text_features()
        
        # Calculate multi-vector similarity using MaxSim
        output, _ = self._get_class_logits(text_features, image_features)
        
        # Get frame size for output processor
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].size()[-2:]
        elif hasattr(imgs[0], 'size'):  # PIL Image
            width, height = imgs[0].size
        else:
            height, width = imgs[0].shape[:2]  # numpy array
        
        frame_size = (width, height)
        
        if self.has_logits:
            self._output_processor.store_logits = self.store_logits
        
        return self._output_processor(
            output, 
            frame_size, 
            confidence_thresh=self.config.confidence_thresh
        )