# ColModernVBert for FiftyOne

Integration of [ColModernVBert](https://huggingface.co/ModernVBERT/colmodernvbert) as a FiftyOne Zoo Model for fine-grained multimodal document retrieval and zero-shot classification.

## Overview

[ColModernVBert](https://huggingface.co/ModernVBERT/colmodernvbert) is a multi-vector vision-language model built on the ModernVBert architecture that generates ColBERT-style embeddings for both images and text. Unlike single-vector models that compress entire images into a single representation, ColModernVBert produces multiple 128-dimensional vectors per input, enabling fine-grained matching between specific image regions and text tokens.

## Key Features

- **Multi-Vector Embeddings**: Variable-length sequences of 128-dimensional vectors
  - Images: ~884 vectors per image
  - Text: ~13 vectors per query
- **MaxSim Scoring**: ColBERT-style late interaction for fine-grained matching
- **Pre-Compressed Vectors**: No token pooling required (already 128-dim per vector)
- **Dual-Mode Operation**: Pooled 128-dim for retrieval, full multi-vectors for classification
- **Zero-Shot Classification**: Use text prompts to classify images without training
- **Document Understanding**: Optimized for visual document analysis

## Architecture

### Multi-Vector Design

ColModernVBert uses a **multi-vector architecture** inspired by ColBERT, where each input (image or text) is represented by multiple vectors rather than a single embedding:

```python
# Image or Text → Processor → Model → (batch, num_vectors, 128)
```

**Benefits of Multi-Vectors:**
- ✅ **Fine-grained matching**: Match specific image regions to text tokens
- ✅ **Better accuracy**: Capture more detailed information than single vectors
- ✅ **Late interaction**: Efficient MaxSim scoring at query time

### Dual-Mode Operation

This integration supports two workflows optimized for different use cases:

#### Mode 1: Retrieval/Similarity Search
For efficient large-scale search, multi-vectors are pooled to fixed 128-dim embeddings:

```python
Multi-vectors (N, 128) → Final Pooling (mean/max) → 128-dim vector
```

**Use case**: Similarity search, embeddings visualization, clustering

#### Mode 2: Zero-Shot Classification
For accurate classification, full multi-vectors are used with MaxSim scoring:

```python
Image multi-vectors × Text multi-vectors → MaxSim → Classification scores
```

**Use case**: Zero-shot classification, fine-grained document analysis

### How MaxSim Works

MaxSim (Maximum Similarity) is a late interaction scoring mechanism:

1. For each text vector, find its maximum similarity with any image vector
2. Sum these maximum similarities across all text vectors
3. Result: A score that captures fine-grained matches between text and image

This allows the model to match specific keywords to relevant image regions, providing better accuracy than single-vector approaches.

## Installation

**Note**: This model requires the `colpali-engine` package which provides the ColModernVBert implementation.

```bash
# Install FiftyOne and BiModernVBert dependencies
pip install fiftyone torch transformers pillow
pip install git+https://github.com/illuin-tech/colpali.git@vbert#egg=colpali-engine
```

## Quick Start

### Load Dataset

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load document dataset from Hugging Face
dataset = load_from_hub(
    "Voxel51/document-haystack-10pages",
    overwrite=True,
    max_samples=250  # Optional: subset for testing
)
```

### Register the Zoo Model

```python
import fiftyone.zoo as foz

# Register this repository as a remote zoo model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/colmodernvbert",
    overwrite=True
)
```

### Basic Workflow

```python
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Load ColModernVBert model
model = foz.load_zoo_model(
    "ModernVBERT/colmodernvbert",
    pooling_strategy="mean"  # or "max"
)

# Compute embeddings for all documents
# Multi-vectors are pooled to 128-dim for storage
dataset.compute_embeddings(
    model=model,
    embeddings_field="colmodernvbert_embeddings"
)

# Check embedding dimensions
print(dataset.first()['colmodernvbert_embeddings'].shape)  # (128,)

# Build similarity index
text_img_index = fob.compute_similarity(
    dataset,
    model="ModernVBERT/colmodernvbert",
    embeddings_field="colmodernvbert_embeddings",
    brain_key="colmodernvbert_sim",
    model_kwargs={"pooling_strategy": "mean"}
)

# Query for specific content
results = text_img_index.sort_by_similarity(
    "invoice from 2024",
    k=10  # Top 10 results
)

# Launch FiftyOne App
session = fo.launch_app(results, auto=False)
```

## Pooling Strategies

The pooling strategy determines how multi-vectors are compressed to fixed-dimension embeddings for retrieval:

### Mean Pooling (Default)

Averages all vectors to create a holistic representation:

```python
model = foz.load_zoo_model(
    "ModernVBERT/colmodernvbert",
    pooling_strategy="mean"
)
```

**Best for:**
- General document retrieval
- Holistic semantic matching
- When overall content matters more than specific details

### Max Pooling

Takes the maximum value across vectors for each dimension:

```python
model = foz.load_zoo_model(
    "ModernVBERT/colmodernvbert",
    pooling_strategy="max"
)
```

**Best for:**
- Keyword-based search
- Finding specific content or phrases
- When any matching element is sufficient

## Advanced Embedding Workflows

### Embedding Visualization with UMAP

Create 2D visualizations of your document embeddings:

```python
import fiftyone.brain as fob

# First compute embeddings
dataset.compute_embeddings(
    model=model,
    embeddings_field="colmodernvbert_embeddings"
)

# Create UMAP visualization
results = fob.compute_visualization(
    dataset,
    method="umap",  # Also supports "tsne", "pca"
    brain_key="colmodernvbert_viz",
    embeddings="colmodernvbert_embeddings",
    num_dims=2
)

# Explore in the App
session = fo.launch_app(dataset)
```

### Similarity Search

Build powerful similarity search:

```python
import fiftyone.brain as fob

results = fob.compute_similarity(
    dataset,
    backend="sklearn",
    brain_key="colmodernvbert_sim",
    embeddings="colmodernvbert_embeddings"
)

# Find similar images
sample_id = dataset.first().id
similar_samples = dataset.sort_by_similarity(
    sample_id,
    brain_key="colmodernvbert_sim",
    k=10
)

# View results
session = fo.launch_app(similar_samples)
```

### Dataset Representativeness

Score how representative each sample is of your dataset:

```python
import fiftyone.brain as fob

# Compute representativeness scores
fob.compute_representativeness(
    dataset,
    representativeness_field="colmodernvbert_represent",
    method="cluster-center",
    embeddings="colmodernvbert_embeddings"
)

# Find most representative samples
representative_view = dataset.sort_by("colmodernvbert_represent", reverse=True)
```

### Duplicate Detection

Find and remove near-duplicate documents:

```python
import fiftyone.brain as fob

# Detect duplicates using embeddings
results = fob.compute_uniqueness(
    dataset,
    embeddings="colmodernvbert_embeddings"
)

# Filter to most unique samples
unique_view = dataset.sort_by("uniqueness", reverse=True)
```

## Zero-Shot Classification

ColModernVBert excels at zero-shot classification using multi-vector MaxSim scoring:

```python
import fiftyone.zoo as foz

# Load model with classes for classification
model = foz.load_zoo_model(
    "ModernVBERT/colmodernvbert",
    classes=["invoice", "receipt", "form", "contract", "other"],
    text_prompt="This document is a",
    pooling_strategy="max"  # Max pooling often works well for classification
)

# Apply model for zero-shot classification
# Uses full multi-vectors with MaxSim (not pooled embeddings)
dataset.apply_model(
    model,
    label_field="document_type_predictions"
)

# View predictions
print(dataset.first()['document_type_predictions'])
session = fo.launch_app(dataset)
```

### Dynamic Classification with Multiple Tasks

Reuse the same model for different classification tasks:

```python
import fiftyone.zoo as foz

# Load model once
model = foz.load_zoo_model(
    "ModernVBERT/colmodernvbert",
    pooling_strategy="max"
)

# Task 1: Classify document types
model.classes = ["invoice", "receipt", "form", "contract"]
model.text_prompt = "This is a " 
dataset.apply_model(model, label_field="doc_type")

# Task 2: Classify importance
model.classes = ["high_priority", "medium_priority", "low_priority"]
model.text_prompt = "The priority level is "
dataset.apply_model(model, label_field="priority")

# Task 3: Classify language
model.classes = ["english", "spanish", "french", "german", "chinese"]
model.text_prompt = "The document language is "
dataset.apply_model(model, label_field="language")

# Task 4: Classify completeness
model.classes = ["complete", "incomplete", "draft"]
model.text_prompt = "The document status is "
dataset.apply_model(model, label_field="status")
```

## Technical Details

### FiftyOne Integration Architecture

**Retrieval Pipeline** (Pooled Mode):
```python
dataset.compute_embeddings(model, embeddings_field="embeddings")
└─> embed_images()
    └─> processor.process_images(imgs)
        └─> model(**inputs)
            └─> Multi-vectors (batch, N, 128)
                └─> Final pooling (mean/max)
                    └─> Returns (batch, 128) pooled embeddings
                        └─> Stores in FiftyOne for similarity search
```

**Classification Pipeline** (Multi-Vector Mode):
```python
dataset.apply_model(model, label_field="predictions")
└─> _predict_all()
    └─> Get image multi-vectors (batch, N, 128)
    └─> Get text multi-vectors for classes (num_classes, M, 128)
    └─> processor.score() with MaxSim
        └─> Returns (batch, num_classes) logits
            └─> Output processor → Classification labels
```


### Typical Use Cases

| Use Case | Mode | Pooling Strategy | Notes |
|----------|------|------------------|-------|
| Document retrieval | Pooled | Mean | Efficient for large-scale search |
| Keyword search | Pooled | Max | Finds specific content matches |
| Zero-shot classification | Multi-vector | N/A | Highest accuracy with MaxSim |
| Fine-grained matching | Multi-vector | N/A | Match specific regions |
| Embeddings visualization | Pooled | Mean | Holistic semantic space |
| Duplicate detection | Pooled | Mean | Fast similarity computation |

## Combining Embeddings and Classification

Use the same model for both workflows:

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# Load model once
model = foz.load_zoo_model(
    "ModernVBERT/colmodernvbert",
    pooling_strategy="mean"
)

# Step 1: Compute pooled embeddings for similarity search
dataset.compute_embeddings(
    model=model,
    embeddings_field="colmodernvbert_embeddings"
)

# Step 2: Build similarity index
index = fob.compute_similarity(
    dataset,
    model="ModernVBERT/colmodernvbert",
    embeddings_field="colmodernvbert_embeddings",
    brain_key="colmodernvbert_sim"
)

# Step 3: Add zero-shot classification (uses full multi-vectors)
model.classes = ["technical", "financial", "legal", "personal"]
model.text_prompt = "This document category is"
dataset.apply_model(model, label_field="category")

# Step 4: Add more classifications
model.classes = ["urgent", "normal", "low_priority"]
model.text_prompt = "The urgency level is"
dataset.apply_model(model, label_field="urgency")

# Explore combined results
session = fo.launch_app(dataset)
```

## Resources

- **Model Hub**: [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert)
- **ColPali Engine**: [colpali-engine](https://github.com/illuin-tech/colpali)
- **FiftyOne Docs**: [docs.voxel51.com](https://docs.voxel51.com)
- **Base Architecture**: ModernVBert
- **Inspiration**: ColBERT late interaction

## Citation

If you use ColModernVBert in your research, please cite:

```bibtex
@misc{teiletche2025modernvbertsmallervisualdocument,
      title={ModernVBERT: Towards Smaller Visual Document Retrievers}, 
      author={Paul Teiletche and Quentin Macé and Max Conti and Antonio Loison and Gautier Viaud and Pierre Colombo and Manuel Faysse},
      year={2025},
      eprint={2510.01149},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2510.01149}, 
}

```

## License

- **Model**: MIT
- **Integration Code**: Apache 2.0 (see [LICENSE](LICENSE))

## Contributing

Found a bug or have a feature request? Please open an issue on GitHub!

## Acknowledgments

- **ModernVBERT Team** for the excellent ColModernVBert model
- **ColPali Engine** for the model implementation and processor
- **ColBERT** for pioneering multi-vector late interaction
- **Voxel51** for the FiftyOne framework and brain module architecture
- **HuggingFace** for model hosting