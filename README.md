# Context Diffusion: In-Context Aware Image Generation

## Overview

This project implements a **context-aware image generation** framework using a **Latent Diffusion Model (LDM)**. The model takes three types of inputs:
1. A **query image** that provides the desired structure/layout.
2. One or more **context images** that define the style, texture, and color.
3. An optional **text prompt** for additional semantic guidance.

By combining these inputs, the model generates a new image that retains the layout from the query image, the style from context images, and any semantic attributes specified by the prompt. This approach is useful for tasks where images need to adhere to specific structural constraints while maintaining a desired visual style.

## Features

- **CLIP-Based Encoding**: Extracts embeddings from both text prompts and images using CLIP.
- **Latent Diffusion Model**: Uses a Stable Diffusion model from the `diffusers` library to perform the image generation.
- **Dynamic Image Normalization**: Calculates mean and standard deviation per image to adaptively normalize inputs, improving flexibility.

## Installation

To run this code, you’ll need Python 3.7 or above. Install the required dependencies using:

```bash
pip install torch torchvision diffusers transformers
```

## Usage

1. **Prepare Input Images**: Provide paths to the query image and one or more context images.
2. **Set an Optional Text Prompt**: Use a prompt to add specific styles or elements (e.g., "A stylized forest with misty mountains").
3. **Run the Model**: Generate an output image that combines the layout, style, and semantic guidance.

### Example Code

The following code demonstrates how to use the model:

```python
# Load query and context images
query_image = load_image_as_tensor("path/to/query_image.jpg")
context_images = [
    load_image_as_tensor("path/to/context_image1.jpg"),
    load_image_as_tensor("path/to/context_image2.jpg"),
    load_image_as_tensor("path/to/context_image3.jpg")
]

# Set an optional text prompt
prompt = "A stylized forest with misty mountains."

# Generate and display the image
generated_image = model(query_image, context_images, prompt)

# Display the generated image
import matplotlib.pyplot as plt
plt.imshow(generated_image)
plt.axis('off')
plt.show()
```

### Code Explanation

- **Image Loading and Encoding**: The `load_image_as_tensor` function resizes and normalizes each input image dynamically, based on its calculated mean and standard deviation.
- **Embedding Calculation**: Text and image embeddings are generated using CLIP’s pre-trained models, allowing the model to understand both the structure and style of the inputs.
- **Latent Diffusion Model**: The Stable Diffusion model combines these embeddings to generate a final output image.

## Future Improvements

1. **Dataset Fine-Tuning**: Additional training on specific datasets could improve style adaptation.
2. **Experimentation with Alternative Embedding Models**: Other embeddings may better capture unique styles and complex layouts.

## Acknowledgments

This project uses [Hugging Face's diffusers library](https://github.com/huggingface/diffusers) and [OpenAI's CLIP model](https://github.com/openai/CLIP). Special thanks to these libraries for making pre-trained models accessible.
