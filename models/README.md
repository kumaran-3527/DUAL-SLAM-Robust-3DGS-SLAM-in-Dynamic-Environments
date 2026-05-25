---
language:
- en
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license
tags:
- nvidia
- fixer
- image2image
pipeline_tag: image-to-image
library_name: fixer
---
# **Fixer: Improving 3D Reconstructions with Single-Step Diffusion Models**  
[**Code**](https://github.com/nv-tlabs/Fixer) | [**Paper**](https://arxiv.org/abs/2503.01774)

## Use the Fixer Model
Please visit the [Fixer repository](https://github.com/nv-tlabs/Fixer) to access all relevant files and code needed to use Fixer

## Description: 
Fixer is a single-step image diffusion model trained to enhance and remove artifacts in rendered novel views caused by
underconstrained regions of three-dimensional (3D) representation. The technology behind Fixer is based on the concepts outlined in the paper titled
[Difix3d+: Improving 3D Reconstructions with Single-Step Diffusion Models](https://arxiv.org/abs/2503.01774).

Fixer has two operation modes: 

* Offline mode: Used during the reconstruction phase to clean up pseudo-training views that are rendered from the reconstruction
  and then distill them back into 3D. This greatly enhances underconstrained regions and improves the overall 3D representation quality. 
* Online mode: Acts as a neural enhancer during inference, effectively removing residual artifacts arising from imperfect 3D
  supervision and the limited capacity of current reconstruction models. 
  
Fixer is an all-encompassing solution, a single model compatible with both Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) representations. This model, however, was trained on 3DGUT data and is highly adaptable to GS scenes.

**Model Developer:** NVIDIA

**Model Versions:** Fixer

**Deployment Geography:** Global

**This model is ready for commercial/non-commercial use.**

### License/Terms of Use:
Your use of the model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

### Use Case:
Fixer is intended for Physical AI developers looking to enhance and improve their Neural Reconstruction pipelines. The model takes an image as an input and outputs a fixed image.

**Release Date:** 
- V1 (Stable Diffusion): June 2025, Hugging Face - https://huggingface.co/nvidia/difix  
- V2 (Cosmos): November 2025, Hugging Face - https://huggingface.co/nvidia/Fixer

## Model Architecture

**Architecture Type**: Linear Diffusion Transformer

**Network Architecture**: Linear-attention Diffusion Transformer with a Deep Compression Autoencoder (DC-AE) for efficient high-resolution image generation.

**Based on**: Cosmos-Predict-0.6B

**Number of model parameters**: 0.6B

## Input

**Input Type(s)**: Image

**Input Format(s)**: Red, Green, Blue (RGB)

**Input Parameters**: Two-Dimensional (2D)

**Other Properties Related to Input**:
* Specific Resolution: [576px x 1024px]

## Output

**Output Type(s)**: Image

**Output Format(s)**: Red, Green, Blue (RGB)

**Output Parameters**: Two-Dimensional (2D)

**Other Properties Related to Output**:
* Specific Resolution: [576px x 1024px]

## Software Integration

**Runtime Engine(s)**: PyTorch

**Supported Hardware Microarchitecture Compatibility**:
* NVIDIA Ampere

**[Preferred/Supported] Operating System(s)**: Linux

**Note**: Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Inference
**Engine**: PyTorch>=2.0.0

**Test Hardware**: 
We tested on H100, A100, H20, and L40:

| GPU Hardware | Inference Runtime |
|--------------|-------------------|
| NVIDIA H100  | 26.5ms             |
| NVIDIA A100  | 50.6ms             |
| NVIDIA H20   | 61.7ms             |
| NVIDIA L40   | 87.8ms             |

## Training, Testing, and Evaluation Datasets

Fixer was trained, tested, and evaluated using an internal dataset, where 80% of the data was used for training, 10% for evaluation, and 10% for testing.

### NVIDIA Internal AV Dataset

- **Data Modality**: Image
- **Image Training Data Size**: 1 Million to 1 Billion Images
- **Data Collection Method**: Sensors
- **Labeling Method by Dataset**: Human
- **Properties**: The dataset contains the autonomous driving image/videos captured by NVIDIA Vehicles. It's collected by autonomous driving vehicles.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)

---

## ModelCard++

### Bias

| Field                                                                                                                                                            | Response |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing: | None     |
| Measures taken to mitigate against unwanted bias:                                                                                                                | None     |

### Explainability

| Field                                                     | Response                                                                                                             |
| :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| Intended Domain:                                          | Advanced Driver Assistance Systems                                                                                   |
| Model Type:                                               | Image-to-Image                                                                                                       |
| Intended Users:                                           | Autonomous Vehicles developers enhancing and improving Neural Reconstruction pipelines.                              |
| Output:                                                   | Image                                                                                                                |
| Describe how the model works:                             | The model takes as an input an image, and outputs a fixed image                                                      |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | None                                                                                                                 |
| Technical Limitations:                                    | The reconstruction relies on the quality and consistency of input images and camera calibrations; any deficiencies in these areas can negatively impact the final output. |
| Verified to have met prescribed NVIDIA quality standards: | Yes                                                                                                                  |
| Performance Metrics:                                      | FID (Fréchet Inception Distance), PSNR (Peak Signal-to-Noise Ratio), LPIPS (Learned Perceptual Image Patch Similarity) |
| Potential Known Risks:                                    | The model is not guaranteed to fix 100% of the image artifacts. Please verify the generated scenarios are context and use appropriate. |
| Licensing:                                                | Your use of the model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). |

### Privacy

| Field                                                               | Response       |
| :------------------------------------------------------------------ | :------------- |
| Generatable or reverse engineerable personal data?                  | No             |
| Personal data used to create this model?                            | No             |
| How often is the dataset reviewed?                                  | Before release |
| Is there provenance for all datasets used in training?              | Yes            |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes            |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | Yes |
| Applicable Privacy Policy                                           | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/ |

### Safety & Security

| Field                                           | Response                                                                                                                                                                                                                                                                                                                             |
| :---------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model Application(s):                           | Image Enhancement - The model can be used to develop Autonomous Vehicles stacks that can be integrated inside vehicles. The Fixer model should not be deployed in a vehicle.                                                                                                                                                        |
| Describe the life critical impact (if present). | N/A - The model should not be deployed in a vehicle and will not perform life-critical tasks.                                                                                                                                                                                                                                       |
| Use Case Restrictions:                          | Your use of the model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).                                                                                                                                                            |
| Model and dataset restrictions:                 | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to.                                                                                                                   |