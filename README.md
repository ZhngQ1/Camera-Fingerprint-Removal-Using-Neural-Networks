# Camera Fingerprint Removal Using Neural Networks

Exploring neural network-based approaches for removing device-specific camera fingerprints (sensor pattern noise) from images, addressing privacy risks in digital forensics.

> **Lead Researcher:** Qile Zhang  
> **Advisor:** Prof. Zhongjie Ba, Zhejiang University  
> **Period:** May – June 2024

## Motivation

Every digital camera leaves a unique fingerprint — known as **sensor pattern noise (SPN)** — embedded in every image it captures. While useful for forensic source identification, this fingerprint poses a **serious privacy risk**: an adversary can link images to a specific device and, by extension, to its owner. This project explores whether neural networks can effectively suppress camera fingerprints while preserving image quality.

## Approach

We designed and evaluated multiple strategies for fingerprint removal:

| Method | Description |
|--------|-------------|
| **Autoencoder (AE)** | Unsupervised fingerprint suppression — the model learns to reconstruct images while implicitly removing device-specific noise patterns |
| **Proxy-based Training** | A training strategy to handle the absence of paired clean/fingerprinted data, using proxy fingerprints to supervise the removal process |
| **Diffusion-based Methods** | Evaluated diffusion models as an alternative denoising approach for fingerprint suppression |
| **CNN Baselines** | Compared against standard CNN-based denoising architectures |

## Key Findings

- The autoencoder-based approach effectively suppresses camera fingerprints while maintaining acceptable image quality
- Proxy-based training successfully addresses the challenge of missing paired training data
- Trade-offs exist between fingerprint removal strength and image fidelity — more aggressive removal can introduce artifacts

## Repository Structure

```
├── Code/                    # Implementation of all methods
├── Project——report.pdf      # Detailed project report
└── README.md
```

## Usage

```bash
cd Code
python main.py  # See Code/ for detailed instructions
```

## Acknowledgments

This project was conducted at Zhejiang University under the supervision of Prof. Zhongjie Ba.
