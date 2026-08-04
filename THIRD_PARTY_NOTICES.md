# Third-party and organizer-provided materials

This repository combines original project code with third-party dependencies
and competition materials. Their rights are not transferred by the repository
license.

## Base model

Runtime image generation uses Stable Diffusion 1.5:

- Model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Pinned revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`

Users must review and comply with the model card and license distributed with
that model.

Fast concept-image generation can additionally load:

- Adapter: `latent-consistency/lcm-lora-sdv1-5`

It is downloaded during model preparation and remains subject to its own model
card and license.

## Organizer-provided materials

Organizer-provided datasets, templates, color tables, training images,
workbooks, and competition documents are intentionally excluded from Git.
Users must supply authorized copies locally under `reference_data/` when
running workflows that depend on those materials. All rights remain with their
respective rights holders.

## Python dependencies

Python packages are installed from `pyproject.toml` or
`requirements-lock.txt`. Each dependency remains subject to its own license.
