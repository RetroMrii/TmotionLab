# TmotionLab — Current Project State

## Current Phase

P11 — Multi-Label Emotion Modeling and Synthetic-Data Research

## Project Transition

TmotionLab's original DAIR.AI single-label emotion
classification laboratory was completed through P10.

The previous ML implementation has been moved to:

`DAIR-MODEL-OLD/`

Its detailed historical state is preserved in:

`DAIR-MODEL-OLD/project_state.md`

Existing DAIR backend/frontend integration and tests
are temporarily allowed to break during the transition.

## Current Objective

Develop a new multi-label emotion classification system
using GoEmotions, followed by controlled experiments
with ChatGPT-generated synthetic training data.

## Selected Dataset

Dataset: google-research-datasets/go_emotions
Configuration: simplified
Task: Multi-label emotion classification
Labels: 27 emotions + neutral (28 total)

Published splits:
- Train: 43,410
- Validation: 5,426
- Test: 5,427

Local ingestion and validation: NOT YET COMPLETED

## Planned Models

1. TF-IDF + One-vs-Rest Logistic Regression
2. RoBERTa-base multi-label classifier

All new models are independent of the previous DAIR
model architecture, vocabulary, and preprocessing.

## Planned Research

Evaluate whether quality-controlled synthetic training
examples improve fine-grained emotion classification
relative to human-only and resampling baselines.

## Current Implementation Status

P11-A: Repository restructuring in progress.

P11-B: GoEmotions ingestion and validation next.

No new model has been trained or promoted.

## Backend and Frontend

Existing FastAPI and React code remains unchanged.

The previous API and UI may be incompatible with the
new ML package until the product-integration phase.

The final website is intended to expose retained model
tracks without assuming identical dataset or label semantics.

## Engineering Policy

- Maintain experiment IDs and reproducible artifacts.
- Keep train/validation/test separation.
- Do not use held-out test examples for synthetic generation.
- Do not assume generated labels are ground truth.
- Introduce new code incrementally.
- Update the decision log when scope or architecture changes.
- Perform final dependency-aware cleanup after integration.