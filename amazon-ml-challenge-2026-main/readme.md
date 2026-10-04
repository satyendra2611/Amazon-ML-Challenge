# Amazon ML Challenge 2026 — Business Entity Resolution

A machine learning pipeline for resolving noisy business records to their
corresponding reference entities. The system combines candidate generation,
text normalization, feature engineering, and supervised classification to
handle large-scale entity matching across different countries and noisy data.

## Problem

The task is to match records from noisy source datasets (S2/S3) with the
correct entities in a reference dataset (S1).

The input records may contain:

- Spelling variations
- Abbreviations
- Formatting differences
- Address variations
- Noisy or incomplete business names
- Country-specific text variations

The solution is designed for an **open-set of countries** and does not rely
on hard-coded country-specific assumptions.

## Approach

### 1. Data Normalization

- Text normalization
- Address cleaning
- Unicode handling
- Indic text romanization
- Abbreviation and alias handling

### 2. Candidate Generation

Implemented a **country-gated IDF-based sparse blocking** strategy to reduce
the number of possible entity pairs while maintaining high candidate recall.

### 3. Feature Engineering

Generated **57 matching features** using signals from:

- Business name similarity
- Address similarity
- Token overlap
- Character-level similarity
- Normalized text
- Alias matches
- Country information

### 4. Classification

Used a **HistGradientBoostingClassifier** to classify candidate pairs as
matching or non-matching entities.

### 5. Evaluation

The pipeline was evaluated using candidate recall and Macro F0.5, with
emphasis on maintaining high precision while preserving candidate coverage.

## Results

| Metric | Result |
|---|---:|
| Validation Macro F0.5 | **0.956248** |
| Candidate Recall | **0.961853** |
| Candidate Pairs | **69,301,748** |
| Matching Features | **57** |

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP / Text Processing
- Machine Learning
- Jupyter Notebook

## Project Structure

```text
amazon-ml-challenge-2026/
│
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── utils/
│       ├── artifacts/
│       ├── models/
│       ├── README.md
│       ├── requirements.txt
│       └── run_all.sh
│
├── notebook/
│   └── entity_resolution.ipynb
│
├── Documentation_template.md
├── UPDATE_STATUS.md
├── README.md
└── .gitignore