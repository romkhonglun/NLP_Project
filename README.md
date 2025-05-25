# Eedi Reranker

This project implements a reranking model for misconceptions in educational data using a listwise approach. The model is trained and evaluated on a dataset of misconceptions, leveraging a pre-trained language model with fine-tuning.

## Features

- Sliding window approach for processing misconceptions.
- Customizable number of choices and sliding parameters.
- Integration with PEFT (Parameter-Efficient Fine-Tuning) for efficient model training.
- Token length distribution visualization.
- Cross-validation support for robust evaluation.
- WandB integration for experiment tracking.

## Requirements

- Python 3.8+
- Required Python libraries are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>python train_reranker.py --fold <fold_number> --config <path_to_config_file>