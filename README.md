# ⚡ Document Classification with PyTorch & TorchText — NLP Pipeline

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/zadkiel-dev/document-classification-nlp/releases)

![news-ml](https://images.unsplash.com/photo-1508385082359-f7f2a3f36b98?auto=format&fit=crop&w=1400&q=80)

Automated document classifier for news articles. It uses PyTorch and TorchText. The project loads and preprocesses data, trains a text classifier, visualizes embeddings, and predicts topics like World, Sports, Business, and Sci/Tech.

Badges
- [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
- [![PyTorch](https://img.shields.io/badge/pytorch-1.10-red?logo=pytorch)](https://pytorch.org/)
- [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
- [![Releases](https://img.shields.io/badge/get%20model-download-orange?logo=github)](https://github.com/zadkiel-dev/document-classification-nlp/releases)

Topics
- ag-news, deep-learning, document-classification, embeddingbag, machine-learning, natural-language-processing, nlp, orchtext, pytorch, text-classification

Why this repo
- Train an end-to-end text classifier for news.
- Use EmbeddingBag for efficient embedding lookup.
- Visualize embeddings with t-SNE or UMAP.
- Run inference on single articles or on a corpus.

Demo GIF
![demo](https://media.giphy.com/media/3o7aCTfyhYawdOXcFW/giphy.gif)

Table of contents
- Features
- Quickstart
- Data and preprocessing
- Training
- Evaluation
- Embedding visualization
- Inference
- Releases and model download
- Project structure
- How it works
- Tips and best practices
- Contributing
- License
- References

Features
- PyTorch model built for speed and clarity.
- TorchText data pipeline and tokenization.
- EmbeddingBag-based architecture for variable-length text.
- Checkpoints and export to TorchScript.
- Scripts for training, evaluation, and inference.
- Visual tools for embedding inspection.

Quickstart (local)
1. Clone the repo.
2. Create a virtual environment and install dependencies.
3. Run training or load a prebuilt release.

Example commands
```bash
git clone https://github.com/zadkiel-dev/document-classification-nlp.git
cd document-classification-nlp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start training (sample)
```bash
python train.py --data ag_news --epochs 5 --batch-size 64 --lr 1e-3
```

Start inference on a single article
```bash
python predict.py --model checkpoints/best.pt --text "Stocks rose after the earnings report" 
```

Data and preprocessing
- Default dataset: AG_NEWS (four classes: World, Sports, Business, Sci/Tech).
- Tokenization: simple token split with optional truecasing.
- Vocabulary: built from training split with cutoff and max size.
- Embeddings: random init or pre-trained GloVe vectors.
- Batching: TorchText iterators and EmbeddingBag collate for fast training.

Preprocessing steps
1. Load raw articles and labels.
2. Tokenize and map tokens to ids.
3. Pad or use EmbeddingBag with offsets.
4. Build DataLoader with shuffle and batch size.

Model (high level)
- EmbeddingBag -> mean pooling for bag-of-words style embedding.
- Feedforward layers with dropout and ReLU.
- Final linear layer with softmax for class scores.

Training
- Use Adam or SGD with linear LR scheduler.
- Save the best checkpoint on validation accuracy.
- Use mixed precision if available.
- Use gradient clipping for stability.

Training tips
- Start with lr=1e-3 and batch 32-128.
- Use dropout 0.2–0.5 to avoid overfit.
- Monitor validation accuracy and loss.
- Save checkpoints every epoch.

Evaluation
- Compute accuracy, precision, recall, F1 per class.
- Confusion matrix plotted with Matplotlib or Seaborn.
- Use a test split for final metrics.
- Export classification report to CSV for analysis.

Embedding visualization
- Export the embedding layer weights.
- Use t-SNE or UMAP for projection to 2D.
- Color points by predicted or true class.
- Tools included:
  - visualize_embeddings.py
  - tensorboard embedding projector export

Example visualization command
```bash
python visualize_embeddings.py --model checkpoints/best.pt --method tsne --output emb_tsne.png
```

Inference examples
- Single article:
```bash
python predict.py --model checkpoints/best.pt --text "NASA launches a new satellite to study climate."
```
- Batch inference:
```bash
python batch_predict.py --model checkpoints/best.pt --input data/new_articles.csv --output preds.csv
```
- API mode:
  - Use the provided Flask app to serve model predictions.
  - Run app: python serve.py --model checkpoints/best.pt --port 8080

Releases and model download
You can download ready-to-run artifacts from the Releases page. Download the release file and execute it as needed. Example:
- Visit the releases page: https://github.com/zadkiel-dev/document-classification-nlp/releases
- Download the model archive or runnable script (for example model.tar.gz or run_model.sh).
- If the release contains a shell script, make it executable and run:
```bash
tar -xzf model.tar.gz
chmod +x run_model.sh
./run_model.sh
```
If the release contains a TorchScript file, load it in Python:
```python
import torch
ts = torch.jit.load("model.pt")
pred = ts(["This is a sample news article"])
```

Project structure
- data/                -> dataset helpers and download scripts
- src/                 -> model, training, utils
- notebooks/           -> EDA and visualization notebooks
- checkpoints/         -> saved checkpoints
- scripts/             -> training and inference scripts
- requirements.txt
- train.py
- predict.py
- visualize_embeddings.py

How it works
- The pipeline moves from raw text to class prediction.
- TorchText builds the token->id mapping and DataLoader.
- EmbeddingBag provides fast bag-of-words embedding with offsets.
- The model learns class weights with cross-entropy loss.
- Validation picks the best checkpoint for inference.

Common commands
- Full train with wandb disabled:
```bash
python train.py --data ag_news --epochs 10 --batch-size 64 --no-wandb
```
- Evaluate a checkpoint:
```bash
python eval.py --model checkpoints/best.pt --data ag_news
```
- Export to TorchScript:
```bash
python export_torchscript.py --model checkpoints/best.pt --out model.pt
```

Performance targets (example)
- Baseline: ~88% accuracy on AG_NEWS with EmbeddingBag + 2-layer MLP.
- With larger vocab and pretrained embeddings: ~91% accuracy.
- Use model ensembling for extra gains.

Tips and best practices
- Use a fixed random seed for reproducibility.
- Log metrics per epoch and keep history files.
- Use small experiments to tune learning rate first.
- Profile data pipeline if CPU becomes a bottleneck.

Contributing
- Open an issue for bug reports or feature requests.
- Fork the repo and submit a PR for code changes.
- Follow the coding style in src/ and add tests for new utilities.
- Add a short description when you open a PR.

License
- MIT License. See LICENSE file.

References and Resources
- AG_NEWS dataset
- TorchText docs: https://pytorch.org/text/stable/
- PyTorch docs: https://pytorch.org/
- EmbeddingBag patterns and tutorials

If you need a ready-made binary or model, download the release asset and run it as shown above. Check the Releases page for the latest artifacts: https://github.com/zadkiel-dev/document-classification-nlp/releases

Support
- Use Issues for bugs and feature requests.
- Use Discussions for questions and design talks.