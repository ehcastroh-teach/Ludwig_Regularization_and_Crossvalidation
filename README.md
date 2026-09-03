# Regularization and Cross-Validation Using Ludwig

This repository teaches how to identify and address common machine learning problems - overfitting, underfitting, and poor generalization - using Ludwig, an open source declarative deep learning framework. You will build and compare multiple NLP model architectures on the Stanford Sentiment Treebank dataset, learning how encoder choice and training hyperparameters affect model performance. No custom training loops or low-level framework code required.

---

## Learning Objectives

- Understand what regularization means in the context of deep learning (early stopping, learning rate tuning)
- Perform model cross-validation by comparing multiple encoder architectures (RNN, CNN, CNNRNN, BERT) on the same dataset
- Load and preprocess text data using Torchtext and convert it into a format Ludwig can consume
- Define Ludwig model configurations as Python dictionaries and train models with a single API call
- Visualize training and validation loss curves to diagnose overfitting vs. underfitting
- Evaluate trained models on held-out test data and compare accuracy across configurations

---

## File Dictionary

| File | Description |
|------|-------------|
| `ludwig_regularization_and_crossvalidation.ipynb` | Main tutorial - builds and compares three Ludwig NLP encoders on the SST dataset using early stopping and learning rate regularization |
| `assets/homeworks/tensorflow_tensors_gradients_companion.ipynb` | Companion exercise covering TensorFlow tensor creation, type casting, graph tracing, and automatic differentiation via `tf.GradientTape` |
| `requirements.txt` | Python package dependencies |
| `assets/content/images/` | Reference images used in the notebook |

---

## Workflow

```
Install Ludwig + Torchtext
        |
        v
Load Stanford Sentiment Treebank (SST)
  - 5 sentiment classes: positive, negative, very positive, very negative, neutral
        |
        v
Define Baseline Model (Vanilla RNN encoder)
  - early_stop: 100
        |
        v
Compare Encoder Architectures
  - Parallel CNN    (learning_rate: 1e-5, early_stop: 10)
  - CNNRNN          (learning_rate: 1e-5, early_stop: 10)
  - Stacked Par CNN (learning_rate: 1e-3, early_stop: 10)
        |
        v
Visualize Learning Curves
  - Compare training vs. validation loss across all runs
        |
        v
Evaluate on Test Set
  - Inspect accuracy, per-class stats
        |
        v
(Optional) Fine-tune BERT encoder
```

---

## Step-by-Step Walkthrough

### 1. Install Ludwig and Torchtext

Ludwig abstracts model architecture details into a configuration dictionary, letting you focus on what you are trying to learn (encoder choice, regularization strategy) rather than implementation boilerplate. Torchtext provides a convenient loader for standard NLP benchmarks.

```bash
pip install -r requirements.txt
```

### 2. Load the Stanford Sentiment Treebank

SST is a sentence-level corpus of 10,662 labeled movie review sentences. Each sentence is assigned one of five fine-grained sentiment labels. Because SST includes subtree-level annotations, `train_subtrees=True` gives significantly more training examples - which is important when comparing architectures that have different sample efficiency.

### 3. Define and Train the Baseline (Vanilla RNN)

The RNN encoder processes text sequentially, making it a natural baseline for sequence classification. The `early_stop: 100` setting means training stops after 100 epochs with no improvement - a form of regularization that prevents the model from memorizing noise in the training set.

### 4. Compare Encoder Architectures

Each subsequent model uses a different encoder:

- **Parallel CNN** - applies multiple convolutional filters in parallel, capturing n-gram patterns at different scales. Paired with a lower learning rate (`1e-5`) to prevent early divergence.
- **CNNRNN** - combines convolutional feature extraction with recurrent processing, attempting to capture both local patterns and global sequence dependencies.
- **Stacked Parallel CNN** - stacks multiple parallel CNN layers; uses a higher learning rate (`1e-3`) since the deeper architecture benefits from faster initial convergence.

Comparing these models on the same dataset is a form of cross-validation - you are evaluating which inductive bias best fits the structure of this problem.

### 5. Visualize Learning Curves

After training, Ludwig's visualization tool plots training and validation loss across epochs. Watch for:
- **Overfitting**: training loss keeps falling while validation loss plateaus or rises
- **Underfitting**: both losses remain high and flat

### 6. Evaluate and Compare

Run each trained model on the held-out test set. Compare accuracy scores to determine which architecture generalizes best.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/ehcastroh-teach/Ludwig_Regularization_and_Crossvalidation.git
cd Ludwig_Regularization_and_Crossvalidation


# 2. Create and activate a virtual environment (Python 3.7+ recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook ludwig_regularization_and_crossvalidation.ipynb

# 5. Run all cells top-to-bottom (Kernel > Restart and Run All)
```

Note: A GPU is strongly recommended. The notebook includes a GPU availability check in the setup section. Without a GPU, training the BERT encoder cell (optional, at the end) will be very slow.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Regularization** | Any technique that reduces overfitting - examples include early stopping, dropout, and weight decay |
| **Early stopping** | Halt training when validation loss stops improving, preventing the model from memorizing the training set |
| **Cross-validation** | Evaluating a model's generalization ability - here used informally to mean comparing multiple model configurations on the same train/val/test split |
| **Encoder** | The component that converts raw input (text tokens) into a fixed-size representation; different encoders (RNN, CNN, BERT) impose different inductive biases |
| **Learning rate** | Controls how large each gradient-descent update step is; too high causes divergence, too low causes slow or incomplete training |
| **Overfitting** | When a model performs well on training data but poorly on unseen data, usually from memorizing noise |
| **Underfitting** | When a model is too simple to capture the underlying structure of the data |
| **Ludwig** | An open source declarative ML framework that lets you train models by specifying a configuration dictionary rather than writing training code |
| **Stanford Sentiment Treebank (SST)** | A benchmark NLP dataset of 10,662 movie review sentences with fine-grained sentiment labels |
| **Inductive bias** | The assumptions built into a model architecture that constrain what it can learn; RNNs assume order matters, CNNs assume local patterns are informative |

---

## Further Reading

- *Deep Learning* - MIT Press
- *Natural Language Processing with PyTorch* - O'Reilly
- Ludwig documentation: [ludwig-ai.github.io/ludwig-docs](https://ludwig-ai.github.io/ludwig-docs/)
- Stanford Sentiment Treebank: [nlp.stanford.edu/sentiment](https://nlp.stanford.edu/sentiment/)
- *Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank* - EMNLP 2013

---

## Credits and Acknowledgements

- Ludwig framework: Piero Molino and the Ludwig team
- *Deep Learning*: Goodfellow, Bengio, Courville (MIT Press, 2016)
- *Natural Language Processing with PyTorch*: Rao, McMahan (O'Reilly, 2019)
- Stanford Sentiment Treebank: Socher et al. (2013), built on the Rotten Tomatoes dataset by Pang and Lee (2005)
- Torchtext: PyTorch team

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
