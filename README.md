# Regularization and Cross-Validation Using Ludwig

This repository teaches how to identify and address common machine learning problems - overfitting, underfitting, and poor generalization - using Ludwig, an open source declarative deep learning framework. Starting from a baseline sentiment classifier built on the Stanford Sentiment Treebank, you will apply early stopping and learning rate control as regularization techniques, then compare three encoder architectures (vanilla RNN, Parallel CNN, CNNRNN) as a practical form of model cross-validation - all without writing a custom training loop or low-level framework code.

---

## Learning Objectives

- Understand what regularization means in deep learning and how early stopping prevents overfitting by capping effective training steps
- Apply learning rate tuning as a complementary regularization lever and recognize the symptoms of rates that are too high or too low
- Compare encoder architectures on the same dataset as a model selection cross-validation strategy
- Load and preprocess text data from the Stanford Sentiment Treebank using Torchtext, including subtree-level training expansion
- Define Ludwig model configurations as Python dictionaries and train models with a single API call
- Interpret training and validation loss curves to diagnose overfitting versus underfitting
- Evaluate and compare trained models on a held-out test set

---

## Data / File Dictionary

| File / Directory | Description |
|---|---|
| `ludwig_regularization_and_crossvalidation.ipynb` | Main tutorial - builds and compares three Ludwig NLP encoders (RNN, Parallel CNN, CNNRNN) on the Stanford Sentiment Treebank using early stopping and learning rate regularization |
| `assets/homeworks/tensorflow_tensors_gradients_companion.ipynb` | Companion exercise covering TensorFlow tensor creation, dtype compatibility, computation graph tracing with `@tf.function`, and automatic differentiation via `tf.GradientTape` |
| `requirements.txt` | Python package dependencies pinned to Ludwig 0.2.1 and TensorFlow 1.15 |
| `assets/content/images/` | Reference images used in notebook cells |

---

## Workflow Diagram

```
Install Ludwig + Torchtext + TensorFlow 1.15
        |
        v
Load Stanford Sentiment Treebank (SST)
  - 5 sentiment classes (very positive, positive, neutral,
    negative, very negative)
  - train_subtrees=True expands training set with phrase-level labels
        |
        v
Define and Train Baseline Model (Vanilla RNN encoder)
  - early_stop: 100 (permissive - establishes regularization concept)
        |
        v
Compare Encoder Architectures
  - Parallel CNN    (learning_rate: 1e-5, early_stop: 10)
  - CNNRNN          (learning_rate: 1e-5, early_stop: 10)
        |
        v
Visualize Learning Curves
  - Training vs. validation loss per epoch for each run
  - Identify overfitting, underfitting, healthy convergence
        |
        v
Evaluate on Held-Out Test Set
  - Compare accuracy across all encoder architectures
```

---

## Step-by-Step Walkthrough

### 1. Environment Setup

Ludwig 0.2.1 depends on TensorFlow 1.15, which is not the version installed by a plain `pip install tensorflow` today. The Appendix in the notebook walks through uninstalling TF 2.x and installing the TF 1.15 build. A GPU is strongly recommended - the CNNRNN encoder is slow on CPU.

We install Ludwig and Torchtext at pinned versions because the Ludwig model configuration API and the Torchtext `Field`/`splits` interface both changed substantially in later releases. Pinning ensures the notebook runs without modification.

### 2. Load the Stanford Sentiment Treebank

The Stanford Sentiment Treebank (SST) is a sentence-level corpus of 10,662 labeled movie review sentences across five fine-grained sentiment classes. What makes it unusual is that it provides labels not just at the sentence level but for every phrase in each sentence's parse tree.

Setting `train_subtrees=True` includes all phrase-level examples in training, expanding the effective training set well beyond 10,662 examples. This matters when comparing architectures: more training data reduces the variance in each model's estimate of generalization performance, making the architecture comparison more reliable. Validation and test sets use sentence-level labels only.

Torchtext returns tokenized sequences; we use `TreebankWordDetokenizer` to join tokens back into strings because Ludwig expects raw text, not lists of tokens.

### 3. Define and Train the Baseline (Vanilla RNN)

The RNN encoder processes text sequentially, token by token. The hidden state at each step carries a summary of all tokens seen so far. This is the natural baseline for sequence classification because it imposes the weakest assumptions - it lets the model learn any sequential pattern, including long-range dependencies like negation.

The `early_stop: 100` setting is permissive: training halts only after 100 consecutive epochs without validation improvement. This gives the RNN ample time to converge while still preventing indefinite training. Early stopping is a genuine regularization technique - it constrains the effective capacity of the model by capping the number of gradient steps, the same way weight decay constrains parameter magnitude.

### 4. Compare Encoder Architectures

Each subsequent model uses a different encoder, each embodying a different inductive bias about what structure in text is most informative for sentiment:

- Parallel CNN (`encoder: parallel_cnn`) - applies convolutional filters of multiple widths across sliding windows of tokens, extracting local n-gram features in parallel. Inductive bias: short, local phrase patterns are the most discriminative features. Applied with `learning_rate: 1e-5` because convolutional weights trained on text embeddings are sensitive to large gradient updates early in training - a lower rate gives filters time to specialize before the output layer's loss gradient dominates.

- CNNRNN (`encoder: cnnrnn`) - stacks CNN layers for local feature extraction, followed by an RNN layer that aggregates those features sequentially across the full sentence. Inductive bias: local n-gram patterns exist and their sequential arrangement across the sentence carries additional information. Combines both prior encoders' strengths at the cost of more computation. Same `learning_rate: 1e-5` rationale as Parallel CNN.

Comparing these models on the same fixed train/validation/test split is the form of cross-validation used in this tutorial: you are asking which architectural assumption about text and sentiment generalizes best to unseen sentences. This is not formal k-fold cross-validation (which would rotate the held-out fold and average results), but it answers the same practical question with far less compute.

### 5. Visualize Learning Curves

After training, Ludwig saves per-epoch training statistics to a `results/` directory. The `ludwig visualize --visualization learning_curves` command reads those statistics and generates accuracy and loss plots.

Reading learning curves is the primary diagnostic tool for understanding what happened during training:

- Healthy training: both losses decrease together early in training, then plateau at similar values. The gap between them stays small.
- Overfitting: training loss keeps falling while validation loss plateaus or rises. The widening gap is the signature. Remedies include reducing `early_stop` patience, adding dropout to the encoder, or choosing a simpler encoder.
- Underfitting: both losses remain high and plateau early. The model lacks capacity, or the learning rate is too low. Remedies include a more expressive encoder or a higher learning rate.

### 6. Evaluate on the Test Set

After reading learning curves, call `model.test(data_df=test_df)` on each trained model to get test accuracy. The test set is held out through the entire training and architecture-selection process - it is used only once, at the end, to get an unbiased estimate of each model's generalization performance. Comparing test accuracy across all three encoders answers the cross-validation question: which inductive bias fits this problem best.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/ehcastroh-teach/Ludwig_Regularization_and_Crossvalidation.git
cd Ludwig_Regularization_and_Crossvalidation

# 2. Create and activate a virtual environment (Python 3.6 or 3.7 required)
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook ludwig_regularization_and_crossvalidation.ipynb

# 5. Run all cells top-to-bottom (Kernel > Restart and Run All)
```

Note: A GPU is strongly recommended. The notebook includes a GPU availability check near the top. Without a GPU, CNNRNN training will be very slow. The requirements file installs `tensorflow-gpu==1.15.2`; replace it with `tensorflow==1.15.2` if no GPU is available.

---

## Key Concepts Glossary

| Term | Definition |
|---|---|
| Regularization | Any technique that reduces a model's tendency to overfit - examples include early stopping, dropout, and weight decay |
| Early stopping | Halt training when validation loss stops improving for N consecutive epochs; constrains effective model capacity by capping gradient steps |
| Overfitting | A model performs well on training data but poorly on unseen data, usually by memorizing noise instead of learning generalizable patterns |
| Underfitting | A model is too simple (or trained too briefly) to capture the underlying structure of the data; both training and validation loss remain high |
| Cross-validation | A strategy for estimating generalization performance; in this tutorial it means comparing multiple encoder architectures on the same fixed train/val/test split |
| Encoder | The component that converts raw input (text tokens) into a fixed-size representation; different encoders (RNN, CNN, CNNRNN) impose different inductive biases about which patterns in the input matter |
| Inductive bias | The assumptions built into a model architecture that constrain what it can learn; RNNs assume token order and long-range dependencies matter, CNNs assume local n-gram patterns are most informative |
| Learning rate | Controls the magnitude of each gradient-descent update; too high causes oscillation or divergence, too low causes slow convergence or early plateauing |
| Ludwig | An open source declarative ML framework that lets you configure and train models by specifying a Python dictionary rather than writing training code |
| Stanford Sentiment Treebank (SST) | A benchmark NLP dataset of 10,662 movie review sentences with five-class fine-grained sentiment labels, notable for providing phrase-level annotations via parse trees |
| Generalization gap | The difference between training loss and validation loss; a small gap indicates the model is learning transferable patterns, a large gap indicates overfitting |
| Learning curve | A plot of training and validation loss across training epochs, used to diagnose overfitting, underfitting, and convergence behavior |
| Torchtext | A PyTorch library that provides loaders for standard NLP benchmark datasets including SST |

---

## Further Reading

- "Deep Learning" - MIT Press
- "Natural Language Processing with PyTorch" - O'Reilly
- "Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank" - EMNLP 2013
- "Convolutional Neural Networks for Sentence Classification" - EMNLP 2014
- "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection" - IJCAI 1995
- "Early Stopping - But When?" - Neural Networks: Tricks of the Trade

---

## Credits and Acknowledgements

- Ludwig framework: Piero Molino and the Ludwig team (Uber AI / Ludwig AI)
- "Deep Learning": Goodfellow, Bengio, Courville (MIT Press, 2016) - source for regularization and practical methodology chapters
- "Natural Language Processing with PyTorch": Rao, McMahan (O'Reilly, 2019)
- Stanford Sentiment Treebank: Socher et al. (2013), built on the Rotten Tomatoes dataset by Pang and Lee (2005)
- Torchtext: PyTorch team
- "Convolutional Neural Networks for Sentence Classification": Kim (2014) - theoretical basis for the CNN encoder comparison

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
