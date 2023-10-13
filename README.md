# LikelihoodClassificationOfBiases

A likelihood-based statistical tool specifically designed to identify and classify various types of biased statements or comments related to different categories such as gender or religion within a given dataset.

## Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Contributing](#contributing)

## Abstract

Detecting biased or discriminatory language in so-
cial media spaces towards protected attributes (such as gen-
der, ethnicity or religion) is a laborious and subjective task,
often necessitating the involvement of human moderators or
computationally intensive neural networks. In this paper, we
provide a likelihood-based statistical approach to identify and
classify biased statements or comments in online communication
platforms. Our model offers a fast solution that can aid in efficient
content moderation, text analysis and annotation. Using word
embeddings and statistical analysis, we can figure out comments
or sentences that may contain certain biases that we may wish to
set apart. For instance, identifying whether a sentence revolves
around male, female, or perhaps none of them. To test the
performance of our model and validate the classification pro-
cess, experiments were performed on various publicly available
datasets, including the Multi-Dimensional Gender Bias dataset

## Features

- **Identify Biases:** Likelihood-based statistical approach specifically designed to identify and classify various
  types of biased statements or comments related to different
  categories such as gender or religion in language.
- **Annotation:** Can be used for content moderation, text analysis and annotation.
- **Controversial Biases:** Identify biased or discriminatory language in so- cial media spaces towards protected attributes (such as gender, ethnicity or religion.

## Installation

Follow these steps to install and set up and use our model to anaylze text on your local machine:

1. Step 1: Install all the prerequisites using the following command

```bash
$ pip install requirements.txt
```

2. Step 2: Choose and install a pre-trained word embedding model from popular options like Bert, Sbert, spaCy, or gensim. (You can also train your own word-embeddings)
3. You can use the example notebooks, such as `example1.ipynb` to analyze your own data with the **sbert_model**, Please feel free to interact with the two experimentation notebooks located in the main directory. You have the flexibility to choose any word embedding model that you prefer for your experiments. By exploring and comparing the results obtained from different word embeddings.

## Usage

By utilizing the model accessible through the `src` folder, you have the capability to detect textual bias within any dataset, while also analyzing the disparities between any two input target parameters. This functionality empowers you to identify and assess potential biases in the textual data, leading to a deeper understanding of how certain factors can influence the information present in the dataset.

```python
# Load the dataset
import pandas as pd
data = pd.read_csv("path/to/data")
```

You can use the following piece of code to annotate your initial dataset to classify comments in thre different categories, namely $c_1$ biased, $c_2$ biased or neutral.

```python
from src.wordbiases import CalculateWordBias
from src.model import LikelihoodModelForNormalDist

# Set two different target sets consitsting of words you wish to target
target_set_1 = ["abuse", "bad", "hate", "worst"]
target_set_2   = ["love", "good", "peace", "calm"]

# Select a list of Focus Words (i.e words that may contribute to the biases in any given comment)
F = ["ADJ", "NOUN", "PRON"]

wbcalc = CalculateWordBias(target_set_1, target_set_2, F, computing_device="cuda")
wbcalc.process_documents(data, "text")
c1, c2 = wbcalc.calculate_target_embeddings()

wbiases, _ , biased_words = wbcalc.calculate_biases()
total_pop = [b for _, b in biased_words]

mu = np.mean(total_pop)
sigma = np.std(total_pop)

likelihood_clf = LikelihoodModelForNormalDist(0.05, 0.95, threshold_limit=0)
likelihood_clf.fit_total_pop(total_pop)

# predictions
preds = likelihood_clf.predict(wbiases)[0]
data["prclass"] = preds
```

## Datasets

Information regarding the datasets used in the experiments

| Dataset                           | Targeted Bias       | Annotated | Total comments | $c_1$-related | $c_2$-related | Neutral comments |
| --------------------------------- | ------------------- | --------- | -------------- | ------------- | ------------- | ---------------- |
| Twitter (social_bias_frames)      | Racism/Antisemitism | Yes       | 30,430         | 3124          | 1143          | 26,163           |
| Reddit (one-million-reddit-jokes) | Racism              | No        | 400,000        | 16,753        | 19,119        | 364,128          |
| McDonald's Store Reviews (US)     | Sentiment           | No        | 33,396         | 6,705         | 2263          | 24428            |
| ImageChat (md_gender_bias)        | Gender-Identity     | Yes       | 329,908        | 21,251        | 62,383        | 225,168          |
| News Classification               | News Categories     | No        | 100,000        | 12,871        | 12,871        | 74,270           |

## Contributing

We welcome contributions from the community! If you'd like to contribute to the project, please follow these guidelines:

1. **Fork the repository.**
   Click on the "Fork" button on the top right corner of this repository to create your own copy.

2. **Create a new branch.**
   In your forked repository, create a new branch to work on your changes. You can name the branch according to the feature or bug fix you are implementing.

3. **Make your changes and commit them.**
   Make the necessary changes and improvements to the code in your branch. Commit your changes with clear and descriptive commit messages.

4. **Push the changes to your fork.**
   Once you are satisfied with your changes, push the changes to the branch in your forked repository.

5. **Submit a pull request to the original repository.**
   Finally, navigate to the original repository and click on the "New Pull Request" button. Select your branch and describe the changes you have made. Submit the pull request, and we will review your contribution.

Thank you for your interest in contributing to our project! We appreciate your support in making this project better. If you have any questions or need assistance, don't hesitate to reach out to us through the Issues section or the pull request itself.
