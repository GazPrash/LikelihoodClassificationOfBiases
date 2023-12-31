import re
import torch
import pandas as pd
import numpy as np
import spacy
from tqdm import trange
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


class CalculateWordBias:
    def __init__(self, target1, target2, focus_words, computing_device="cpu") -> None:
        self.target1 = target1
        self.target2 = target2
        self.focus_words = focus_words
        self.device = computing_device
        self.data = None
        self.target_col = None
        self.spacy_nlp = spacy.load("en_core_web_md")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def gpu_avail(self):
        return torch.cuda.is_available()

    def calculate_target_embeddings(self):
        c1 = self.sbert_model.encode(self.target1)
        c2 = self.sbert_model.encode(self.target2)
        
        self.c1 = torch.tensor(np.mean(c1, axis=0), device=self.device)
        self.c2 = torch.tensor(np.mean(c2, axis=0), device=self.device)

        return (self.c1, self.c2)

    def process_documents(self, dataset, text_column):
        if not isinstance(dataset, pd.DataFrame):
            raise Exception(
                "Unsupported data format, please use a pandas DataFrame instead."
            )

        regex_pattern1 = r"(@\/w+)|#|&|!"
        regex_pattern2 = r"http\S+"

        dataset[text_column] = dataset[text_column].replace(
            regex=regex_pattern1, value=""
        )
        dataset[text_column] = dataset[text_column].replace(
            regex=regex_pattern2, value=""
        )

        self.data = dataset
        self.target_col = text_column

    def calculate_biases(self):
        word_biases = []
        word_dict = {}
        word_embeddings = {}

        for i in trange(len(self.data)):
            sent = self.data[self.target_col].iloc[i]
            try:
                doc = self.spacy_nlp(str(sent))
            except Exception as e:
                print(e)
                print(i)
                continue

            bias_list = []
            for token in doc:
                if token.pos_ in self.focus_words:
                    word = token.text_with_ws.strip().lower()
                    if word in word_dict:
                        bias_list.append(word_dict[word])
                        continue

                    wv = torch.tensor(self.sbert_model.encode(word), device=self.device)
                    a = util.cos_sim(self.c1, wv)
                    b = util.cos_sim(self.c2, wv)
                    bias = (a - b).item()
                    word_embeddings[word] = wv
                    bias_list.append(bias)
                    word_dict[word] = bias
            word_biases.append(np.array(bias_list))

        biased_words = list(word_dict.items())
        biased_words.sort(key=lambda x: x[1])

        return (word_biases, word_embeddings, biased_words)
