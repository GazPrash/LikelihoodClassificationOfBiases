import pandas as pd
import numpy as np
from wordbiases import CalculateWordBias
from model import LikelihoodModelForNormalDist


def preprocess_data(data):
    ...
    return data


def eval_T(k=2):
    data = pd.read_csv("data/McReview.csv")
    data = preprocess_data(data)

    target_set_1 = ["sports", "football", "athletics", "game"]
    target_set_2 = ["crime", "murder", "theft", "violence"]
    F = ["ADJ", "NOUN", "PRON"]
    wbcalc = CalculateWordBias(target_set_1, target_set_2, F, computing_device="cuda")
    wbcalc.process_documents(data, "review")
    c1, c2 = wbcalc.calculate_target_embeddings()

    wbiases, _, biased_words = wbcalc.calculate_biases()
    total_pop = [b for _, b in biased_words]

    mu = np.mean(total_pop)
    sigma = np.std(total_pop)

    # TODO : Find a way to compute or estimate t1 or t2
    likelihood_clf = LikelihoodModelForNormalDist(
        mu - k * sigma, mu + k * sigma, threshold_limit=0
    )
    likelihood_clf.fit_total_pop()

    preds = likelihood_clf.predict(wbiases)[0]
    data["prclass"] = preds

    return data


def main():
    pass


if __name__ == "__main__":
    main()

