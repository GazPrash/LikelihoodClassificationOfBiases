import numpy as np
import numpy as np
from scipy.stats import norm
from tqdm import trange


class LikelihoodModelForNormalDist:
    def __init__(self, t1, t2, threshold_limit=0) -> None:
        self.t1 = t1
        self.t2 = t2
        self.thresholds = []
        self.threshold_limit = threshold_limit

        self.U1 = None
        self.U2 = None
        self.Un = None

    def ensemble_thresholds():
        # process for generating thresholds
        for t1, t2 in self.thresholds:
            return self.fit_total_pop() # SOMETHING like this

    def fit_total_pop(self, total_pop):
        sorted_data = np.sort(total_pop)
        quantiles = np.quantile(sorted_data, [self.t1, self.t2])
        self.U2 = sorted_data[sorted_data <= quantiles[0]]
        self.Un = sorted_data[
            (sorted_data > quantiles[0]) & (sorted_data < quantiles[1])
        ]
        self.U1 = sorted_data[sorted_data > quantiles[1]]
        # self.U2 = sorted_data[sorted_data <= self.t1]
        # self.Un = sorted_data[(sorted_data > self.t1) & (sorted_data < self.t2)]
        # self.U1 = sorted_data[sorted_data >= self.t2]

        return (self.U1, self.U2, self.Un)

    def predict(self, word_biases):
        c1_biased = []
        c2_biased = []
        neutral = []
        predictions = []

        for ind in trange(len(word_biases)):
            sample = word_biases[ind]

            likelihood1 = np.sum(
                (norm.logpdf(sample, np.mean(self.U1), np.std(self.U1)))
            )
            likelihood2 = np.sum(
                (norm.logpdf(sample, np.mean(self.U2), np.std(self.U2)))
            )
            likelihood3 = np.sum(
                (norm.logpdf(sample, np.mean(self.Un), np.std(self.Un)))
            )

            all_likelihoods = [likelihood1, likelihood2, likelihood3]
            best_fit_population = np.argmax(all_likelihoods)

            if abs(max(all_likelihoods)) < self.threshold_limit:
                neutral.append((ind, max(all_likelihoods)))
                predictions.append(2)
                continue

            predictions.append(best_fit_population)
            if best_fit_population == 2:
                neutral.append((ind, all_likelihoods))
            elif best_fit_population == 0 and max(all_likelihoods):
                c1_biased.append((ind, all_likelihoods))
            else:
                c2_biased.append((ind, all_likelihoods))

        return (predictions, c1_biased, c2_biased, neutral)
