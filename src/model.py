import numpy as np
from scipy.stats import norm
from tqdm import trange
import concurrent.futures


class LikelihoodModelForNormalDist:
    def __init__(self, t1, t2, word_biases, threshold_limit=0) -> None:
        self.t1 = t1
        self.t2 = t2
        self.thresholds = []
        self.threshold_limit = threshold_limit
        self.word_biases = word_biases

        self.U1 = None
        self.U2 = None
        self.U3 = None

    def fit_total_pop(self, total_pop):
        sorted_data = np.sort(total_pop)
        quantiles = np.quantile(sorted_data, [self.t1, self.t2])
        self.U2 = sorted_data[sorted_data <= quantiles[0]]
        self.Un = sorted_data[
            (sorted_data > quantiles[0]) & (sorted_data < quantiles[1])
        ]
        self.U1 = sorted_data[sorted_data > quantiles[1]]

        return (self.U1, self.U2, self.Un)

    def predict(self, t1, t2, total_pop):
        c1_biased = []
        c2_biased = []
        neutral = []
        predictions = []

        sorted_data = np.sort(total_pop)
        quantiles = np.quantile(sorted_data, [t1, t2])
        self.U2 = sorted_data[sorted_data <= quantiles[0]]
        self.Un = sorted_data[
            (sorted_data > quantiles[0]) & (sorted_data < quantiles[1])
        ]
        self.U1 = sorted_data[sorted_data > quantiles[1]]

        for ind in trange(len(self.word_biases)):
            sample = self.word_biases[ind]

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

    def ensemble_predict(self):
        # for t1, t2 in self.thresholds:
        #     stats.append(self.predict(word_biases, t1, t2))
        with concurrent.futures.ProcessPoolExecutor() as exec:
            stats = exec.map(self.predict, self.thresholds)

            for res in stats:
                print(res)

        return stats

    def prepare_thresholds(self, t1_range, t2_range, std_deviation, array_size):
        thresh_dist_1 = np.random.normal(self.t1, std_deviation, array_size)
        thresh_dist_2 = np.random.normal(self.t2, std_deviation, array_size)
        thresh_collection_1 = np.clip(thresh_dist_1, *t1_range)
        thresh_collection_2 = np.clip(thresh_dist_2, *t2_range)

        return (thresh_collection_1, thresh_collection_2)

