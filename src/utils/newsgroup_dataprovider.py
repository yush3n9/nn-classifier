from src.utils.dataset_provider_base import IterableDataProvider
from sklearn.datasets import fetch_20newsgroups


class TwentyNewsgroup(IterableDataProvider):
    def __init__(self, categories=['alt.atheism', 'sci.space']):
        self.categories = categories

    def inspect_dataset(self):
        pass

    def categories_size(self):
        return len(self.categories)

    def fetch_dataset_train(self):
        return fetch_20newsgroups(subset='train', categories=self.categories)

    def fetch_dataset_test(self):
        return fetch_20newsgroups(subset='test', categories=self.categories)
