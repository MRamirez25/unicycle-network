import tensorflow_datasets as tfds
import numpy as np

class MNISTDataLoader:
    def __init__(self, bs_train, bs_test, classes=None, new_fraction=1.0, test_fraction=1.0, seed=0):
        self.bs_train = bs_train
        self.bs_test = bs_test
        self.classes = classes
        self.new_fraction = new_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self._prepare_data()

    def _prepare_data(self):
        ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], as_supervised=True, batch_size=-1)

        def ds_to_numpy(ds):
            images, labels = tfds.as_numpy(ds)
            images = images.astype(np.float32) / 255.0
            images = images.reshape(-1, 28, 28, 1)  # NHWC format
            return images, labels

        X_train, y_train = ds_to_numpy(ds_train)
        X_test, y_test = ds_to_numpy(ds_test)

        if self.classes is not None:
            class_set = set(self.classes)
            train_mask = np.isin(y_train, list(class_set))
            test_mask = np.isin(y_test, list(class_set))
            X_train, y_train = X_train[train_mask], y_train[train_mask]
            X_test, y_test = X_test[test_mask], y_test[test_mask]

        rng = np.random.default_rng(self.seed)
        train_size = int(self.new_fraction * len(X_train))
        test_size = int(self.test_fraction * len(X_test))

        train_idx = rng.choice(len(X_train), size=train_size, replace=False)
        test_idx = rng.choice(len(X_test), size=test_size, replace=False)

        X_train, y_train = X_train[train_idx], y_train[train_idx]
        X_test, y_test = X_test[test_idx], y_test[test_idx]

        n_val = int(0.1 * len(X_train))
        self.X_valid, self.y_valid = X_train[:n_val], y_train[:n_val]
        self.X_train, self.y_train = X_train[n_val:], y_train[n_val:]
        self.X_test, self.y_test = X_test, y_test

    def get_batches(self, split='train'):
        if split == 'train':
            X, y, bs = self.X_train, self.y_train, self.bs_train
            shuffle = True
        elif split == 'valid':
            X, y, bs = self.X_valid, self.y_valid, self.bs_test
            shuffle = False
        elif split == 'test':
            X, y, bs = self.X_test, self.y_test, self.bs_test
            shuffle = False
        else:
            raise ValueError(f"Unknown split: {split}")

        N = len(X)
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, N - bs + 1, bs):
            batch_idx = idx[start:start + bs]
            yield X[batch_idx], y[batch_idx]

    def get_full_dataset(self, split='train'):
        if split == 'train':
            return self.X_train, self.y_train
        elif split == 'valid':
            return self.X_valid, self.y_valid
        elif split == 'test':
            return self.X_test, self.y_test
        else:
            raise ValueError(f"Unknown split: {split}")
