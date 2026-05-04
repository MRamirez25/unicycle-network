# unicycle-network

The code for the currently working unicycle reservoir is in `unicycle_network.py`, and you can run it for several benchmarks with the scripts in the `evaluation_datasets` folder. The files called `<dataset>_evaluation_statistics.py` can be run directly, and will train a model 5 times with 5 different random seeds and provide the mean and standard deviation at the end. The file `utils.py` has several dataloaders, heavily based on the same file in the [RON project](https://github.com/AndreaCossu/RandomizedCoupledOscillators/tree/master).

All of these scripts are loading an [optuna](https://github.com/optuna/optuna) database from the `optuna_databases` folder, and loading the best hyperparameters for that dataset. 

The files called `optuna_<dataset>` were used to run optuna optimizations to find these hyperaparameters.

# Dependencies
* [NumPy](https://numpy.org/install/)
* [PyTorch](https://pytorch.org/get-started/locally/) (With CUDA if you want to use your GPU)
* [Scikit-learn](https://scikit-learn.org/stable/install.html)
* [Optuna](https://github.com/optuna/optuna)
* [tqdm](https://github.com/tqdm/tqdm)
* [matplotlib](https://matplotlib.org/stable/install/index.html)