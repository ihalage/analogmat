## Supplementary Materials for "Analogical discovery of disordered perovskites by crystal structure information hidden in unsupervised material fingerprints"

### Achintha Ihalage, Yang Hao

## Installation

The main project requires `python3.6`. Make sure you have the `pip3` module installed. The web scraping tool requires `python2.7.

1. Inside the root directory (analogmat), execute `pip install --ignore-installed -r requirements.txt` to install the dependencies.
2. If you wish to use the web scraper, execute  `pip2 install --ignore-installed -r requirements2.txt` command as well.
3. This should install all packages required to run analogmat on your machine. Please open an issue if installation errors occur.

## Usage 

### Evaluation of Machine learning classification models

from analogmat.ML.classification import PVClassifier

```python

clf = PVClassifier()
clf.train_and_test(algo='gradient_boosting')  # algo`: {‘gradient_boosting’, ‘random_forest’, ‘decision_tree’, '`svm`}, default=’gradient_boosting’
clf.plot_confusion_matrix(algo='gradient_boosting') # 10-fold CV confusion matrix

```

### Screening potential perovskites from the composition space

```python

clf.get_perovskite_candidates(prob_threshold=0.95, no_iterations=100)

```
