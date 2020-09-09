<img alt="GitHub" src="https://img.shields.io/github/license/ihalage/analogmat">

## Supplementary Materials for "Analogical discovery of disordered perovskites by crystal structure information hidden in unsupervised material fingerprints"

### Achintha Ihalage, Yang Hao

## Installation

The main project requires `python3.6`. Make sure you have the `pip3` module installed. The web scraping tool requires `python2.7`.

1. Inside the root directory (analogmat), execute `pip install --ignore-installed -r requirements.txt` to install the dependencies.
2. If you wish to use the web scraper, execute  `pip2 install --ignore-installed -r requirements2.txt` command as well.
3. This should install all packages required to run analogmat on your machine. Please open an issue if installation errors occur.

## Usage 

### Evaluation of Machine learning classification models



```python
from analogmat.ML.classification import PVClassifier
clf = PVClassifier()
clf.train_and_test(algo='gradient_boosting')  # algo`: {‘gradient_boosting’, ‘random_forest’, ‘decision_tree’, '`svm`}, default=’gradient_boosting’
clf.plot_confusion_matrix(algo='gradient_boosting') # 10-fold CV confusion matrix
clf.plot_roc_curve()  # 10-fold CV ROC curve


```

### Screening potential perovskites from the composition space

```python

clf.get_perovskite_candidates(prob_threshold=0.95, no_iterations=100)
```

```

100%|████████████████████████████████████████████████████████████| 100/100 [03:43<00:00,  2.23s/it]
 
##################### Classification Results ###############################

46228 new perovskite candidates were found out of 591129 hypothetical compounds!
92.18 % of total compounds were discarded!

############################################################################


```

## Autoencoders (VAE & vanilla)

*Autoencoder* class implements the unsupervised material fingerprinting model. Materials analogies can be investigated in a bi-directional manner. That is, "What are the analogous experimental materials to an arbitary composition?" (enabling crystal structure prediction) and "What are the analogous unstudied perovskites to a target experimental material?" (enabling analogical materials discovery).

Following is the code snippet to find 5 experimental analogies (nearest neighbours-NNs) to the composition (K<sub>0.5</sub>Bi<sub>0.5</sub>)ZrO<sub>3</sub>.
It is important to write the chemical formula in standard notation with brackets to identify the disordered site - (A<sub>1-x</sub>A'<sub>x</sub>)BO<sub>3</sub> or A(B<sub>1-x</sub>B'<sub>x</sub>)O<sub>3</sub>. Note that R<sub>A</sub> should be greater than R<sub>B</sub>.

```python
from analogmat.autoencoder import AutoEncoder
ae = AutoEncoder()
model = ae.build_AE(vae=True)  # vae=False for vanilla autoencoder
model.load_weights('saved_models/best_model_VAE.h5')  # best_model_AE.h5 for vanilla autoencoder
exp_analogs = ae.most_similar(model, '(K0.5Bi0.5)ZrO3',  n=5, vae=True)
print (exp_analogs)
```

```
   CollectionCode        HMS CrystalSystem      StructuredFormula  Euclidean Distance
0           92640  P 4/m m m    tetragonal  (K0.667Th0.333)(TiO3)            0.064202
1           28621  P 4/m m m    tetragonal     (Ba0.8Pb0.2)(TiO3)            0.076766
2          291164    P 4 m m    tetragonal     (Ba0.95Pb0.05)TiO3            0.080477
3          157807    P 4 m m    tetragonal   (Ba0.67Pb0.33)(TiO3)            0.083409
4            5513    P 4 m m    tetragonal        (K0.5Bi0.5)TiO3            0.116703
```

Unstudied compositions to a target material, for example (Ba<sub>0.5</sub>Nd<sub>0.5</sub>)MnO<sub>3</sub> can be obtained as follows.

```python
cand_analogs = ae.most_similar_cand_perovskites(model, '(Ba0.5Nd0.5)MnO3',  n=5)
print (cand_analogs)
```

```
    StructuredFormula  Mean_classification_prob  Euclidean Distance
0  (Ba0.35Gd0.65)MnO3                  0.990954            0.018537
1  (Sr0.95Pb0.05)TcO3                  0.985223            0.025213
2     (K0.5Yb0.5)HfO3                  0.984748            0.027341
3  (Ba0.45Eu0.55)MnO3                  0.984248            0.028650
4    (Sr0.9Pb0.1)TcO3                  0.985134            0.030750
```



We can bypass the new compositions having toxic or expensive elements with `except_elems` argument. A probability threshold can also be set. For example, relaxor ferroelectric Pb(Mg<sub>0.33</sub>Nb<sub>0.67</sub>)O<sub>3</sub> as the target material;

```python
cand_analogs = ae.most_similar_cand_perovskites(model, 'Pb(Mg0.33Nb0.67)O3', except_elems=[ 'Tl', 'Pb', 'Hg',  'Cd'], prob_threshold=0.80, n=5, vae=True)
print (cand_analogs)
```

```
     StructuredFormula  Mean_classification_prob  Euclidean Distance
0     Bi(Sc0.2Ni0.8)O3                  0.841687            0.013217
1     Bi(Sc0.2Co0.8)O3                  0.837163            0.017750
9   Bi(Ti0.55Cr0.45)O3                  0.896630            0.071381
12   Bi(Ti0.95V0.05)O3                  0.873715            0.073236
13  Bi(Ti0.75Cr0.25)O3                  0.873786            0.075084

```
