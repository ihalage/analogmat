<img alt="GitHub" src="https://img.shields.io/github/license/ihalage/analogmat">

## Supplementary Materials for "Analogical discovery of disordered perovskites by crystal structure information hidden in unsupervised material fingerprints"

### Achintha Ihalage and Yang Hao

## Installation

The main project requires `python3.6`. Make sure you have the `pip3` module installed. The web scraping tool requires `python2.7`.

1. Inside the root directory (analogmat), execute `pip install --ignore-installed -r requirements.txt` to install the dependencies.
   (Note that installing pymatgen and tensorflow using pip may produce errors. In this case, please follow the installation documentations of [pymatgen](https://pymatgen.org/installation.html) and [tensorflow](https://www.tensorflow.org/install/pip) for clean installation)
2. If you wish to use the web scraper, execute  `pip2 install -r requirements2.txt` command as well.
3. This should install all packages required to run analogmat on your machine. Please open an issue if installation errors occur.

## Usage 

### Evaluation of machine learning classification models



```python
from ML.classification import PVClassifier
clf = PVClassifier()
clf.train_and_test(algo='gradient_boosting')  # algo`: {‘gradient_boosting’, ‘random_forest’, ‘decision_tree’, '`svm`}, default=’gradient_boosting’
clf.plot_confusion_matrix(algo='gradient_boosting') # 10-fold CV confusion matrix
clf.plot_roc_curve()  # 10-fold CV ROC curve
```

Confusion matrix             |  ROC curve
:-------------------------:|:-------------------------:
![Alt text](analogmat/figures/gradient_boosting_conf_mat.png?raw=true) | ![Alt text](analogmat/figures/roc_curve.png?raw=true)

### Plotting the perovskite likelihood of selected A(B<sub>1-x</sub>B'<sub>x</sub>)O<sub>3</sub> compositions

```python
from ML.plot_results import ABBO3_Viz
viz = ABBO3_Viz()
viz.plot_Bdoped()
```

![alt text](https://github.com/ihalage/analogmat/blob/master/analogmat/figures/clf_results_Bdoped.png)

### Screening potential perovskites from the composition space

The database of total possible compositions is too large for github (332Mb). You can download this database [here](https://figshare.com/articles/dataset/all_generated_compounds_csv/13033262). Place this file inside `ICSD_data` directory and execute the following code.

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
from autoencoder import AutoEncoder
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

Following snippet can be used to predict the crystal system and space group of 2104 experimental compositions based on the plurality vote of 5 nearest neighbours in the fingerprint space.

```python
from validate_fingerprints import CrystalSystem
cc = CrystalSystem()
cc.validate()  # prediction of crystal system and space group
cc.get_confusion_matrix()
cc.get_spg_conf_mat()
```
Crystal system             |  Space group
:-------------------------:|:-------------------------:
![Alt text](analogmat/figures/fingerprint_conf_mat.png?raw=true) | ![Alt text](analogmat/figures/fingerprint_spg_conf_mat.png?raw=true)

Next, we assess the capability of supervised machine learning algorithms to classify crystal system and space group of 2104 experimental compositions with 10-fold cross validation.

```python
from ML.crystal_system_clf import StructureClf
sclf = StructureClf()
sclf.crystal_system_clf(algo='knn')  # 10-fold CV   # algo`: {‘gradient_boosting’, ‘random_forest’, ‘decision_tree’, '`svm`}, default=’gradient_boosting’
sclf.cross_val_conf_mat(algo='knn')
sclf.spg_clf(algo='gradient_boosting')    # space group classification
```


The fingerprint spaces obtained by VAE and vanilla autoencoder for the experimental database can be visualized with;

```python
from autoencoder import AutoEncoder
from plot_df import Fingerprints
ae = AutoEncoder()
fprints = Fingerprints(ae)
fprints.plot_fingerprints(model='vae')    # model='ae' for vanilla autoencoder
```
Variational autoencoder             |  Vanilla autoencoder
:-------------------------:|:-------------------------:
![Alt text](analogmat/figures/vae_fingerprints.png?raw=true) | ![Alt text](analogmat/figures/ae_fingerprints.png?raw=true)


T-SNE and PCA are widely used dimensionality reduction algorithms. High dimensional discrete material features can be projected to two-dimensions (2D) using these algorithms and visualized as follows.

```python
fprints.plot_pca_tsne(algo='tsne')  # algo = 'pca' to visualize with PCA algorithm
```
t-SNE             |  PCA
:-------------------------:|:-------------------------:
![Alt text](analogmat/figures/tsne_visualisation.png?raw=true) | ![Alt text](analogmat/figures/pca_visualisation.png?raw=true)

We can retrain the autoencoders as follows. This will overwrite the existing model. The parameters can be changed from the code.

```python
from autoencoder import AutoEncoder
ae = AutoEncoder()
ae.train(vae=True)
```

## Web scraper

The tool is implemented in `python2.7` to scrape the Bing search engine. This would require numpy, scipy, pandas, monty and pymatgen versions compatible with `python2.7` as listed in `requirements2.txt` file. The usage is as follows.

First, navigate to the `web_scraper` directory. Next, run the following python2 program.
```python
from bing_scraper import BingScraper
bs = BingScraper()
result = bs.scrape_compound('(Ba0.5Sr0.5)TiO3')
print result
```

```
########################################
(Ba0.5Sr0.5)TiO3  is found on web!!! 
See below for results


# TITLE: <h2>(PDF) Dielectric properties of (Ba0.5Sr0.5)TiO3 thin films ...</h2>
#
# DESCRIPTION: <p>Dielectric properties of (Ba0.5Sr0.5)TiO3 thin films</p>
# ___________________________________________________________
#
# TITLE: <h2>Dielectric properties of (Ba0.5Sr0.5)TiO3 thin films - CORE</h2>
#
# DESCRIPTION: <p>The dielectric properties of (Ba0.5Sr0.5)TiO3 (BST) thin films with high electrical resistivity were investigated. BST films are deposited on Pt/TiO2/SiO2/Si substrates by a metal-organic deposition (MOD) method. The dielectric permittivity and ac conductivity of the films are measured in the frequency range 102-105 Hz. The dielectric permittivity εr decreases slightly with frequency f ...</p>
# ___________________________________________________________
#
# TITLE: <h2>Leakage current of (Ba0.5Sr0.5)TiO3 thin film ... - CORE</h2>
#
# DESCRIPTION: <p>The leakage current and relative permittivity of (Ba0.5Sr0.5)TiO3 (BST) thin films prepared by pulsed-laser deposition (PLD) were investigated. It was found that the leakage current for positive bias voltage was higher than that for negative bias voltage, which was attributed to the lattice mismatch between the bottom Pt electrode and the BST thin film. A time-dependent breakdown process under ...</p>
# ___________________________________________________________
#
# TITLE: <h2>Dielectric Properties and Leakage Current Characteristics ...</h2>
#
# DESCRIPTION: <p>The X-ray studies indicated that both MT and Ba0.5Sr0.5TiO3 are highly oriented and remain as two distinct individual entities in the composite films and a considerable reduction in the dielectric loss and leakage currents has been observed.</p>
# ___________________________________________________________

...
...
...
```

## Questions and comments
Please contact a.a.ihalage@qmul.ac.uk or y.hao@qmul.ac.uk.

## Funding
We acknowledge funding received by The Institution of Engineering and Technology (IET) under the AF Harvey Research Prize.
