````markdown
# Analyse des données hospitalières et modélisation prédictive

## Présentation du projet

Ce projet porte sur l’analyse d’un jeu de données hospitalières dans le but de mieux comprendre les facteurs expliquant les coûts d’hospitalisation et les durées de séjour des patients.

L’étude combine :
- une analyse exploratoire des données,
- une analyse avancée des hospitalisations et des coûts,
- une modélisation prédictive en machine learning,
- la génération automatisée d’un rapport analytique en HTML.

L’objectif global est de produire des insights utiles pour l’aide à la décision dans un contexte de gestion hospitalière.

---

## Objectifs

Les principaux objectifs du projet sont les suivants :

1. Décrire et explorer les données hospitalières.
2. Identifier les départements, maladies et traitements les plus fréquents.
3. Étudier les déterminants de la durée de séjour.
4. Analyser les facteurs expliquant les coûts hospitaliers.
5. Construire des modèles de machine learning pour :
   - prédire le coût d’hospitalisation (régression),
   - prédire les séjours longs (classification).
6. Générer un rapport analytique automatique et structuré.

---

## Jeu de données

Le dataset contient des informations sur **500 patients hospitalisés**.

### Variables principales
- `Age`
- `Sexe`
- `Departement`
- `Maladie`
- `Traitement`
- `DureeSejour`
- `Cout`
- `DateAdmission`
- `DateSortie`

### Variables dérivées
- `DureeCalculee`
- `Ecart`
- `CoutParJour`
- `MoisAdmission`
- `SejourLong`
- `TrancheAge`
- `Anomalie`
- `AnomalieLabel`

---

## Structure du projet

```bash
Données Hospitalières/
│
├── data/
│   └── hospital_data.csv
│
├── notebooks/
│   └── tp_hospital_analysis.ipynb
│
├── scripts/
│   └── generate_report.py
│
├── reports/
│   └── rapport_hospitalisation.html
│
├── .venv/
│
├── requirements.txt
└── README.md
````

---

## Étapes du projet

### 1. Analyse exploratoire

L’analyse exploratoire a permis d’étudier :

* la distribution de l’âge,
* la répartition par sexe,
* les départements les plus sollicités,
* les maladies les plus fréquentes,
* la distribution de la durée de séjour,
* la distribution des coûts.

### 2. Analyse avancée

Une analyse plus approfondie a ensuite permis d’identifier :

* la durée moyenne de séjour par département,
* les maladies associées aux longs séjours,
* la relation entre l’âge et la durée de séjour,
* la cohérence des dates d’admission et de sortie,
* le coût moyen par département,
* le coût moyen par maladie,
* la relation entre coût et durée de séjour,
* le coût journalier par département,
* les cas extrêmes de coûts,
* les anomalies détectées par Isolation Forest.

### 3. Machine Learning

Deux approches ont été mises en œuvre :

#### Option 1 — Régression

Prédiction du coût d’hospitalisation à partir de :

* l’âge,
* le sexe,
* le département,
* la maladie,
* le traitement,
* la durée de séjour.

Modèles testés :

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Indicateurs d’évaluation :

* MAE
* RMSE
* R²

#### Option 2 — Classification

Prédiction des séjours longs, définis comme :

* `1` si `DureeSejour > moyenne`
* `0` sinon

Modèles testés :

* Logistic Regression
* Random Forest Classifier
* Decision Tree Classifier

Indicateurs d’évaluation :

* Accuracy
* Precision
* Recall
* F1-score

---

## Principaux résultats

### Régression — Prédiction du coût

| Modèle            |    MAE |    RMSE |   R² |
| ----------------- | -----: | ------: | ---: |
| Linear Regression | 792.10 | 1016.21 | 0.82 |
| Random Forest     | 773.61 | 1029.66 | 0.81 |
| Gradient Boosting | 768.56 |  974.48 | 0.83 |

**Meilleur modèle : Gradient Boosting**

### Classification — Prédiction des séjours longs

| Modèle              | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |     1.00 |      1.00 |   1.00 |     1.00 |
| Random Forest       |     1.00 |      1.00 |   1.00 |     1.00 |
| Decision Tree       |     1.00 |      1.00 |   1.00 |     1.00 |

### Remarque méthodologique importante

Les performances parfaites obtenues en classification s’expliquent par une **fuite de données (data leakage)**, car la variable cible `SejourLong` a été construite à partir de `DureeSejour`, qui a également été utilisée comme variable explicative.

Cette limite a été identifiée et explicitement discutée dans le projet.

---

## Interprétation des résultats

Les analyses réalisées montrent que :

* la **durée de séjour** est le principal facteur expliquant les coûts hospitaliers ;
* certaines pathologies complexes sont davantage associées aux séjours prolongés ;
* les départements spécialisés concentrent souvent des coûts et durées plus élevés ;
* les modèles de régression sont globalement fiables ;
* la classification doit être interprétée avec prudence à cause du data leakage.

---

## Recommandations

À partir des résultats obtenus, plusieurs pistes d’optimisation peuvent être proposées :

* réduire les durées de séjour lorsque cela est médicalement possible ;
* cibler les pathologies et services générant les coûts les plus élevés ;
* surveiller les cas extrêmes et anomalies ;
* utiliser les modèles prédictifs pour anticiper certains besoins hospitaliers ;
* enrichir les données futures avec des variables cliniques supplémentaires.

---

## Génération du rapport

Le rapport analytique final est généré automatiquement à partir du script Python :

```bash
python scripts/generate_report.py
```

Le rapport généré est disponible ici :

```bash
reports/rapport_hospitalisation.html
```

Pour l’ouvrir dans le navigateur :

```bash
start reports/rapport_hospitalisation.html
```

---

## Installation

### 1. Cloner le projet

```bash
git clone <lien-du-repo>
cd "Données Hospitalières"
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Lancer le notebook

```bash
code .
```

Puis ouvrir :

```bash
notebooks/tp_hospital_analysis.ipynb
```

---

## Technologies utilisées

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* VS Code
* Jupyter Notebook

---

## Démo en ligne

Application déployée sur Render: 
[Voir le dashboard en ligne] --> https://data-visualisation-et-machine-learning.onrender.com

---

## Déploiement

Le projet a été déployé sur Render avec:
- Build Command : `pip install -r requirements.txt`
- Start Command : `gunicorn app:server`

## Auteur

** Ndeye Madeleine DIALLO CDSD M2 ISM**

Projet réalisé dans le cadre d’un travail académique en **Data Visualisation d'Analyse Prédictive**.

---

````


