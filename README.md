# Détection de fraude bancaire — Pipeline Data Science + CI/CD

Projet personnel : pipeline complet de détection de fraude sur des transactions
bancaires, du cadrage jusqu'à la mise en production, avec un pipeline CI/CD
(intégration continue, recette, mise en production) qui s'exécute réellement
sur GitHub Actions à chaque push — voir l'onglet **Actions** de ce dépôt.

## ⚠️ À propos des données

Les données utilisées (`data/transactions.csv`, générée au moment du build)
sont **générées synthétiquement** par `data/generate_data.py` — il ne s'agit
pas de données bancaires réelles (confidentialité oblige). Les features et la
logique de fraude s'inspirent de schémas réalistes couramment utilisés en
détection de fraude par carte bancaire (distance par rapport au domicile,
écart par rapport au montant habituel, utilisation de la puce/du code PIN,
achat en ligne, heure de la transaction...).

## Structure du projet

```
fraude_bancaire/
├── data/
│   └── generate_data.py       # génère le jeu de données synthétique
├── src/
│   ├── preprocessing.py       # chargement, split train/test, mise à l'échelle
│   ├── train.py               # entraînement du modèle (RandomForest)
│   ├── evaluate.py            # calcul des métriques (precision, recall, F1)
│   └── predict.py             # API Flask de scoring (mise en production)
├── tests/
│   ├── test_preprocessing.py  # tests unitaires du prétraitement
│   ├── test_model_performance.py  # test de non-régression (seuil F1)
│   └── test_acceptance.py     # tests de recette (scénarios métier sur l'API)
├── .github/workflows/ci-cd.yml  # pipeline CI/CD (GitHub Actions)
├── Dockerfile
└── requirements.txt
```

## Pipeline Data Science

1. **Cadrage** : détecter des transactions frauduleuses à partir de variables
   comportementales (distance, montant, canal utilisé, horaire...).
2. **Préparation des données** : nettoyage, split train/test stratifié, mise à
   l'échelle (`StandardScaler`).
3. **Modélisation** : `RandomForestClassifier` (scikit-learn), `class_weight="balanced"`
   pour tenir compte du déséquilibre des classes (~3,5 % de fraude).
4. **Évaluation** : precision, recall, F1-score, matrice de confusion.
5. **Déploiement** : API Flask (`/predict`) packagée dans une image Docker.

### Résultats obtenus (preuve — run réel, voir `models/metrics.json` généré en CI)

| Métrique | Valeur |
|---|---|
| Precision | 0.84 |
| Recall | 0.90 |
| F1-score | 0.869 |
| Échantillons de test | 4 000 (140 fraudes) |

Seuil minimal accepté en CI : F1-score ≥ 0.75 (`src/evaluate.py`).

## Utilisation en local

```bash
pip install -r requirements.txt
python3 data/generate_data.py      # génère transactions.csv
python3 src/train.py               # entraîne et sauvegarde le modèle
python3 -m pytest tests/ -v        # tests unitaires + non-régression
python3 src/predict.py             # démarre l'API sur http://localhost:5000
```

Exemple d'appel à l'API :

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "distance_from_home": 50, "distance_from_last_transaction": 30,
  "ratio_to_median_purchase_price": 5, "amount": 800, "repeat_retailer": 0,
  "used_chip": 0, "used_pin_number": 0, "online_order": 1, "transaction_hour": 3
}'
```

## Pipeline CI/CD (`.github/workflows/ci-cd.yml`)

Ce pipeline s'exécute réellement sur GitHub Actions à chaque push sur ce dépôt
(voir l'onglet **Actions**) :

| Étape | Déclencheur | Ce qu'elle fait |
|---|---|---|
| **1. Intégration continue** | chaque push / pull request | installe, génère les données, entraîne le modèle, exécute les tests unitaires **et** un test de non-régression (le F1-score ne doit pas descendre sous 0.75) |
| **2. Recette (UAT)** | push sur `main`, après succès de l'étape 1 | redémarre l'API avec le modèle entraîné et exécute des scénarios métier concrets (transaction à risque détectée, transaction normale non signalée, champ manquant rejeté) |
| **3. Mise en production** | push sur `main`, après succès de la recette | construit l'image Docker de l'API (modèle inclus) ; le push vers un registre réel (Docker Hub, etc.) est indiqué en commentaire dans le workflow, à activer une fois les secrets configurés |

Si une étape échoue, les suivantes ne se déclenchent pas : rien n'est déployé
tant que les tests et la recette ne sont pas validés.

## Limites connues (honnêteté)

- Les données sont synthétiques, pas des données bancaires réelles.
- L'étape "mise en production" construit l'image Docker mais ne la publie pas
  vers un registre réel (aucun identifiant de registre n'est configuré) — le
  workflow montre comment l'activer.
- Le Dockerfile n'a pas pu être testé avec un build Docker local pendant le
  développement (pas de démon Docker actif dans cet environnement), mais suit
  une structure standard qui fonctionne sur les runners GitHub Actions.
