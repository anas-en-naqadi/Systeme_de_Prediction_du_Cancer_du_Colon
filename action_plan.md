# Plan d'Action Final — Colon Cancer ML Project

## Vision du projet

Un système clinique professionnel de prédiction du cancer du côlon, avec une interface médicale soignée, une API robuste et un pipeline ML rigoureux, prêt pour un portfolio senior.

## Missions

### Mission 1 — Pipeline ML

Durée estimée : environ 30 min de génération.

```text
training/
├── train.py              ← orchestrateur principal
└── utils/
    ├── ffs.py            ← Forward Feature Selection
    ├── evaluate.py       ← métriques, confusion matrix, ROC curve
    └── logger.py         ← logs colorés dans le terminal
```

Ce qui se passe :

- Chargement et nettoyage du dataset
- FFS avec AUC-ROC en scoring (CV 5-fold)
- Sélection des 6 meilleurs gènes
- Split stratifié 70/15/15
- Comparaison LR vs SVM-Linear vs SVM-RBF
- GridSearchCV sur chaque modèle
- Évaluation finale sur un Eval set isolé
- Sauvegarde de model.pkl, scaler.pkl, selected_genes.json et model_metadata.json

### Mission 2 — Backend FastAPI

Dossier : app/backend/

```text
backend/
├── main.py               ← routes + CORS + static files
├── predictor.py          ← chargement artefacts + logique prédiction
└── schemas.py            ← modèles Pydantic typés
```

Routes :

| Méthode | Route | Description |
| --- | --- | --- |
| GET | / | welcome |
| GET | /health | status + modèle chargé + algo utilisé |
| GET | /genes | 6 gènes + métadonnées |
| GET | /metadata | métriques du modèle (AUC, F1, Recall...) |
| POST | /predict | prédiction + confidence + probabilités |

### Mission 3 — Frontend React

C'est ici que le projet se distingue visuellement.

```text
frontend/
├── index.html
├── vite.config.js
├── package.json
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── api/
    │   └── client.js           ← appels API centralisés
    ├── components/
    │   ├── Header.jsx           ← logo + titre clinique
    │   ├── GeneInputCard.jsx    ← carte input par gène
    │   ├── PredictionResult.jsx ← résultat animé
    │   ├── ConfidenceBar.jsx    ← barre de confiance animée
    │   ├── ModelBadge.jsx       ← algo + métriques du modèle
    │   └── LoadingSpinner.jsx
    └── styles/
        └── global.css
```

Design system — thème médical premium :

| Élément | Choix |
| --- | --- |
| Palette | Fond sombre #0A0F1E + accents #00D4FF (cyan médical) |
| Typography | Inter pour UI + mono pour valeurs génétiques |
| Layout | Dashboard 2 colonnes, inputs à gauche, résultat à droite |
| Animations | Framer Motion, entrées fluides, résultat animé |
| Gènes | Slider + input numérique par gène avec tooltip explicatif |
| Résultat | Card héroïque avec badge coloré (vert Normal / rouge Abnormal) |
| Confiance | Barre animée avec gradient couleur selon niveau |
| Métriques | Mini dashboard modèle (AUC, F1, Recall) en bas de page |

UX flow :

1. Chargement de la page
2. Skeleton loading pendant le fetch de /genes
3. 6 cards gènes apparaissent avec stagger animation
4. L'utilisateur ajuste les sliders
5. Bouton Analyze avec état loading
6. Le résultat apparaît avec animation hero
7. Badge Normal en vert ou Abnormal en rouge
8. Barre de confiance animée
9. Option Reset propre

### Mission 4 — Docker + Docker Compose

```text
colon-cancer-ml/
├── docker-compose.yml
├── training/
│   ├── Dockerfile        ← Python slim, génère model/
└── app/
    └── Dockerfile        ← build React → FastAPI sert le static
```

Architecture containers :

```text
┌─────────────────────────────────┐
│  docker compose run training    │
│  → génère model/ (volume)       │
└────────────────┬────────────────┘
                 │ volume partagé model/
┌────────────────▼────────────────┐
│  docker compose up app          │
│  ┌──────────┐  ┌─────────────┐  │
│  │  React   │  │   FastAPI   │  │
│  │  build   │→ │  port 8000  │  │
│  │ (static) │  │             │  │
│  └──────────┘  └─────────────┘  │
└─────────────────────────────────┘
```

Compatibilité Windows :

- `.dockerignore` propres
- Line endings gérés
- Volumes avec paths Windows-safe
- `--reload` désactivé en prod

### Mission 5 — README + GitHub

- Badges Python, FastAPI, React et Docker
- Screenshot du frontend
- Architecture diagram ASCII
- Getting Started en 3 commandes
- Explication scientifique FFS
- Métriques du modèle
- Structure du projet

## Structure finale complète

```text
colon-cancer-ml/
│
├── docker-compose.yml
├── README.md
├── .gitignore
│
├── model/                          ← généré par training
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── selected_genes.json
│   └── model_metadata.json
│
├── training/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py
│   ├── utils/
│   │   ├── ffs.py
│   │   ├── evaluate.py
│   │   └── logger.py
│   └── data/
│       └── colon_cancer_dataset.csv
│
└── app/
    ├── Dockerfile
    ├── requirements.txt
    │
    ├── backend/
    │   ├── main.py
    │   ├── predictor.py
    │   └── schemas.py
    │
    └── frontend/
        ├── index.html
        ├── vite.config.js
        ├── package.json
        └── src/
            ├── main.jsx
            ├── App.jsx
            ├── api/
            │   └── client.js
            ├── components/
            │   ├── Header.jsx
            │   ├── GeneInputCard.jsx
            │   ├── PredictionResult.jsx
            │   ├── ConfidenceBar.jsx
            │   ├── ModelBadge.jsx
            │   └── LoadingSpinner.jsx
            └── styles/
                └── global.css
```

## Ordre d'exécution final

```bash
# 1. Entraîner le modèle
docker compose run training

# 2. Lancer l'application
docker compose up --build

# 3. Ouvrir le navigateur
http://localhost:8000
```

## Récapitulatif des missions

| Mission | Contenu | Statut |
| --- | --- | --- |
| Mission 1 | Pipeline ML complet | ⏳ À faire |
| Mission 2 | Backend FastAPI | ⏳ À faire |
| Mission 3 | Frontend React WOW | ⏳ À faire |
| Mission 4 | Docker + Compose | ⏳ À faire |
| Mission 5 | README + GitHub | ⏳ À faire |