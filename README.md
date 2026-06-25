
# Pipeline expérimental H-Neurons

## Description

Ce notebook implémente l'ensemble du pipeline expérimental utilisé pour identifier, sélectionner et manipuler des neurones associés à certains comportements dans les grands modèles de langage (LLMs) (la factualité, les hallucinations et le'incertitude), en s'appuyant sur l'approche H-Neurons de Gao et al.(2025).

Le pipeline permet :

* de générer des réponses à partir du dataset TriviaQA ;
* d'extraire les activations internes du modèle ;
* d'entraîner un classifieur comportemental ;
* d'identifier les neurones les plus importants ;
* d'effectuer des interventions neuronales ;
* de générer de nouvelles réponses après intervention ;
* d'évaluer quantitativement l'impact de ces interventions.

Le notebook est présenté avec l'exemple du modèle **Mistral-7B-v0.3**, mais il peut être adapté à tout autre modèle compatible avec Hugging Face Transformers.

---

# Structure du pipeline

Le notebook suit les étapes suivantes :

## 1. Chargement du modèle

Chargement du tokenizer et du modèle de langage.

Exemple :

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
```

---

## 2. Génération des réponses

Le modèle génère une réponse pour chaque question du benchmark sélectionné.

Les réponses sont sauvegardées afin de constituer le jeu de données utilisé dans les étapes suivantes.

---

## 3. Préparation des données

Les réponses sont associées à leurs annotations comportementales puis :

* nettoyées ;
* filtrées ;
* équilibrées entre les différentes classes.

---

## 4. Extraction des activations

Les activations cachées sont extraites à partir des couches du transformeur.

Ces activations constituent les caractéristiques utilisées pour entraîner le classifieur.

Les activations sont sauvegardées afin d'éviter de devoir les recalculer lors des exécutions ultérieures.

---

## 5. Entraînement du classifieur

Un classifieur de régression logistique est entraîné afin d'identifier les neurones les plus discriminants.

Le notebook supporte deux configurations :

### Régression logistique L1

Favorise la sélection d'un nombre réduit de neurones fortement discriminants.

Exemple :

```python
LogisticRegression(
    penalty="l1",
    solver="saga"
)
```

### Régression logistique L2

Répartit l'importance sur un ensemble plus large de neurones.

Exemple :

```python
LogisticRegression(
    penalty="l2"
)
```

---

## 6. Sélection des neurones

Les coefficients du classifieur sont utilisés pour :

* classer les neurones ;
* sélectionner les neurones les plus importants ;
* construire les ensembles de neurones utilisés lors des interventions.

---

## 7. Intervention neuronale

Les neurones sélectionnés sont modifiés pendant l'inférence.

Plusieurs configurations peuvent être testées en faisant varier la force d'intervention α.

---

## 8. Génération après intervention

Le modèle génère de nouvelles réponses après modification des activations neuronales.

Chaque configuration est exécutée séparément.

Les résultats sont enregistrés dans des fichiers CSV et JSONL.

---

## 9. Évaluation

Les réponses générées sont comparées au ground truth.

Le notebook calcule ensuite les métriques nécessaires à l'analyse expérimentale, notamment le Compliance Rate.

---

# Utilisation avec un autre modèle

Le notebook est fourni avec l'exemple :

```python
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
```

Pour utiliser un autre modèle, remplacer :

```python
MODEL_NAME
```

et

```python
MODEL_PATH
```

par le nom du modèle Hugging Face souhaité.

Exemples :

```python
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
```

```python
MODEL_NAME = "Qwen/Qwen2-3B"
```

Après modification du modèle :

1. relancer la génération des réponses ;
2. relancer l'extraction des activations ;
3. réentraîner le classifieur ;
4. recalculer les neurones sélectionnés.

Les activations étant spécifiques à chaque modèle, elles ne peuvent pas être réutilisées d'un modèle à un autre.

---

# Passage de L1 à L2

Le changement de stratégie de sélection neuronale se fait au niveau de la définition de la régression logistique.

Version L1 :

```python
LogisticRegression(
    penalty="l1",
    solver="saga"
)
```

Version L2 :

```python
LogisticRegression(
    penalty="l2"
)
```

Après modification :

* réentraîner le classifieur ;
* recalculer les neurones importants ;
* relancer les expériences d'intervention.

Les neurones sélectionnés avec L1 et L2 sont généralement différents.

---

# Fichiers générés

Le notebook produit notamment :

* les réponses générées ;
* les activations neuronales ;
* les résultats du classifieur ;
* les neurones sélectionnés ;
* les réponses après intervention ;
* les métriques d'évaluation ;
* les Compliance Rates.

---

# Dépôt d'origine

Ce travail est basé sur le dépôt H-Neurons :

https://github.com/thunlp/H-Neurons

L'implémentation a été adaptée et étendue afin de permettre l'exécution du pipeline complet sur plusieurs modèles et plusieurs benchmarks comportementaux.

---

# Exécution recommandée

Les cellules doivent être exécutées dans l'ordre du notebook.

Lorsqu'un nouveau modèle est utilisé ou lorsqu'on passe d'une sélection L1 à une sélection L2, il est recommandé de relancer l'ensemble du pipeline à partir de l'étape d'extraction des activations.
