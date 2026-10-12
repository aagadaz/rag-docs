# Évaluation de la recherche

35 questions annotées · corpus : 472 chunks · k = 4

| Configuration | Hit rate@4 | MRR@4 |
|---|---|---|
| Similarité pure | 60% | 0.44 |
| MMR λ=0.5 | 54% | 0.41 |
| MMR λ=0.7 (réglage actuel) | 60% | 0.44 |
| Réécriture LLM (EN + API) | 100% | 0.87 |
| HyDE | 66% | 0.60 |
| Réécriture + MMR λ=0.7 (application) | 97% | 0.86 |
| Similarité pure, questions en anglais | 71% | 0.60 |

## Détail par question

| Question | Similarité pure | MMR λ=0.5 | MMR λ=0.7 (réglage actuel) | Réécriture LLM (EN + API) | HyDE | Réécriture + MMR λ=0.7 (application) | Similarité pure, questions en anglais |
|---|---|---|---|---|---|---|---|
| comment calculer la moyenne par groupe ? | 1 | 1 | 1 | 1 | ✗ | 1 | 1 |
| comment appliquer plusieurs fonctions d'agrégation en même temps sur un groupby ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| comment donner un nom personnalisé aux colonnes produites par une agrégation ? | 1 | 1 | 1 | 1 | 1 | 2 | ✗ |
| comment ajouter la moyenne du groupe sur chaque ligne sans changer la taille du DataFrame ? | 3 | ✗ | 3 | 3 | ✗ | 2 | ✗ |
| comment ne garder que les groupes qui contiennent plus de deux lignes ? | ✗ | ✗ | ✗ | 1 | ✗ | 1 | 4 |
| comment récupérer les deux premières lignes de chaque groupe ? | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| comment parcourir les groupes un par un dans une boucle ? | ✗ | ✗ | ✗ | 1 | 2 | 1 | 3 |
| comment extraire un seul groupe à partir de sa clé ? | ✗ | 2 | 3 | 3 | ✗ | ✗ | ✗ |
| les lignes dont la clé de regroupement vaut NaN sont-elles ignorées ? | 2 | ✗ | ✗ | 1 | 1 | 1 | 1 |
| comment numéroter les lignes à l'intérieur de chaque groupe ? | ✗ | ✗ | ✗ | 3 | ✗ | 3 | 4 |
| how to compute a rolling window within each group? | 1 | 1 | 1 | 3 | ✗ | 3 | 1 |
| à quoi sert la méthode nth sur un GroupBy ? | 2 | 3 | 2 | 1 | ✗ | 1 | 3 |
| comment empiler deux DataFrames l'un sous l'autre ? | 3 | ✗ | 2 | 1 | 1 | 1 | ✗ |
| comment faire une jointure gauche comme un LEFT JOIN en SQL ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| comment savoir de quelle table provient chaque ligne après une fusion ? | ✗ | ✗ | ✗ | 1 | ✗ | 1 | ✗ |
| comment fusionner sur la date la plus proche plutôt que sur une égalité exacte ? | ✗ | ✗ | ✗ | 1 | 1 | 1 | 1 |
| comment afficher les différences entre deux DataFrames ? | 2 | 2 | 2 | 1 | ✗ | 1 | ✗ |
| d'où viennent les suffixes _x et _y après un merge ? | 2 | ✗ | ✗ | 1 | ✗ | 1 | 1 |
| comment passer un tableau du format large au format long ? | 2 | ✗ | 2 | 1 | 2 | 1 | 1 |
| comment créer un tableau croisé dynamique avec des totaux de lignes et de colonnes ? | ✗ | 3 | ✗ | 1 | 1 | 1 | 1 |
| comment encoder une colonne catégorielle en variables indicatrices (one-hot) ? | ✗ | ✗ | ✗ | 1 | 1 | 1 | 4 |
| comment découper une variable numérique continue en intervalles ? | 4 | 2 | 2 | 1 | 1 | 1 | ✗ |
| comment transformer une colonne qui contient des listes en plusieurs lignes ? | ✗ | ✗ | ✗ | 4 | 3 | 2 | ✗ |
| comment supprimer les lignes qui contiennent des valeurs manquantes ? | ✗ | ✗ | ✗ | 1 | 1 | 1 | ✗ |
| comment remplir les trous d'une série par interpolation linéaire ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| que représente pd.NA et en quoi diffère-t-il de NaN ? | 1 | 1 | 1 | 1 | ✗ | 1 | 1 |
| comment couper une chaîne de caractères selon un séparateur dans une colonne ? | ✗ | ✗ | ✗ | 1 | 1 | 1 | 1 |
| comment extraire avec une expression régulière une partie du texte de chaque ligne ? | 1 | 1 | 1 | 2 | 2 | 2 | 1 |
| comment tester si une colonne de texte contient un mot donné ? | 2 | 2 | 2 | 1 | 1 | 1 | 2 |
| comment convertir une colonne de chaînes en dates avec un format précis ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| comment générer une suite de dates jour par jour entre deux bornes ? | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| comment agréger des mesures horaires en valeurs journalières ? | ✗ | ✗ | ✗ | 1 | 1 | 1 | 1 |
| comment décaler une série temporelle d'une période vers le futur ? | ✗ | ✗ | ✗ | 1 | 1 | 1 | ✗ |
| comment convertir des horodatages vers un autre fuseau horaire ? | ✗ | 4 | 4 | 1 | 1 | 1 | 1 |
| comment définir des jours ouvrés personnalisés en excluant des jours fériés ? | 1 | 1 | 1 | 2 | ✗ | 2 | 1 |

Rang du premier passage pertinent (1 = en tête) ; ✗ = absent des 4 passages.
