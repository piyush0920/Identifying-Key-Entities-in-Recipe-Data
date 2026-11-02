# Identifying Key Entities in Recipe Data

Named Entity Recognition (NER) on raw recipe ingredient lists, using a **Conditional Random Field (CRF)** with hand-engineered token features. The model tags every word as an **ingredient**, **quantity** or **unit**, turning messy free-text lists into structured data.

> Built as part of the Executive PG Diploma in Data Science & AI (IIIT Bangalore, via upGrad).

---

## Business objective

Recipe sites, dietary-tracking apps and e-commerce platforms all need ingredient lists in a structured form: what is needed, how much, and in what unit. This project trains a sequence-labelling model that reads text such as

```
2-1/2 cups rice cooked 3 tomatoes teaspoons ... 1 teaspoon chickpea lentils ...
```

and labels each token:

```
2-1/2 -> quantity | cups -> unit | rice -> ingredient | cooked -> ingredient | 3 -> quantity | tomatoes -> ingredient ...
```

The output can feed recipe databases, nutrition calculators, shopping-list generators and product-matching systems.

---

## Dataset

`ingredient_and_quantity.json` holds 285 recipe ingredient lists. Each record has two fields:

| Field | Description |
|-------|-------------|
| `input` | Raw ingredient text from a recipe |
| `pos` | Space-separated label for each token in `input` (`ingredient`, `quantity` or `unit`) |

**Label distribution (raw data, 9,000+ tokens):**

| Label | Tokens |
|-------|-------:|
| ingredient | 7,543 |
| quantity | 1,414 |
| unit | 1,190 |

**Cleaning:** 28 records were removed, either because the token count of `input` did not match `pos` (indices 17, 27, 79, 164, 207) or because the text contained stray sentence words (`is`, `for`) that signal non-ingredient noise. The remaining **257 recipes** were split **70:30** (`random_state=42`) into **179 training** and **78 validation** recipes (6,241 and 2,772 tokens respectively).

---

## Approach

1. **Ingestion and validation:** load the JSON into pandas, tokenise `input` and `pos`, check that lengths align and drop malformed rows.
2. **Exploratory analysis:** flatten tokens, categorise them by label and plot the top 10 ingredients and units. The most frequent ingredient tokens include *powder*, *Salt*, *seeds*, *Oil* and *Red*. The most frequent units are *teaspoon*, *cup* and *tablespoon*.
3. **Feature engineering (`word2features`):** per-token features including
   - lexical: lowercased token, title/upper case, has digit, has alpha, hyphen and slash flags
   - quantity and unit detection: keyword sets for units (cup, tbsp, tsp, gram, clove, sprig...) and quantities (half, quarter, one...), plus a regex for integers, decimals and fractions
   - context: previous and next token, whether the neighbours are quantities or units, and beginning/end-of-sentence markers
4. **Class weighting:** inverse-frequency weights computed for the three labels to account for imbalance (ingredients dominate the data).
5. **Model:** `sklearn-crfsuite` CRF trained with L-BFGS.

| Hyperparameter | Value |
|----------------|-------|
| `algorithm` | `lbfgs` |
| `c1` (L1) | 0.5 |
| `c2` (L2) | 1.0 |
| `max_iterations` | 100 |
| `all_possible_transitions` | `True` |

6. **Evaluation and error analysis:** flat classification report, confusion matrices, a dataframe of every misclassified token with its surrounding context, and per-label accuracy alongside class weights.

---

## Results and evaluation caveat

The notebook's originally submitted validation report shows 99.93% token accuracy (2 errors among 2,772 validation tokens). **That figure is not a valid generalization result:** the committed implementation adds a class weight derived from each token's true label to that token's input feature dictionary. This leaks target information into the model features.

A separate diagnostic rerun with that target-derived feature removed reported the following validation results:

| Label | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| ingredient | 0.985 | 0.996 | 0.990 |
| quantity | 1.000 | 0.987 | 0.993 |
| unit | 0.972 | 0.922 | 0.946 |

Overall accuracy in that separate rerun was **98.6%**. This rerun is not yet reproduced by the committed notebook, so treat 98.6% as a corrected diagnostic result rather than a fully reproducible repository result. The next code correction is to remove the `features['weight'] = weight_dict[label]` input feature and apply any class weighting through a training-time weighting mechanism that does not expose the true label as an input.
## Limitations and next steps

- Small dataset (257 usable recipes), mostly Indian-cuisine ingredient lists, so performance on other cuisines or formats is untested.
- Labels use only three classes. Extending to `preparation` (chopped, diced), `size` or `state` would make the output more useful.
- Features look at the immediate neighbours only. Wider context windows, or comparing against a BiLSTM-CRF or a transformer-based token classifier, are natural next experiments.
- Rare multi-word ingredients are the main source of residual errors.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, sklearn-crfsuite, spaCy, Matplotlib, Seaborn, joblib

---

## Repository structure

```
.
├── Identifying_Key_Entities_in_Recipe_Data_piyush_sharma.ipynb  # EDA, features, CRF, evaluation
├── ingredient_and_quantity.json                                # labelled recipe data
└── README.md
```

---

## How to run

```bash
git clone https://github.com/piyush0920/Identifying-Key-Entities-in-Recipe-Data.git
cd Identifying-Key-Entities-in-Recipe-Data

pip install pandas numpy matplotlib seaborn scikit-learn joblib spacy sklearn_crfsuite==0.5.0
python -m spacy download en_core_web_sm

jupyter notebook Identifying_Key_Entities_in_Recipe_Data_piyush_sharma.ipynb
```

The notebook reads `ingredient_and_quantity.json` using a relative path, so keep the JSON file in the same folder as the notebook. Run the cells in order.

**Saved model note:** The notebook contains a `joblib.dump` step that creates `crf_model.pkl` when executed. That generated file is not currently committed in the repository. After running the notebook, load it with `joblib.load("crf_model.pkl")` and use the feature-building functions defined in the notebook.

---

## Author

**Piyush Sharma**
[LinkedIn](https://linkedin.com/in/piyush-sharma-6591171ba) · [GitHub](https://github.com/piyush0920) · [Portfolio](https://piyush0920.github.io/)
