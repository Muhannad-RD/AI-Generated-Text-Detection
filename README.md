# AI-Generated Text Detection: TF-IDF and Linguistic Feature Fusion

## Project Overview
This project focuses on developing a lightweight and highly accurate machine learning framework to distinguish between human-written and AI-generated text. As Large Language Models (LLMs) advance, maintaining academic integrity and digital authenticity has become increasingly challenging. This study introduces a hybrid feature engineering architecture that merges deep lexical features with superficial grammatical patterns to outperform standard baseline models.

## Modeling & Methodology
The pipeline implements a robust text preprocessing and advanced feature fusion mechanism evaluated via classical machine learning:
1. **Feature Engineering & Fusion:** * **Lexical Features:** Optimized Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to capture deep n-gram word patterns.
   * **Syntactic Stylometric Features:** Part-of-Speech (POS) tagging densities extracted via **spaCy** (specifically targeting Verbs, Adjectives, and Pronouns) to capture structural anomalies typical of AI writing styles.
2. **Classifier (Logistic Regression):** A highly interpretable, standardized Logistic Regression model trained on the stacked feature matrices, offering rapid inference speed and concrete feature importance metrics.

## 📊 Performance: Feature Fusion

| Metric | Hybrid Feature Fusion |
| :--- | :--- |
| **Accuracy** | 93% |
| **Precision** | 93% |
| **Recall** | 93% |
| **F1-Score** | 0.93 |

> *Note: The results above demonstrate that fusing lexical token hierarchies with superficial grammatical patterns effectively mitigates dataset noise and yields a highly robust classification mechanism.*

## Repository Contents
* `Main code.ipynb`: The complete Jupyter Notebook implementation, including dataset cleaning, spaCy POS density extraction, vector stacking, hyperparameter tuning, and ROC-AUC plotting.
* `AI-Generated Text Detection.pdf`: The comprehensive final technical report detailing the academic framework, related literature, thorough error analysis, and confusion matrices.

## Tech Stack
* **Languages:** Python
* **NLP Tools:** spaCy (`en_core_web_sm`)
* **ML Frameworks:** Scikit-learn (Logistic Regression, TF-IDF Vectorizer)
* **Data Science & Storage:** Pandas, NumPy, SciPy (Sparse Matrices), Matplotlib

## Future Directions
* Expanding the training matrix to include multi-LLM corpora like the DAIGT v2 dataset to reduce single-model bias.
* Extending the English-only limitation by training language-specific pipelines or leveraging multilingual pre-trained embeddings (e.g., mBERT).
* Evaluating the pipeline's robustness against advanced adversarial paraphrasing techniques and lightly edited AI prompts.

## Author
* **Muhannad Al-Raddadi** - Data Science & Analytics Student

## License
This project is licensed under the **MIT License**.
