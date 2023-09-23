# KES-2023
Datasets, sentiment analysis model, and results to support the paper "SART &amp; COVIDSentiRo: Datasets for Sentiment Analysis Applied to Analyzing COVID-19 Vaccination Perception in Romanian Tweets"

## Repo Description
- SART and COVIDSentiRO datasets are located in the `datasets` folder
- full paper can be read [here](https://github.com/Alegzandra/KES-2023/blob/main/SART%20%26%20COVIDSentiRo%20Datasets%20for%20Sentiment%20Analysis%20Applied%20to%20Analyzing%20COVID-19%20Vaccination%20Perception%20in%20Romanian%20Tweets.pdf)
- to train the BERT-based model use `train_bert.py` and to evaluate the model use `evaluate bert.py`. Both need `bert_torch_dataset_creator.py`
- to train the fasText-based model use `train_fasttext.py` and to evaluate the created model use `evaluate fasttext.py`
- to analyze correlations between weekly number of vaccinated people against COVID-19 and how Romanian Twitter users perception regarding COVID-19 vaccination changes in the time-frame January 2021 - February 2022, please use  `plots_sentiment_analysis.R`

## BibTeX entry and citation info
If you are using the code and/or data please cite:

```bash
@inproceedings{SART_COVIDSentiRO,
    title = "SART & COVIDSentiRo: Datasets for Sentiment Analysis Applied to Analyzing COVID-19 Vaccination Perception in Romanian Tweets",
    author = "Ciobotaru, Alexandra  and Dinu, Liviu P.",
    booktitle = "Proceedings of the 27th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2023)",
    month = sep,
    year = "2023",
    address = "Athens, Greece",
    abstract = "Vaccination is an important subject of discussion adjacent to the COVID-19 pandemic. Sentiments generated online by this topic are worth analyzing using opinion mining tools, and it is interesting to do so in online content written in an under-researched language, like Romanian. For this reason, we modified and enlarged an existing sentiment analysis dataset comprised of Romanian
tweets labeled as negative or positive. The resulting dataset, SART (Sentiment Analysis from Romanian Tweets), comprised of three classes (positive, negative, and neutral) containing 1300 Romanian tweets each, was used to train two different sentiment analysis models: a fastText-based one and a fine-tuned BERT model. We further show the usefulness of the sentiment analysis model by analyzing the sentiment of Romanian tweets regarding vaccination using a corpus created and collected by the authors between January 2021 and February 2022 (COVIDSentiRo).",
}
```
