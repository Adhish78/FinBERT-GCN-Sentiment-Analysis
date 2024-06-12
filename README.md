# Advanced Sentiment Analysis using FinBERT and Graph Convolutional Networks

## Project Overview
This project combines the power of FinBERT, a sentiment analysis model fine-tuned for financial text, with Graph Convolutional Networks (GCNs) to perform advanced sentiment analysis. The goal is to predict sentiment categories of financial text data using both contextual encoding from FinBERT and structural information from dependency graphs.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Pre-processing](#data-pre-processing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [FinBERT Context Encoding](#finbert-context-encoding)
- [Dependency Graph Construction](#dependency-graph-construction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Sentiment Analysis using FinBERT](#sentiment-analysis-using-finbert)

## Introduction
The project involves:
1. Loading and preprocessing multiple datasets related to financial sentiment analysis.
2. Combining the datasets into a single dataset.
3. Encoding the textual data using FinBERT.
4. Constructing dependency graphs for each sentence.
5. Training and evaluating a Graph Convolutional Network (GCN) on the encoded data.
6. Comparing the GCN results with the FinBERT sentiment analysis pipeline.

## Dataset
This project uses the following datasets:
1. FiQA 2018 Task1
    - **Source**: This dataset was developed for the WWW’18 conference, focusing on financial opinion mining. It can be accessed through the publication by Maia et al. (2018).
        - Maia, M., Handschuh, S., Freitas, A., Davis, B., McDermott, R., Manel Zarrouk and Balahur, A. (2018) ‘WWW’18 Open Challenge’, ARAN (University of Galway Research Repository) (Ollscoil na Gaillimhe – University of Galway). Available at: https://doi.org/10.1145/3184558.3192301.
    - **Content**: The FiQA 2018 dataset comprises two parts:
        * The first part contains 438 financial headlines, each associated with sentiment scores. For this project, the headlines and their sentiment scores are specifically utilized.
        * The second part includes 675 financial posts/messages, again with accompanying sentiment scores.
    - **Format**: Both parts of the dataset are provided in JSON format, facilitating ease of processing and integration.

2. SemEval-2017 Task 5 - Subtask 2
    - **Source**: This dataset is part of the SemEval-2017 Task 5 challenge, which focuses on fine-grained sentiment analysis on financial microblogs and news. The dataset is detailed in a publication by Cortis et al. (2017).
        - Cortis, K., Freitas, L., Daudert, T., M. Huerlimann, Manel Zarrouk, Handschuh, S. and Davis, B.J. (2017) ‘SemEval-2017 Task 5: Fine-Grained Sentiment Analysis on Financial Microblogs and News’, Association for Computational Linguistics. Available at: https://doi.org/10.18653/v1/s17-2089.
    - **Content**: It consists of 1142 financial headlines and news items, each annotated with sentiment scores.
    - **Format**: The dataset is available in JSON format, aligning with the format used in the FiQA dataset.

3. Financial Phrase Bank
    - **Source**: The Financial Phrase Bank dataset, detailed in a study by Malo et al. (2014), was sourced from financial news in the LexisNexis database.
        - Malo, P., Sinha, A., Korhonen, P., Wallenius, J. and Takala, P. (2014) ‘Good debt or bad debt: Detecting semantic orientations in economic texts’, Journal of the Association for Information Science and Technology, 65, pp. 782–796. Available at: https://doi.org/10.1002/asi.23062.
    - **Content**: It encompasses 4846 sentences, each meticulously categorized into positive, negative, or neutral sentiments by annotators experienced in financial markets.
    - **Access and Format**: The Financial Phrase Bank dataset was accessed using the Hugging Face ‘datasets’ library’s ‘load_dataset’ method directly into the Pandas DataFrame. This approach ensures a streamlined and efficient retrieval of the dataset.

## Requirements
- Python 3.x
- Google Colab (for running the notebook)
- Libraries: `pandas`, `matplotlib`, `seaborn`, `numpy`, `re`, `contractions`, `spacy`, `torch`, `transformers`, `torch-geometric`, `scikit-learn`, `optuna`

## Data Pre-processing
1. **Loading and Cleaning Data**: Load JSON data files, extract relevant fields, and clean the text.
2. **Sentiment Mapping**: Convert sentiment scores to categorical labels (negative, neutral, positive).
3. **Combining Datasets**: Merge multiple datasets into a single dataset.
4. **Text Cleaning**: Expand contractions, remove URLs, usernames, and special characters from the text.

## Exploratory Data Analysis
1. **Histograms**: Plot sentiment score distributions for each dataset.
2. **Count Plots**: Visualize the frequency of sentiment categories in the combined dataset.

## FinBERT Context Encoding
1. **Tokenization**: Tokenize sentences using FinBERT tokenizer.
2. **Contextual Embedding**: Encode sentences to obtain embeddings using FinBERT.
3. **Padding**: Ensure all embeddings have a uniform shape by padding sentences.

## Dependency Graph Construction
1. **Dependency Parsing**: Use spaCy to parse sentences and extract dependency relations.
2. **Adjacency Matrices**: Construct adjacency matrices representing dependency graphs for each sentence.
3. **Padding Adjacency Matrices**: Pad adjacency matrices to ensure uniform size.

## Model Training and Evaluation
1. **Dataset Preparation**: Convert data into PyTorch Geometric `Data` objects.
2. **Train-Test Split**: Split the dataset into training and testing sets.
3. **GCN Model**: Define and train a GCN model using `GCNConv` layers.
4. **Hyperparameter Tuning**: Use Optuna for hyperparameter tuning.
5. **Evaluation**: Evaluate the model using accuracy, classification report, and confusion matrix.

## Sentiment Analysis using FinBERT
1. **Pipeline Setup**: Use the FinBERT model for sentiment analysis.
2. **Sentiment Prediction**: Predict sentiment for the combined dataset.
3. **Evaluation**: Compare FinBERT predictions with actual labels using accuracy, classification report, and confusion matrix.



