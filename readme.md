# Recommendation Framework with Community Edge Suppression

A comprehensive recommendation system framework that implements multiple collaborative filtering algorithms with built-in community bias analysis and community edge suppression capabilities. This is the code of the Bachelor Thesis of Leon Orou, the study can be fount [here]([https://jkulinz-my.sharepoint.com/:b:/g/personal/k12125027_students_jku_at/EeJ1QR-L3ABNka_SjrS4c5QBLB6CEzWsbnYfJvz39J3meQ?e=202xNb](https://1drv.ms/b/c/b834fc234b1005bc/EYNLKO-QNqlDgBLGALry0UsBBtB_0dMTpGBBFBCKUM7Gdg?e=df0t7t)).

## Usage

```bash
pip install -r requirements.txt
```

```bash
python main.py [OPTIONS]
```

### Command Line Arguments

#### Model and Dataset Selection
- `--model_name` (default: 'MultiVAE')
  - **Choices**: 'LightGCN', 'ItemKNN', 'MultiVAE'
  - **Description**: Specifies which recommendation algorithm to use

- `--dataset_name` (default: 'ml-100k')
  - **Choices**: 'ml-100k', 'ml-1m', 'lastfm'
  - **Description**: Dataset to use for training and evaluation

#### Community Detection Parameters
- `--users_top_percent` (default: 0.05)
  - **Range**: 0.0 - 1.0
  - **Description**: Percentage of top-connected users to consider as power nodes

- `--items_top_percent` (default: 0.00)
  - **Range**: 0.0 - 1.0
  - **Description**: Percentage of top-connected items to consider as power nodes

#### Bias Suppression Parameters
- `--users_dec_perc_suppr` (default: 0.625)
  - **Range**: 0.0 - 1.0
  - **Description**: Percentage of biased user connections to suppress for bias mitigation. Biased edges are usually ~60% of all edges

- `--items_dec_perc_suppr` (default: 0.0)
  - **Range**: 0.0 - 1.0
  - **Description**: Percentage of biased item connections to suppress for bias mitigation. Biased edges are usually ~60% of all edges

- `--community_suppression` (default: 0.8)
  - **Range**: 0.0 - 1.0
  - **Description**: Strength of community-based edge suppression (higher = more suppression)

#### Suppression Strategy
- `--suppress_power_nodes_first` (default: 'True')
  - **Choices**: 'True', 'False'
  - **Description**: Whether to prioritize suppressing connections of highly-connected nodes

- `--use_suppression` (default: 'True')
  - **Choices**: 'True', 'False'
  - **Description**: Enable/disable community bias suppression entirely

## Output

The system generates:
- **Metrics**: NDCG, Recall, Recall, MRR, Hit Rate, Item Coverage, Gini Index, Average Recomended Popularity, Popularity Lift, Popularity Miscalibration, Simpson Index (of item genres), Intra List Diversity (of item genres), Normalized Genre Entropy, Unique Genres Recommended, User Community Bias
- **Logs**: Generates configuration-, fold- and fold average results logs in logs/ folder
