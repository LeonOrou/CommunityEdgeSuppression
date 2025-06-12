

# load genre labels of ml-100k dataset and save them in a dictionary and locally
import os
import json
from collections import Counter, defaultdict


def save_ml100k_genre_labels(file_path, dataset_name='ml-100k'):
    """
    Load genre labels from a file and return them as a dictionary.

    Args:
        file_path (str): Path to the file containing genre labels.

    Returns:
        dict: A dictionary where keys are item IDs and values are lists of genre labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    genre_labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            item_id = int(parts[0])
            # The last 19 parts are genre labels
            # they are 0 or 1 indicating whether the item belongs to that genre
            genres = parts[-19:]
            # Convert to a list of genre labels with the label being the ith position
            genres = [i for i, label in enumerate(genres) if label == '1']
            # if len(genres) > 3:
            #     genres = genres[:3]
            genre_labels[item_id] = genres

    # Save the genre labels to a local JSON file
    output_file = f'dataset/{dataset_name}/saved/item_genre_labels_{dataset_name}.json'
    with open(output_file, 'w') as json_file:
        json.dump(genre_labels, json_file, indent=4)
    return genre_labels


def save_lfm_genre_labels(input_file="dataset/LFM1M/tags_all_music.tsv", dataset_name='lfm'):
    genre_counter = Counter()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] == 'track_id':
                continue
            if len(parts) >= 4 and parts[4:]:
                genre_counter.update(parts[4:])

    top_genres = [g for g, _ in genre_counter.most_common(100)]
    genre_to_id = {g: i + 1 for i, g in enumerate(top_genres)}

    item_genres = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4 and parts[4]:
                genres = parts[4:]
                genres = [genre_to_id[g] for g in genres if g in genre_to_id]
                if genres:
                    # in order to stay equivalent with the other datasets we also take all genres
                    # item_genres[parts[0]] = genres[:3]  # Limit to first 3 genres
                    item_genres[parts[0]] = genres

    with open(f'dataset/{dataset_name}/saved/item_genre_labels_{dataset_name}.json', 'w', encoding='utf-8') as f:
        json.dump(dict(item_genres), f)

def save_ml20m_genre_labels(file_path, dataset_name='ml-20m'):
    """
    Load genre labels from a file and return them as a dictionary.

    Args:
        file_path (str): Path to the file containing genre labels.

    Returns:
        dict: A dictionary where keys are item IDs and values are lists of genre labels.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    genre_mapping = {
        "(no genres listed)": 0,
        "Action": 1,
        "Adventure": 2,
        "Animation": 3,
        "Children": 4,
        "Comedy": 5,
        "Crime": 6,
        "Documentary": 7,
        "Drama": 8,
        "Fantasy": 9,
        "Film-Noir": 10,
        "Horror": 11,
        "Musical": 12,
        "Mystery": 13,
        "Romance": 14,
        "Sci-Fi": 15,
        "Thriller": 16,
        "War": 17,
        "Western": 18,
        "IMAX": 19
    }

    genre_labels = {}
    with open(file_path, 'r', encoding="utf8") as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == 'movieId':
                continue  # Skip header line
            item_id = int(parts[0])
            genres_txt = parts[-1]
            genres_txt_list = genres_txt.split('|')
            # if len(genres_txt_list) > 3:
            #     genres_txt_list = genres_txt_list[:3]
            genre_labels_list = [genre_mapping[genre] for genre in genres_txt_list]
            genre_labels[item_id] = genre_labels_list

    # Save the genre labels to a local JSON file
    output_file = f'dataset/{dataset_name}/saved/item_genre_labels_{dataset_name}.json'
    with open(output_file, 'w') as json_file:
        json.dump(genre_labels, json_file, indent=4)
    return genre_labels


# save_ml100k_genre_labels('dataset/ml-100k/u.item')
save_lfm_genre_labels('dataset/LFM1M/tags_all_music.tsv')
# save_ml20m_genre_labels('dataset/ml-20m/movies.csv')

