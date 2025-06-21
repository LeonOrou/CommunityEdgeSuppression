

# load genre labels of ml-100k dataset and save them in a dictionary and locally
import os
import json
from collections import Counter, defaultdict

genre_mapping = {
        "0": "unknown",
        "1": "Action",
        "2": "Adventure",
        "3": "Animation",
        "4": "Children's",
        "5": "Comedy",
        "6": "Crime",
        "7": "Documentary",
        "8": "Drama",
        "9": "Fantasy",
        "10": "Film-Noir",
        "11": "Horror",
        "12": "Musical",
        "13": "Mystery",
        "14": "Romance",
        "15": "Sci-Fi",
        "16": "Thriller",
        "17": "War",
        "18": "Western",
        "(no genres listed)": 0,
        "Action": 1,
        "Adventure": 2,
        "Animation": 3,
        "Children's": 4,
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

def save_ml100k_genre_labels(file_path, genre_mapping, dataset_name='ml-100k', keep_labels=False, use_three_genres=False):
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

    if keep_labels:
        # format {movie_string: [genre1, genre2, ...]}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                item_name = parts[1]
                genres = parts[-19:]
                genre_id_list = [str(i) for i, label in enumerate(genres) if label == '1']
                genre_labels[item_name] = [genre_mapping[genre] for genre in genre_id_list if genre in genre_mapping]
    elif use_three_genres:
        item_genres_labels = json.load(open(f'dataset/{dataset_name}/saved/item_genre_labels_{dataset_name}_labelsTrue_top3.json', ))

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if parts[0] == 'movieId':
                    continue
                item_id = int(parts[0])
                item_name = parts[1]
                genre_labels_list = item_genres_labels[item_name]
                if len(genre_labels_list) > 3:  #sanity check
                    print(f"Item {item_name} has more than 3 genres: {genre_labels_list}. Truncating to first 3 genres.")
                #get label ids from genre labels
                genre_labels[item_id] = [genre_mapping[genre] for genre in genre_labels_list if genre in genre_mapping]
    else:
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
    output_file = f'dataset/{dataset_name}/saved/item_genre_labels_{dataset_name}_labels{keep_labels}.json'
    with open(output_file, 'w') as json_file:
        json.dump(genre_labels, json_file, indent=4)
    return genre_labels


def save_lfm_genre_labels(input_file="dataset/lastfm/user_taggedartists.dat", dataset_name='lfm'):
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

def save_ml1m_genre_labels(file_path, genre_mapping, dataset_name='ml-1m', keep_labels=False, use_three_genres=True):
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
    if keep_labels:
        with open(file_path, 'rb') as f:
            for line in f.read().splitlines():
                parts = line.decode('latin-1').strip().split('::')
                item_name = parts[1]
                genres_txt = parts[-1]
                genre_labels[item_name] = genres_txt.split('|')
    else:
        with open(file_path, 'rb') as f:
            if use_three_genres:
                # load json 3 genres file
                item_genres_labels = json.load(open('dataset/ml-1m/saved/item_genre_labels_ml-1m_names.json', ))
            for line in f.read().splitlines():
                parts = line.decode('latin-1').strip().split('::')
                if parts[0] == 'movieId':
                    continue  # Skip header line
                item_id = int(parts[0])
                if use_three_genres:
                    item_name = parts[1]
                    genre_labels_list = item_genres_labels[item_name]
                    if len(genre_labels_list) > 3:
                        print(f"Item {item_name} has more than 3 genres: {genre_labels_list}. Truncating to first 3 genres.")
                    genre_labels[item_id] = genre_labels_list
                genres_txt = parts[-1]
                genres_txt_list = genres_txt.split('|')
                # if len(genres_txt_list) > 3:
                #     genres_txt_list = genres_txt_list[:3]
                genre_labels_list = [genre_mapping[genre] for genre in genres_txt_list]
                genre_labels[item_id] = genre_labels_list

    # Save the genre labels to a local JSON file
    output_file = f'dataset/{dataset_name}/saved/item_genre_labels_{dataset_name}_labels{keep_labels}_3genres{use_three_genres}.json'
    with open(output_file, 'w') as json_file:
        json.dump(genre_labels, json_file, indent=4)
    return genre_labels


# save_ml100k_genre_labels('dataset/ml-100k/u.item', genre_mapping=genre_mapping, use_three_genres=True)
# save_lfm_genre_labels('dataset/lastfm/user_taggedartists.dat')
# save_ml1m_genre_labels('dataset/ml-1m/movies.dat', genre_mapping=genre_mapping, use_three_genres=True)

