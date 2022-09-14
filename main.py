from pathlib import Path
from tqdm import tqdm

from util import lines, flatten
from annotators import SpacyAnnotator, TextualUnitAnnotator
from dense_index import DenseIndex


def load_raw_data(data_file, max_texts=None):
    return [
        dict(id=i, text=text)
        for i, text in enumerate(lines(data_file, max=max_texts))
        ]


def load_data(data_file, annotator, max_texts=None):
    data = load_raw_data(data_file, max_texts=max_texts)
    for inst in tqdm(data):
        inst['annot'] = annotator(inst['text'])
    return data


def index_data(data, key_types):
    annotators = list(map(TextualUnitAnnotator.from_key_type, key_types))
    for inst in data:
        inst['keys'] = {}
        for annotator in annotators:
            keys = annotator(inst)
            inst['keys'].update(keys)


def to_keys_and_ids(data, token_sep=''):
    keys_with_ids = []
    for inst in data:
        inst_id = inst['id']
        for key_idx, key in enumerate(flatten(inst['keys'].values())):
            key = token_sep.join(str(token) for token in key)
            key_id = {'inst_id': inst_id, 'key_idx': key_idx}
            keys_with_ids.append((key, key_id))
    return zip(*keys_with_ids)


if __name__ == '__main__':
    # example data from a Japanese UD treebank
    data_file = Path('data/ja_gsd-ud-dev.txt')
    # small number of texts since this is just a toy example
    max_texts = 10

    # use a spacy model for Japanese
    spacy_model = 'ja_core_news_lg'
    index_name = data_file.name
    # use a sentence transforemr model that supports Japanese
    transformer_model = 'xlm-r-bert-base-nli-stsb-mean-tokens'

    # parse texts
    annotator = SpacyAnnotator(spacy_model)
    data = load_data(data_file, annotator, max_texts)

    # extract various types of "keys" associated with each text, e.g. n-grams
    # supported  types are ['sentence', 'ngrams', 'catena', 'component']
    key_types = ['ngrams']
    index_data(data, key_types)
    keys, key_ids = to_keys_and_ids(data)

    # create a dense index by encoding keys with a pretrained language model
    index = DenseIndex(
        name=index_name,
        keys=keys,
        faiss_indexkey='flatip',
        transformer_model=transformer_model,
        )

    query = '方法'
    max_hits = 10
    # this will perform a dense search:
    # 1. encode the query with the same language model that was used
    # 2. run a nearest neigbor search
    # 3. return matches along with their distances to the query (or the inner product in case of an IP index)
    matches = index(query, k=max_hits)

    # get the original text in which the best match was found
    best_match = matches[0]
    key_id = key_ids[best_match.obj['key_idx']]
    inst = data[key_id['inst_id']]
    print('query', query, 'best match:', inst['text'])
