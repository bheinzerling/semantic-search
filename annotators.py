from dataclasses import dataclass
from collections import defaultdict
from itertools import permutations, islice

from boltons.iterutils import windowed

from util import (
    cached_property,
    SubclassRegistry,
    avg,
    flatten,
    )


@dataclass
class TextAnnotator:
    def __call__(self):
        raise NotImplementedError


@dataclass
class SpacyAnnotator:
    spacy_model: str

    @cached_property
    def model(self):
        import spacy
        return spacy.load(self.spacy_model)

    def __call__(self, text):
        return self.model(text)


@dataclass
class TextualUnitAnnotator(SubclassRegistry):
    def __call__(self, parsed_text):
        raise NotImplementedError

    @property
    def key(self):
        raise NotImplementedError

    @staticmethod
    def from_key_type(key_type):
        return TextualUnitAnnotator.get(key_type + 'annotator')()


class SentenceAnnotator(TextualUnitAnnotator):
    def __call__(self, parsed_text):
        units = [sent.text for sent in parsed_text['annot'].sents]
        return {'sentences': units}


class NgramsAnnotator(TextualUnitAnnotator):
    ngram_orders: list[int] = [1, 2, 3]

    def __call__(self, parsed_text):
        ngramss = {}
        sents = parsed_text['annot'].sents
        tokens = [token.text for sent in sents for token in sent]
        for order in self.ngram_orders:
            ngrams = windowed(tokens, order) if len(tokens) > order else []
            ngramss[f'{order}-grams'] = ngrams
        return ngramss


class CatenaAnnotator(TextualUnitAnnotator):
    max_catena_len: int = 5
    max_component_len: int = 10
    catena_top_k: int = None
    component_top_k: int = None
    components_only: bool = False

    def __call__(self, parsed_text):
        def path_to_root(token):
            while True:
                yield token
                if token.head.i == token.i:
                    break
                token = token.head

        non_informative_pos_tags = {'AUX', 'DET', 'PUNCT'}

        all_catenae = []

        for parsed_sent in parsed_text['annot'].sents:

            monotonic_paths = [list(path_to_root(t)) for t in parsed_sent]

            depths = list(map(len, monotonic_paths))
            max_depth = max(depths)

            def score_token(token):
                if token.pos_ in non_informative_pos_tags:
                    return 2 * max_depth
                return depths[token.i]

            def score_path(path):
                return avg([score_token(t) for t in path])

            token_idx2paths = defaultdict(list)
            for path in monotonic_paths:
                for i in range(len(path)):
                    token_idx2paths[path[i].i].append(path[:i + 1])

            pairs = permutations(
                [p for p in token_idx2paths[3] if len(p) > 1],
                r=2)

            for token_idx, paths in token_idx2paths.items():
                paths = [p for p in paths if len(p) > 1]
                pairs = (
                    (p1, p2)
                    for i, p1 in enumerate(paths[:-1])
                    for j, p2 in enumerate(paths[i + 1:], start=i + 1)
                    )
                for p1, p2 in pairs:
                    if p1[-2].i != p2[-2].i:
                        updown_path = p1 + p2[:-1]
                        token_idx2paths[token_idx].append(updown_path)

            paths = flatten(token_idx2paths.values())

            if self.max_catena_len:
                paths = filter(lambda p: len(p) <= self.max_catena_len, paths)
            elif self.max_component_len and self.components_only:
                paths = filter(
                    lambda p: len(p) <= self.max_component_len, paths)
            scored_paths = sorted(paths, key=score_path)
            catenae = (sorted(p, key=lambda t: t.i) for p in scored_paths)
            if self.components_only:
                def is_component(c):
                    return [t.i for t in c] == list(range(c[0].i, c[-1].i + 1))

                catenae = filter(is_component, catenae)
                catenae = islice(catenae, self.component_top_k)
            else:
                catenae = islice(catenae, self.catena_top_k)
            all_catenae.extend(catenae)
        return {'catenae': all_catenae}


class ComponentAnnotator(CatenaAnnotator):
    components_only: bool = True

    def __call__(self, parsed_text):
        components = super().__call__(parsed_text)['catenae']
        return {'components': components}
