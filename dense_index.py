from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path
import argparse

from boltons.iterutils import remap

from dougu import (
    cached_property,
    flatten,
    )


def argsparser():
    desc = 'TODO'
    a = argparse.ArgumentParser(description=desc)
    arg = a.add_argument
    arg('--infile', type=Path, required=True)
    arg('--gpu-id', type=int, default=0)
    return a


@total_ordering
class ObjectWithScore:
    __slots__ = ['obj', 'score']

    def __init__(self, obj, score):
        self.obj = obj
        self.score = score

    def __eq__(self, other):
        return self.obj == other.obj

    def __hash__(self):
        return hash(self.obj)

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f'[{self.score:.2f} {self.obj}]'


@dataclass
class DenseIndex:
    name: str
    keys: list[str]
    transformer_model: str
    # Index type for dense search. See FAISS documentation for details
    faiss_indexkey: str = 'ivfflat'
    # set to space for writing systems with white-space separated tokens
    # batch size for encoding keys
    batch_size: int = 5000
    # FAISS index on GPU is faster, but has limit of k=2048 nearest neighbors
    index_on_gpu: bool = True
    gpu_id: int = 0

    @cached_property
    def larger_score_is_better(self):
        # larger scores are better in case of inner product indexes, but worrse
        # for L2 indexes
        return self.faiss_indexkey == 'flatip'

    @property
    def conf_str(self):
        fields = [
            self.name,
            self.transformer_model,
            ]
        return '.'.join(fields)

    @property
    def vectors(self):
        print(f'encoding {len(self.keys)} keys')
        return self.encode(self.keys).cpu().numpy()

    @cached_property
    def index(self):
        import faiss
        d = self.vectors.shape[1]
        nlist = 100
        index_key = self.faiss_indexkey
        print('creating index ' + index_key)
        if index_key == 'flatip':
            index = faiss.IndexFlatIP(d)
        elif index_key == 'ivfflat':
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(self.vectors)
        else:
            index = faiss.index_factory(d, self.faiss_indexkey)
            index.train(self.vectors)
        if self.index_on_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
        index.add(self.vectors)
        print('created index')
        return index

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(
            self.transformer_model,
            device=self.gpu_id,
            )

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(texts)

    def encode_complex(self, query):
        import torch

        def encode_conjunct(conjunct):
            enc_pos = self._encode(conjunct['pos'])[0]
            if conjunct['neg']:
                enc_neg = self._encode(conjunct['neg']).sum(dim=0)
                enc_pos -= enc_neg
            return enc_pos

        def encode_disjunct(disjunct):
            conjuncts = torch.stack(
                list(map(encode_conjunct, disjunct['conjuncts'])), dim=0)
            if conjuncts.dim() > 1:
                conjuncts = conjuncts.mean(dim=0)
            return conjuncts.unsqueeze(0).cpu().numpy()

        return list(map(encode_disjunct, query['disjuncts']))

    def _encode(self, texts):
        import torch
        if len(texts) > 500000:
            from tqdm import tqdm
            from boltons.iterutils import chunked
            chunks = tqdm(chunked(texts, 10000))
        else:
            chunks = [texts]
        chunks_enc = [
            self.encoder.encode(
                chunk,
                batch_size=self.batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=False,
                ).cpu()
            for chunk in chunks]
        enc = torch.cat(chunks_enc, dim=0)
        if enc.dim() == 1:
            enc = enc.unsqueeze(0)
        return enc

    def __call__(
            self,
            query,
            *,
            k=None,
            j=100,
            min_score=None,
            max_score=None,
            vector_algebra=False):
        if isinstance(query, str):
            query = self.parse_query(query)
            if vector_algebra:
                queries_enc = self.encode_complex(query)
            elif isinstance(query, dict):
                def visit(path, key, value):
                    if isinstance(value, str):
                        query_enc = self.encode(value).cpu().numpy()
                        value = self.index_search(
                            query_enc,
                            k=j,
                            min_score=min_score,
                            max_score=max_score,
                            )
                    return key, value

                return self.aggregate_matches(remap(query, visit))
            else:
                query_enc = self.encode(query).cpu().numpy()
                queries_enc = [query_enc]
        else:
            # already encoded
            query_enc = query
            queries_enc = [query_enc]

        matches = []
        for query_enc in queries_enc:
            match = self.index_search(
                query_enc, k=j, min_score=min_score, max_score=max_score)
            matches.append(match)
        return self.aggregate_matches(matches, k=k)

    def aggregate_matches(self, matches, k=None):
        if isinstance(matches, dict):
            def match_hash(m):
                return (m.obj['text_id'], m.obj['sent_id'])

            def visit_disj(path, key, value):
                if key == 'disjuncts':
                    value = [
                        [conj for conj in v['conjuncts']]
                        for v in value
                        ]
                return key, value

            def visit_conj(path, key, value):
                if key == 'conjuncts':
                    def subtract(conj):
                        neg_hashes = set(map(match_hash, conj['neg']))
                        return [
                            p for p in conj['pos']
                            if match_hash(p) not in neg_hashes]
                    value = [
                        subtract(conj)
                        for conj in value
                        ]
                    if len(value) > 1:
                        pos_hashess = [
                            set(map(match_hash, conj))
                            for conj in value[1:]]
                        value = [
                            v for v in value[0]
                            if all(
                                match_hash(v) in pos_hashes
                                for pos_hashes in pos_hashess)
                            ]
                    else:
                        value = value[0]
                return key, value

            def visit_neg(path, key, value):
                if key == 'neg':
                    value = list(flatten(value))
                return key, value

            matches = remap(matches, visit_neg)
            matches = remap(matches, visit_conj)
            matches = remap(matches, visit_disj)['disjuncts']
        if len(matches) == 1:
            matches = matches[0]
        else:
            matches = list(flatten(matches))
        print(f'{len(matches)} matches')
        return sorted(matches, reverse=self.larger_score_is_better)[:k]

    def index_search(self, query_enc, k=10, min_score=None, max_score=None):
        scores, match_idxs = self.index.search(query_enc, k=k)
        scores = scores.squeeze(0)
        match_idxs = match_idxs.squeeze(0)
        matches = [
            ObjectWithScore(
                {'key_idx': match_idx, 'key': self.keys[match_idx]},
                score)
            for score, match_idx in zip(scores, match_idxs)
            ]
        if min_score is not None:
            matches = [
                match for match in matches
                if match.score >= min_score]
        if max_score is not None:
            matches = [
                match for match in matches
                if match.score <= max_score]
        return matches

    def parse_query(self, query_str):
        is_complex = False

        def parse_conjunct(q):
            nonlocal is_complex
            pos, *neg = q.split('MINUS')
            if neg:
                is_complex = True
            return dict(pos=pos, neg=neg)

        def parse_disjunct(disjunct):
            nonlocal is_complex
            conjuncts = disjunct.split('AND')
            if len(conjuncts) > 1:
                is_complex = True
            return {'conjuncts': list(map(parse_conjunct, conjuncts))}

        disjuncts = query_str.split('OR')
        if len(disjuncts) > 1:
            is_complex = True
        query = {'disjuncts': list(map(parse_disjunct, disjuncts))}
        print(f'is_complex: {is_complex} query: {query}')
        if is_complex:
            return query
        return query_str
