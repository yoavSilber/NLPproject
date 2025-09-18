import logging

import os
import pickle
from dataclasses import dataclass
from logging import Logger

from enum import Enum

import numpy as np
import pandas as pd
from itertools import count

from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from TeamimTree import TeamimTree
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from transformers import AutoTokenizer, AutoModel
import torch


class Chumash(Enum):
    Genesis     = 'gn'
    Exodus      = 'ex'
    Leviticus   = 'lv'
    Numbers     = 'nu'
    Deuteronomy = 'dt'


CHUMASH_LIST = [c.name for c in Chumash]
DH_SOURCES = ('J', 'E', 'P', 'R', 'Dtr1', 'Dtr2', 'Dtn', 'Other')
DH_SOURCES_FINAL = ('J', 'E', 'P', 'R', 'D')


@dataclass
class PasukData:
    source_book: str | None = None
    pasuk_text: str | None = None
    original_text: str = ''
    constituency_tree: dict | None = None
    dependencies_tree: dict | None = None
    teamim_tree: TeamimTree | None = None
    dh_sources_list: list[tuple[str, int, int]] = None


class PasukClassifier:
    def __init__(self, dataset: dict[str, PasukData], logger: Logger = None):
        self.logger: Logger = logger
        self.logger.setLevel(level=logging.INFO)
        self.pasuk_data = dataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dicta_tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint')
        self.dicta_model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint',
                                                     trust_remote_code=True, do_prefix=False).to(self.device)
        self.dicta_model.eval()
        self.dict_vec_cons = DictVectorizer(sparse=False)

        self.dict_vec_dep = DictVectorizer()

        self.dict_vec_teamim = DictVectorizer()

        self.tfidf_vect_words = TfidfVectorizer(lowercase=False)
        self.tfidf_vect_lex = TfidfVectorizer(lowercase=False)
        self.label_encoder = LabelEncoder()

        self.debug_pickle = 'classifier_pickle.pkl'


    @staticmethod
    def extract_features_from_dependency_tree(tree: dict):
        extracted_features = dict()
        for (idx, token) in enumerate(tree['tokens']):
            for token_key, token_value in token.items():
                if isinstance(token_value, dict):
                    for key, value in token_value.items():
                        if key not in ['word', 'token']:
                            if isinstance(value, dict):
                                for f, f_val in value.items():
                                    extracted_features[f'{f.lower()}_{idx}'] = f_val
                            else:
                                extracted_features[f'{key}_{idx}'] = value
                else:
                    extracted_features[f'{token_key}_{idx}'] = token_value

        lex = ' '.join([value for key, value in extracted_features.items()
                        if key.startswith('lex_') and value != '[BLANK]'])
        for key in [key for key in extracted_features.keys() if key.startswith(('lex', 'dep_head'))]:
            del extracted_features[key]
        return lex, extracted_features

    @staticmethod
    def extract_features_from_teamim_tree(tree: TeamimTree, counter, offset=0):
        if tree is None:
            return dict()

        idx = next(counter)
        tree.idx = idx
        extracted_features = dict()

        curr_offset = offset
        for child in tree.children:
            child.offset = curr_offset + len(child.pasuk) + 1
            extracted_features.update(PasukClassifier.extract_features_from_teamim_tree(child, counter, offset))
        if tree.rank > 0:
            extracted_features.update({
                f'pos_{idx}': offset,
                f'rank_{idx}': tree.rank,
                f'taam_{idx}': tree.taam})
        else:
            extracted_features.update({'tree_depth': tree.height})


        return extracted_features

    def extract_features_and_labels(self, pasuks: list[PasukData], fit: bool = True, label_dh: bool = False):
        extracted_dep = []
        extracted_teamim = []
        extracted_words = []
        tokenizer_dep = []
        extracted_lex = []

        labels = []
        for i, pasuk_data in enumerate(pasuks):
            lex, synt = self.extract_features_from_dependency_tree(pasuk_data.dependencies_tree)
            extracted_dep.append(synt)
            extracted_lex.append(lex)
            extracted_teamim.append(self.extract_features_from_teamim_tree(pasuk_data.teamim_tree, count()))
            extracted_words.append(pasuk_data.pasuk_text)

            if label_dh:
                source = max(pasuk_data.dh_sources_list, key=lambda s: s[2] - s[1])[0]
                if source[0] == 'D':
                    label = 'D'
                else:
                    label = source
            else:
                label = pasuk_data.source_book
            inputs = self.dicta_tokenizer(pasuk_data.pasuk_text, padding='longest', truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # calculate the logits
            with torch.no_grad():
                hidden = self.dicta_model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            tokenizer_dep.append(hidden[1][:, 0, :].to('cpu').numpy()[0])
            labels.append(label)


        y_final = self.label_encoder.fit_transform(labels)

        if fit:
            x_dep = self.dict_vec_dep.fit_transform(extracted_dep, y_final)
            x_teamim = self.dict_vec_teamim.fit_transform(extracted_teamim, y_final)
            x_words = self.tfidf_vect_words.fit_transform(extracted_words, y_final)
            x_lex = self.tfidf_vect_lex.fit_transform(extracted_lex, y_final)
        else:
            x_dep = self.dict_vec_dep.transform(extracted_dep)
            x_teamim = self.dict_vec_teamim.transform(extracted_teamim)
            x_words = self.tfidf_vect_words.transform(extracted_words)
            x_lex = self.tfidf_vect_lex.fit_transform(extracted_lex)


        x_union = VarianceThreshold().fit_transform(hstack([x_dep, x_words, x_teamim, x_lex, tokenizer_dep]))
        x_final = x_union
        print(x_final.shape)
        return x_final, y_final

    def run_classification(self, classify_dh: bool = False, overwrite_pickle: bool = False):

        if not overwrite_pickle and os.path.exists(self.debug_pickle):
            self.logger.info('Debug pickle found, loading data.')
            with open(self.debug_pickle, "rb") as debug_pkl:
                X, y = pickle.load(debug_pkl)
        else:
            X, y = self.extract_features_and_labels(list(self.pasuk_data.values()), label_dh=classify_dh)
            inputs_targets = (X, y)
            self.logger.info('Finished Extracting features. Saving pickle...')
            with open(self.debug_pickle, "wb") as debug_pkl:
                pickle.dump(inputs_targets, debug_pkl)

        X = SelectKBest(f_classif, k=25000).fit_transform(X, y)
        X =  MinMaxScaler().fit_transform(np.asarray(X.todense()))
        k_fold_score = KFold(n_splits=10, random_state=42, shuffle=True)

        model = ComplementNB()

        self.logger.info('Finished fitting')

        scoring = {'accuracy': make_scorer(accuracy_score),
                   'precision': make_scorer(precision_score, zero_division=0.0, average='weighted'),
                   'recall': make_scorer(recall_score, average='weighted'),
                   'f1_macro': make_scorer(f1_score, average='weighted')}

        score = cross_validate(estimator=model, X=X, y=y, cv=k_fold_score, n_jobs=1, scoring=scoring, verbose=100)
        self.logger.info('Finished evaluating')
        score_dataframe = pd.DataFrame(score)
        score_report_pickle = 'score.pkl'
        with open(score_report_pickle, 'wb') as score_report_pickle:
            score_dataframe.to_pickle(score_report_pickle)

        score_dataframe.loc['mean'] = score_dataframe.mean()
        print(score_dataframe.to_string())






