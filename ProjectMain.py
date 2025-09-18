import os
import pickle
import logging
import re
import sys
from logging import Logger

from xml.etree import ElementTree

import pandas as pd

import json

from transformers import AutoTokenizer, AutoModel
import torch

from TeamimTree import TeamimTree
from PasukClassifier import PasukClassifier, PasukData, Chumash

from collections import OrderedDict

from ProjectStatistics import ProjectStatistics

deuteronomy_xml_file = "PSOOKIM/Deuteronomy.xml"
exodus_xml_file = "PSOOKIM/Exodus.xml"
genesis_xml_file = "PSOOKIM/Genesis.xml"
leviticus_xml_file = "PSOOKIM/Leviticus.xml"
numbers_xml_file = "PSOOKIM/Numbers.xml"

TANACH_DIR = 'TANACH.US'
SHEBANQ_DIR = 'SHEBANQ'
DH_MARKINGS_DIR = 'DH_Markings'
PATH_TO_TEAMIM = "Teamim.xlsx"
NUM_OF_PASUKS = 5847

PASUKS_TO_ANALYZE = ('gn22:1', 'gn24:60', 'gn28:8', 'gn30:7', 'gn34:7',
                     'ex2:23', 'ex7:28', 'ex12:29', 'ex14:30', 'ex15:17',
                     'lv16:30',
                     'dt3:11', 'dt5:19', 'dt9:28', 'dt19:4')

DEBUG_MODE = False  # Use the created pickle file of pasuk_db in order to save time each run

# Namespace (required for parsing TEI XML
NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0',
         'xml': 'http://www.w3.org/XML/1998/namespace',
         'xsi': 'http://www.w3.org/2001/XMLSchema-instance'}


def parse_teamim_to_list(teamim: str):
    taam_list = []
    for taam in teamim.strip().split('/'):
        if taam == '':
            continue
        elif taam == '-':
            taam_list.append(100)
        else:
            taam_list.append(int(taam[:2]))
    return taam_list


class ProjectNLP:

    def __init__(self, logger: Logger = None):
        self.logger: Logger = logger

        self.pasuk_db: dict[str, PasukData] = OrderedDict()

        self.debug_pickle = 'debug_pickle.pkl'
        self.debug_mode = not (os.path.exists(self.debug_pickle) and DEBUG_MODE)
        if self.debug_mode:
            self.logger.info('No debug pickle found, rebuilding DataBase...')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f'Using device: {self.device}')
            self.tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint')

            self.model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint',
                                                   trust_remote_code=True, do_prefix=False)
            self.model.to(self.device)

            self.model.eval()

    def parse_constituency_data_from_chumash(self, chumash: Chumash) -> None:
        self.logger.info(f'Parsing constituency data from chumash {chumash.name}...')
        xml_path = os.path.join(SHEBANQ_DIR, chumash.name + '.xml')
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        # Namespace (required for parsing TEI XML
        syntactic_data = dict()
        for pasuk in root.findall(".//tei:s/[tei:syntacticInfo]", NSMAP):

            pasuk_xml_id = pasuk.attrib.get("{" + NSMAP['xml'] + "}id")
            pasuk_id_for_data = chumash.value + ':'.join(pasuk_xml_id.rsplit('.')[-2:])

            pasuk_data = {'clauses': []}

            for clause in pasuk.findall("tei:syntacticInfo/tei:sentence/tei:clause", NSMAP):
                clause_id = clause.attrib.get("id")
                clause_type = clause.attrib.get("type")

                clause_data = {"type": clause_type, "phrases": []}

                for phrase in clause.findall("tei:phrase", NSMAP):
                    phrase_id = phrase.attrib.get("id")
                    phrase_type = phrase.attrib.get("type")
                    phrase_function = phrase.attrib.get("function")
                    phrase_word = [w for w in pasuk.findall("tei:w", NSMAP)
                                    if w.find(f"*[@phraseId='{phrase_id}']", NSMAP) is not None][0]

                    clause_data["phrases"].append({
                        # "phrase_id": phrase_id,
                        "type": phrase_type,
                        "lemma": phrase_word.attrib.get('lemma'),
                        "root": phrase_word.attrib.get('root'),
                        "word": phrase_word.text,
                        "function": phrase_function
                    })

                pasuk_data["clauses"].append(clause_data)

            self.pasuk_db[pasuk_id_for_data] = PasukData(constituency_tree=pasuk_data,
                                                         source_book=chumash.name, dh_sources_list=[])

    def create_dependency_trees(self, chumash: Chumash):
        self.logger.info(f'Building dependency trees from chumash {chumash.name}...')
        xml_path = os.path.join(TANACH_DIR, chumash.name+'.xml')
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        for chapter in root.findall(".//tanach/book/c", NSMAP):
            chapter_id_in_tannach = chumash.value + chapter.get('n')
            for pasuk in chapter.findall("v"):
                pasuk_id = chapter_id_in_tannach + ':' + pasuk.get('n')

                original_pasuk_text = ' '.join([word.text for word in pasuk.findall("w")])
                self.pasuk_db[pasuk_id].original_text = original_pasuk_text
                self.pasuk_db[pasuk_id].pasuk_text = re.sub(r"[\u0591-\u05C7]", '', original_pasuk_text)
                pasuk_prediction = self.model.predict([self.pasuk_db[pasuk_id].pasuk_text],
                                                      self.tokenizer, output_style='json')[0]
                # Process the output to remove `seg`, `offsets` and simplify `dep_func`

                pasuk_prediction["tokens"] = [
                    {
                        key: (
                            value.split(":")[0] if key == "dep_func" else value
                        ) if key == "dep_func" else value
                        for key, value in token.items()
                        if key not in {"offsets"}
                        # add {"prefixes", "offsets"} so it will delete prefixes definition if meny says so
                    }
                    for token in pasuk_prediction["tokens"]
                ]
                if "ner_entities" in pasuk_prediction:
                    pasuk_prediction["ner_entities"] = [
                        {key: entity[key] for key in ("phrase", "label")}
                        for entity in pasuk_prediction["ner_entities"]
                    ]

                self.pasuk_db[pasuk_id].dependencies_tree = pasuk_prediction

    def create_teamim_trees(self):
        self.logger.info('Building teamim trees...')
        teamim_data = pd.read_excel(PATH_TO_TEAMIM, sheet_name='Sheet1', nrows=NUM_OF_PASUKS, header=None,
                                         usecols='B,C', names=['Pasuk', 'Teamim'], index_col='Pasuk')
        teamim_data = teamim_data.map(lambda x: str(x).strip())
        teamim_dict = teamim_data['Teamim'].map(parse_teamim_to_list).to_dict()

        for pasuk_id, teamim in teamim_dict.items():
            broken_pasuk = self.pasuk_db[pasuk_id].pasuk_text.split(' ')
            self.pasuk_db[pasuk_id].teamim_tree = TeamimTree(broken_pasuk=broken_pasuk, teamim=teamim)

    @staticmethod
    def parse_range_dh(range_dh: str) -> (dict[str, int], dict[str, int]):
        range_split = range_dh.replace(' ', '').split('-')
        if len(range_split) == 1:
            range_split += range_split
        split_dict = [dict(), dict()]

        for i in range(2):
            split_dict[i]['Chapter'] = 0
            split_dict[i]['Word'] = 0
            split = range_split[i]
            if ':' in split:
                split = range_split[i].split(':')
                split_dict[i]['Chapter'] = int(split[0])
                if '.' in split[1]:
                    split = [split[0]] + split[1].split('.')
                    split_dict[i]['Word'] = int(split[2])-1
                split_dict[i]['Pasuk'] = int(split[1])
            else:
                split_dict[i]['Pasuk'] = int(split)
                if i == 1:
                    split_dict[i]['Chapter'] = split_dict[0]['Chapter']

        return split_dict[0], split_dict[1]

    def get_next_range_from_chumash_dh(self, chumash: Chumash):
        xml_path = os.path.join(DH_MARKINGS_DIR, chumash.name+'.xml')
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        for word_range in root.findall(".//excerpts/excerpt", NSMAP):
            start_range, end_range = self.parse_range_dh(word_range.find('range', NSMAP).text)
            source_dh = word_range.find('source', NSMAP).text
            yield start_range, end_range, source_dh

    @staticmethod
    def split_id_to_multi_index(p: str):
        split = p[2:].split(':')

        return p[:2], int(split[0]), int(split[1])

    def map_dh_source(self):
        self.logger.info('Mapping Pasuks to their DH source...')
        index = pd.MultiIndex.from_tuples([self.split_id_to_multi_index(p) for p in self.pasuk_db.keys()])
        pasuk_df = pd.Series(data=list(self.pasuk_db.keys()), index=index).sort_index()
        for chumash in Chumash:
            self.logger.info(f'Mapping Pasuks to their DH for chumash {chumash.name}')
            for start_dict, end_dict, source in self.get_next_range_from_chumash_dh(chumash):
                start_range = (chumash.value, start_dict['Chapter'], start_dict['Pasuk'])
                end_range = (chumash.value, end_dict['Chapter'], end_dict['Pasuk'])
                pasuk_range_list = pasuk_df.loc[start_range:end_range].to_list()
                if len(pasuk_range_list) == 1:
                    self.pasuk_db[pasuk_range_list[0]].dh_sources_list.append((source,
                                                                              start_dict['Word'],
                                                                              end_dict['Word']))
                else:
                    if start_dict['Word'] > 0:
                        pasuk_len = len(self.pasuk_db[pasuk_range_list[0]].pasuk_text.split(' '))
                        self.pasuk_db[pasuk_range_list[0]].dh_sources_list.append((source,
                                                                                   start_dict['Word'],
                                                                                   pasuk_len))
                        pasuk_range_list = pasuk_range_list[1:]

                    if end_dict['Word'] > 0:
                        self.pasuk_db[pasuk_range_list[-1]].dh_sources_list.append((source,
                                                                                   0,
                                                                                   end_dict['Word']))
                        pasuk_range_list = pasuk_range_list[:-1]

                    for pasuk_id in pasuk_range_list:
                        pasuk_len = len(self.pasuk_db[pasuk_id].pasuk_text.split(' '))
                        self.pasuk_db[pasuk_id].dh_sources_list.append((source, 0,  pasuk_len))

    def build_pasuk_db(self):
        if not self.debug_mode:
            self.logger.info('Debug pickle found, loading data.')
            with open(self.debug_pickle, "rb") as debug_pkl:
                self.pasuk_db = pickle.load(debug_pkl)
        else:
            for chumash in Chumash:
                self.parse_constituency_data_from_chumash(chumash)
                self.create_dependency_trees(chumash)

            self.create_teamim_trees()
            self.map_dh_source()

            self.logger.info('Finished Building DB. Saving pickle...')
            with open(self.debug_pickle, "wb") as debug_pkl:
                pickle.dump(self.pasuk_db, debug_pkl)

    def analyze_pasuks(self, pasuks: list[str] = PASUKS_TO_ANALYZE):
        self.logger.info('Analyzing Pasuks')
        out_json = dict()

        for pasuk_id in pasuks:
            out_json[pasuk_id] = dict()
            pasuk_data = self.pasuk_db[pasuk_id]
            for field, field_data in vars(pasuk_data).items():
                if field == 'teamim_tree':
                    out_json[pasuk_id][field] = field_data.to_json()
                    out_json[pasuk_id]['teamim_clauses'] = field_data.get_clauses()
                else:
                    out_json[pasuk_id][field] = field_data

        output_path = 'analyzed_pasuks.json'
        self.logger.info(f'Saving analyzed pasuks to {output_path}')
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2, ensure_ascii=False)


    def main(self):
        self.build_pasuk_db()  # Parse and create the DB of the pasuks

        self.analyze_pasuks()  # Analyze the specific pasuks`in PASUKS_TO_ANALYZE and output to json

        classifier = PasukClassifier(self.pasuk_db, self.logger)  # The pasuks Classifier

        self.logger.info('Classifying Pasuks By Chumash...')
        classifier.run_classification(classify_dh=False, overwrite_pickle=True)  # Classifying to Chumash
        self.logger.info('Classifying Pasuks By DH...')
        classifier.run_classification(classify_dh=True, overwrite_pickle=True)  # Classifying to DH sources

        # Creating the DataFrame with all data needed for statistical analysis
        statistics = ProjectStatistics(self.pasuk_db, self.logger)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        summary_by_book, summary_by_dh = statistics.extract_stats()

        print("\n=== Summary by Book ===")
        print(summary_by_book.to_string())

        print("\n=== Summary by DH Source ===")
        print(summary_by_dh.to_string())


def start_logger() -> Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger


def main():
    logger = start_logger()
    logger.info("Program Starting")
    project = ProjectNLP(logger)
    project.main()


if __name__ == '__main__':
    main()
