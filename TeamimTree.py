from __future__ import annotations

import pandas as pd

TEAMIM_RANK = {
    'Root': 0,
    'Emperor': 1,
    'King': 2,
    'Minister': 3,
    'Official': 4,
    'Servant': 5,
    'Irrelevant': 6
}
TEAMIM_RANK.update({rank: taam for (taam, rank) in TEAMIM_RANK.items()})

PATH_TO_TEAMIM = "Teamim.xlsx"
TEAMIM_DF = pd.read_excel(PATH_TO_TEAMIM, sheet_name='Sheet2', usecols='A:C')
TEAMIM_DF['ID'].map(int)

TEAMIM_DICT = TEAMIM_DF.set_index('ID').to_dict(orient='index')
for i in TEAMIM_DICT.keys():
    TEAMIM_DICT[i]['Rank'] = TEAMIM_RANK[TEAMIM_DICT[i]['Rank']]

TEAMIM_DICT[100] = {'Rank': 6}


class TeamimTree(object):
    def __init__(self, parent: TeamimTree = None, word_text: str = None,
                 taam: int = None, broken_pasuk: list[str] = None, teamim: list[int] = None):
        self.parent: TeamimTree = parent
        self.taam: int = taam
        self.pasuk: str = " ".join(broken_pasuk)
        self.word_text: str = word_text if taam is not None else self.pasuk
        self.rank: int = TEAMIM_DICT[self.taam]['Rank'] if taam is not None else 0
        self.taam_name: str = TEAMIM_DICT[self.taam]['Name'] if taam is not None else 'Root'
        self.teamim: list[int] = teamim

        self.height: int = 0

        self.children: list[TeamimTree] = []
        self.children_rank = min(map(lambda x: TEAMIM_DICT[x]['Rank'], self.teamim)) if len(self.teamim) > 0 else 5
        if self.children_rank < 6 and len(self.teamim) > 1:
            self.find_children(broken_pasuk)

        self.idx = -1 # For classifier

    def find_children(self, broken_pasuk: list[str]):
        start = 0
        for (child_idx, child_taam) in enumerate(self.teamim):
            if TEAMIM_DICT[child_taam]['Rank'] == self.children_rank:
                end_idx = child_idx
                while end_idx > 0 and self.teamim[end_idx - 1] == 100:
                    end_idx -= 1
                child_passuk = broken_pasuk[start: end_idx]
                pasuk_word = ' '.join(broken_pasuk[end_idx:child_idx + 1])
                child_teamim = self.teamim[start:end_idx]

                child = TeamimTree(parent=self, word_text=pasuk_word, taam=child_taam,
                                   broken_pasuk=child_passuk, teamim=child_teamim)

                if child.height >= self.height:
                    self.height = child.height + 1
                self.children.append(child)

                start = child_idx + 1

    def get_tree_height(self):
        return self.height

    def get_domain(self):
        return (' '.join([self.pasuk, self.word_text])).strip()

    def get_clauses(self):
        if self.rank > 0:
            return []

        clauses = []
        for emperor in self.children:
            emperor_clauses = []
            for king in emperor.children:
                emperor_clauses.append(king.get_domain())

            total_len_kings = len(' '.join(emperor_clauses))
            end_range = (len(emperor.pasuk) - total_len_kings)
            if end_range > 0:
                emperor_reminder = ' '.join([emperor.pasuk[-end_range:], emperor.word_text])
            else:
                emperor_reminder = emperor.word_text
            emperor_clauses.append(emperor_reminder.strip())

            clauses.extend(emperor_clauses)

        map(str.strip, clauses)
        return clauses

    def get_tree_graph(self, with_data: bool = True):
        if self.taam is not None:
            taam_rank = TEAMIM_RANK[self.rank]
            if with_data:
                tree_str = f"Taam={self.taam_name}, Rank={taam_rank}, Text={self.word_text}\n"
            else:
                tree_str = 'Node\n'

        else:
            tree_str = f"Pasuk: {self.pasuk}\n" if with_data else "Root\n"
        for child in self.children:
            pad_num = self.rank + 1
            pad = '\t\t' * pad_num
            tree_str += pad + '|\n' + pad + '--' + child.get_tree_graph(with_data=with_data)

        return tree_str

    def __str__(self):
        return self.get_tree_graph(with_data=True)

    def __repr__(self):
        return str(self)

    def to_json(self):
        json_data = {'Name': self.taam_name,
                     'Rank': TEAMIM_RANK[self.rank],
                     'Text': self.word_text,
                     'Children': [child.to_json() for child in self.children]}
        return json_data
