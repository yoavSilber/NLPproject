import os
from logging import Logger

import pandas as pd
import re
from collections import Counter, defaultdict

from TeamimTree import TeamimTree
from PasukClassifier import PasukData


class ProjectStatistics:
    def __init__(self, pasuk_data: dict[str, PasukData], logger: Logger = None):
        self.logger: Logger = logger
        self.pasuk_data: dict[str, PasukData] = pasuk_data

    @staticmethod
    def parse_id(pasuk_id):
        match = re.match(r"([a-z]{2})(\d+):(\d+)", pasuk_id)
        if not match:
            return ("Unknown", 0, 0)
        return match.groups()[1:]

    @staticmethod
    def dependency_tree_depth(tree):
        tokens = tree.get("tokens", [])
        if not tokens:
            return 0

        children_map = {i: [] for i in range(len(tokens))}
        root_index = None

        for i, token in enumerate(tokens):
            syntax = token.get("syntax", {})
            head_idx = syntax.get("dep_head_idx")
            if head_idx == -1:
                root_index = i
            elif head_idx is not None and 0 <= head_idx < len(tokens):
                children_map[head_idx].append(i)

        if root_index is None:
            return 1

        def get_depth(index):
            if not children_map[index]:
                return 1
            return 1 + max(get_depth(child) for child in children_map[index])

        return get_depth(root_index)

    #how many times the Rank changes between parent -> child in the tree.
    @staticmethod
    def get_taam_movement(tree: TeamimTree):
        moves = 0
        parent_rank = tree.rank
        for child in tree.children:
            child_rank = child.rank
            if parent_rank != child_rank:
                moves += 1
            moves += ProjectStatistics.get_taam_movement(child)
        return moves

    @staticmethod
    def format_teamim_tree(tree: TeamimTree, depth=0):
        output = ""
        output += "  " * depth + f"{tree.rank} - {tree.taam_name}\n"
        for child in tree.children:
            output += ProjectStatistics.format_teamim_tree(child, depth + 1)
        return output

    def group_statistics(self, df: pd.DataFrame):
        # Explode DH sources list to rows (so one pasuk can appear for multiple DHs)
        exploded_dh = df.explode("dh_sources")

        # Group by Book (Chumash)
        chumash_group = df.groupby("book").agg({
            "num_words": "mean",
            "unique_words": "mean",
            "char_length": "mean",
            "num_phrases": "mean",
            "lexical_diversity": "mean",
            "phrase_type_transitions": "mean",
            "clause_count": "mean",
            "teamim_clause_count": "mean",
            "num_dh_sources": "mean",
            "taam_movement": "mean",
            "teamim_segments": "mean",
            "teamim_tree_height": "mean",
            "dependency_tree_depth": "mean"
        })

        # Group by DH Source
        dh_group = exploded_dh.groupby("dh_sources").agg({
            "num_words": "mean",
            "unique_words": "mean",
            "char_length": "mean",
            "num_phrases": "mean",
            "lexical_diversity": "mean",
            "phrase_type_transitions": "mean",
            "clause_count": "mean",
            "teamim_clause_count": "mean",
            "taam_movement": "mean",
            "teamim_segments": "mean",
            "teamim_tree_height": "mean",
            "dependency_tree_depth": "mean"
        })

        # count psukim per group
        chumash_group["pasuk_count"] = df.groupby("book").size()
        dh_group["pasuk_count"] = exploded_dh.groupby("dh_sources").size()

        top_words_by_book = self.get_top_words_per_group(df, group_col="book", top_n=10)
        top_words_by_dh = self.get_top_words_per_group(exploded_dh, group_col="dh_sources", top_n=10)

        top_words_by_book_series = pd.Series({
            group: ', '.join([f"{w} ({c})" for w, c in top_words])
            for group, top_words in top_words_by_book.items()
        })
        top_words_by_dh_series = pd.Series({
            group: ', '.join([f"{w} ({c})" for w, c in top_words])
            for group, top_words in top_words_by_dh.items()
        })

        chumash_group["top_words"] = chumash_group.index.map(top_words_by_book_series).fillna("")
        dh_group["top_words"] = dh_group.index.map(top_words_by_dh_series).fillna("")

        # For Book Summary (global average from raw df)
        overall_book = df.drop(
            columns=["word_freq_dict", "dh_sources", "teamim_tree_structure", "pasuk_id", "book", "chapter", "verse"])
        overall_book = overall_book.mean(numeric_only=True)
        overall_book["pasuk_count"] = len(df)

        # Compute top 10 words for All Books
        all_words_book = Counter()
        for freq in df["word_freq_dict"]:
            all_words_book.update(freq)
        overall_book["top_words"] = ', '.join([f"{w} ({c})" for w, c in all_words_book.most_common(10)])
        chumash_group.loc["All Books"] = overall_book

        # For DH Summary (global average from exploded_dh)
        overall_dh = exploded_dh.drop(
            columns=["word_freq_dict", "dh_sources", "teamim_tree_structure", "pasuk_id", "book", "chapter", "verse"])
        overall_dh = overall_dh.mean(numeric_only=True)
        overall_dh["pasuk_count"] = len(exploded_dh)

        # Compute top 10 words for All DH Sources
        all_words_dh = Counter()
        for freq in exploded_dh["word_freq_dict"]:
            all_words_dh.update(freq)
        overall_dh["top_words"] = ', '.join([f"{w} ({c})" for w, c in all_words_dh.most_common(10)])
        dh_group.loc["All DH Sources"] = overall_dh

        self.logger.info("Saved group summaries by book and DH source.")

        return chumash_group, dh_group

    def get_top_words_per_group(self, df: pd.DataFrame, group_col: str, top_n: int = 10):
        group_counters = defaultdict(Counter)

        for _, row in df.iterrows():
            group = row[group_col]
            if isinstance(group, list):
                for g in group:
                    group_counters[g].update(row["word_freq_dict"])
            else:
                group_counters[group].update(row["word_freq_dict"])

        top_words = {
            group: counter.most_common(top_n)
            for group, counter in group_counters.items()
        }

        return top_words

    def extract_stats(self):
        # Extract stats per pasuk
        self.logger.info("Extracting statistics...")
        rows = []
        for pasuk_id, data in self.pasuk_data.items():
            chapter, verse = self.parse_id(pasuk_id)
            book = data.source_book
            words = data.pasuk_text.split()
            unique_words = set(words)
            word_freq_dict = dict(Counter(words))
            clauses = data.constituency_tree["clauses"]
            teamim_clause_count = len(data.teamim_tree.get_clauses())
            num_phrases = sum(len(clause["phrases"]) for clause in clauses)
            num_dh_sources = len(data.dh_sources_list)
            dh_sources = [src[0] for src in data.dh_sources_list]
            if any([s.startswith('D') for s in dh_sources]):
                dh_sources.append('D')
            teamim_tree_structure = self.format_teamim_tree(data.teamim_tree)
            taam_movement = self.get_taam_movement(data.teamim_tree)
            teamim_segments = len(data.teamim_tree.get_clauses())
            teamim_tree_height = data.teamim_tree.height
            dep_depth = self.dependency_tree_depth(data.dependencies_tree)
            char_length = len(data.pasuk_text.strip(' '))
            clause_count = len(clauses)

            # Count transitions between different phrase types
            phrase_type_transitions = 0
            for clause in clauses:
                types = [phrase.get("type") for phrase in clause.get("phrases", []) if "type" in phrase]
                phrase_type_transitions += sum(1 for i in range(1, len(types)) if types[i] != types[i - 1])

            rows.append({
                "pasuk_id": pasuk_id,
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "num_words": len(words),
                "unique_words": len(unique_words),
                "lexical_diversity": len(unique_words) / len(words),
                "char_length": char_length,
                "num_phrases": num_phrases,
                "phrase_type_transitions": phrase_type_transitions,
                "clause_count": clause_count,
                "teamim_clause_count": teamim_clause_count,
                "num_dh_sources": num_dh_sources,
                "dh_sources": dh_sources,
                "taam_movement": taam_movement,
                "teamim_segments": teamim_segments,
                "teamim_tree_height": teamim_tree_height,
                "dependency_tree_depth": dep_depth,
                "teamim_tree_structure": teamim_tree_structure,
                "word_freq_dict": word_freq_dict
            })

        df = pd.DataFrame(rows)
        self.logger.info("Finished extracting statistics. Saving to CSV...")
        df.to_csv("pasuk_statistics.csv", index=False, encoding="utf-8-sig")
        self.logger.info(f"Saved to CSV : {os.path.abspath('pasuk_statistics.csv')}")

        # Compute group summaries
        summary_by_book, summary_by_dh = self.group_statistics(df)

        return summary_by_book, summary_by_dh

