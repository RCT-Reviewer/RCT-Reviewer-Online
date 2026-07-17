"""
Tests for vectorizer and utility classes in the ML modules.
These do NOT require model weight files — only the code logic.
"""

import numpy as np
import pytest
from scipy.sparse import issparse

from rct_reviewer.ml.pico_robot import (
    PICO_vectorizer,
    Drugbank,
    PICORobot,
)
from rct_reviewer.ml.bias_robot import (
    InteractionHashingVectorizer,
    ModularVectorizer,
)


# ======================================================================
# PICO_vectorizer
# ======================================================================


class TestPICOVectorizer:
    @pytest.fixture
    def vectorizer(self):
        return PICO_vectorizer()

    def test_token_contains_number(self, vectorizer):
        assert vectorizer.token_contains_number("abc123") is True
        assert vectorizer.token_contains_number("abc") is False
        assert vectorizer.token_contains_number("42") is True
        assert vectorizer.token_contains_number("") is False

    def test_extract_structural_features_returns_12_features(self, vectorizer):
        fv = vectorizer.extract_structural_features("This is a test sentence.")
        assert len(fv) == 12

    def test_extract_structural_features_single_line(self, vectorizer):
        fv = vectorizer.extract_structural_features("A simple sentence.")
 
        assert fv[0] == 1

    def test_extract_structural_features_many_newlines(self, vectorizer):
        text = "\n".join(["line"] * 50)
        fv = vectorizer.extract_structural_features(text)

        assert fv[2] == 1 or fv[3] == 1

    def test_extract_numeric_features_shape(self, vectorizer):
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        X = vectorizer.extract_numeric_features(sentences)
        assert X.shape[0] == 3
        assert X.shape[1] == 12

    def test_transform_returns_sparse_matrix(self, vectorizer):
        sentences = ["This is sentence one.", "This is sentence two."]
        X = vectorizer.transform(sentences)
        assert issparse(X)
        assert X.shape[0] == 2

    def test_transform_with_extra_features(self, vectorizer):
        sentences = ["Sentence one.", "Sentence two."]
        extra = [{"DocumentPositionQuintile0": 1}, {"DocumentPositionQuintile1": 1}]
        X = vectorizer.transform(sentences, extra_features=extra)
        assert issparse(X)
        assert X.shape[0] == 2

    def test_dict_vectorizer_feature_names(self, vectorizer):
        expected = [
            'DocumentPositionQuintile0',
            'DocumentPositionQuintile1',
            'DocumentPositionQuintile2',
            'DocumentPositionQuintile3',
            'DocumentPositionQuintile4',
            'DocumentPositionQuintile5',
            'DocumentPositionQuintile6',
        ]
        assert vectorizer.dict_vectorizer.feature_names_ == expected


# ======================================================================
# Drugbank
# ======================================================================


class TestDrugbank:
    def test_no_data_returns_zero(self):
        """When drugbank file is missing, contains_drug returns 0."""
        db = Drugbank()

        if not db.data:
            assert db.contains_drug("aspirin") == 0

    def test_contains_drug_with_empty_data(self):
        db = Drugbank()
        db.data = {}
        assert db.contains_drug("anything") == 0

    def test_contains_drug_with_mock_data(self):
        db = Drugbank()
        db.data = {"aspirin": True, "ibuprofen": True}
        assert db.contains_drug("take aspirin daily") == 1
        assert db.contains_drug("take paracetamol") == 0


# ======================================================================
# PICORobot._get_positional_features (static method)
# ======================================================================


class TestPICORobotPositionalFeatures:
    def test_small_number_of_sentences(self):
        sents = ["sent1", "sent2", "sent3"]
        features = PICORobot._get_positional_features(sents)
        assert len(features) == 3
        for f in features:
            assert isinstance(f, dict)

    def test_large_number_of_sentences(self):
        sents = [f"sentence {i}" for i in range(100)]
        features = PICORobot._get_positional_features(sents)
        assert len(features) == 100

        assert "DocumentPositionQuintile0" in features[0]

    def test_single_sentence(self):
        features = PICORobot._get_positional_features(["only one"])
        assert len(features) == 1

    def test_empty_sentences(self):
        features = PICORobot._get_positional_features([])
        assert features == []

    def test_quintile_capped_at_6(self):
        sents = [f"s{i}" for i in range(200)]
        features = PICORobot._get_positional_features(sents)
        for f in features:
            for key in f:
                quintile_num = int(key.replace("DocumentPositionQuintile", ""))
                assert quintile_num <= 6


# ======================================================================
# InteractionHashingVectorizer
# ======================================================================


class TestInteractionHashingVectorizer:
    def test_transform_plain_text(self):
        vec = InteractionHashingVectorizer(ngram_range=(1, 2))
        X = vec.transform(["hello world", "foo bar"])
        assert issparse(X)
        assert X.shape[0] == 2

    def test_transform_with_interaction(self):
        vec = InteractionHashingVectorizer(ngram_range=(1, 2))

        X = vec.transform([("hello world", "INT-"), ("foo bar", "")])
        assert issparse(X)
        assert X.shape[0] == 2


# ======================================================================
# ModularVectorizer
# ======================================================================


class TestModularVectorizer:
    def test_builder_add_docs(self):
        mv = ModularVectorizer(ngram_range=(1, 2))
        mv.builder_add_docs(["hello world", "foo bar"])
        assert mv.X is not None
        assert issparse(mv.X)

    def test_builder_clear(self):
        mv = ModularVectorizer(ngram_range=(1, 2))
        mv.builder_add_docs(["hello world"])
        mv.builder_clear()
        assert mv.X is None

    def test_builder_transform(self):
        mv = ModularVectorizer(ngram_range=(1, 2))
        mv.builder_add_docs(["hello world", "foo bar"])
        X = mv.builder_transform()
        assert issparse(X)
        assert X.shape[0] == 2

    def test_builder_add_docs_with_weighting(self):
        mv = ModularVectorizer(ngram_range=(1, 2))
        mv.builder_add_docs(["hello world"])
        mv.builder_add_docs(["foo bar"], weighting=2)
        X = mv.builder_transform()
        assert issparse(X)

    def test_builder_add_interaction_docs(self):
        mv = ModularVectorizer(ngram_range=(1, 2))
        mv.builder_add_docs(["hello world"])
        mv.builder_add_docs([("hello world", "INT-domain")])
        X = mv.builder_transform()
        assert issparse(X)

    def test_multiple_add_then_transform(self):
        mv = ModularVectorizer(ngram_range=(1, 1))
        mv.builder_add_docs(["sentence one"])
        mv.builder_add_docs(["sentence two"])
        X = mv.builder_transform()
        assert issparse(X)
      
        assert X.shape[0] == 1