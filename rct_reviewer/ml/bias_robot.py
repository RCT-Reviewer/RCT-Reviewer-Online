import numpy as np
import scipy as sp
from scipy.sparse import hstack
from rct_reviewer import get_data_path
from rct_reviewer.ml.classifier import MiniClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
import logging

log = logging.getLogger(__name__)


class InteractionHashingVectorizer(HashingVectorizer):
    """HashingVectorizer with interaction term prefixes."""
    
    def __init__(self, **kwargs):
       
        kwargs['norm'] = None
        kwargs['binary'] = True
        kwargs['alternate_sign'] = False 
        super().__init__(**kwargs)

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        tokenize = self.build_tokenizer()
        return lambda doc_i: self._word_ngrams(
            tokenize(preprocess(self.decode(doc_i[0]))), 
            None, 
            doc_i[1]
        )

    def _word_ngrams(self, tokens, stop_words=None, i_term=None):
        tokens = super()._word_ngrams(tokens, stop_words)
        if i_term:
            return [i_term + token for token in tokens]
        return tokens

    def transform(self, X_si):
        """Transform with interaction terms."""
        analyzer = self.build_analyzer()
        
        def process(doc):
            if isinstance(doc, tuple):
                return (doc[0], doc[1]) if doc[1] else ("", "")
            return (doc, "")
        
        X = self._get_hasher().transform(
            analyzer(process(doc)) for doc in X_si
        )
        X.data.fill(1)
        return X


class ModularVectorizer:
    """Memory-efficient vectorizer for interaction features."""
    
    def __init__(self, **kwargs):
        self.vec = InteractionHashingVectorizer(**kwargs)
        self.X = None

    def builder_clear(self):
        self.X = None

    def _combine_matrices(self, X_part, weighting=1):
        X_part.data.fill(weighting)
        if self.X is None:
            self.X = X_part
        else:
            self.X = self.X + X_part

    def builder_add_docs(self, X_si, weighting=1):
        X_part = self.vec.transform(X_si)
        self._combine_matrices(X_part, weighting=weighting)

    def builder_transform(self):
        return self.X


class BiasRobot:
    def __init__(self, top_k=3):
        log.info("Loading Bias models...")
        self.sent_clf = MiniClassifier('bias/bias_sent_level.npz')
        self.doc_clf = MiniClassifier('bias/bias_doc_level.npz')
        
     
        self.vec = ModularVectorizer(
            ngram_range=(1, 2), 
            n_features=2**26
        )
        
        self.bias_domains = [
            'Random sequence generation', 
            'Allocation concealment',
            'Blinding of participants and personnel', 
            'Blinding of outcome assessment',
            'Incomplete outcome data', 
            'Selective reporting'
        ]
        self.top_k = top_k

    def annotate(self, sentences, full_text):
        """Annotate with Risk of Bias assessment."""
        if not sentences:
            return []
        
        results = []
        sent_texts = [s['text'] for s in sentences]
        
        for domain in self.bias_domains:
            try:
          
                doc_domains = [domain] * len(sent_texts)
                doc_X_i = list(zip(sent_texts, doc_domains))
                
                self.vec.builder_clear()
                self.vec.builder_add_docs(sent_texts)
                self.vec.builder_add_docs(doc_X_i)
                
                X_sents = self.vec.builder_transform()
                

                sent_scores = self.sent_clf.decision_function(X_sents)
                
            
                high_indices = np.argsort(sent_scores)[-self.top_k:][::-1]
                high_sents = [sent_texts[i] for i in high_indices]
                
                
                combined_text = " ".join(high_sents)
                sent_domain_interaction = "-s-" + domain
                
                self.vec.builder_clear()
                self.vec.builder_add_docs([full_text])
                self.vec.builder_add_docs([(full_text, domain)])
                self.vec.builder_add_docs([(combined_text, sent_domain_interaction)])
                
                X_doc = self.vec.builder_transform()
                bias_pred = self.doc_clf.predict(X_doc)
                
                results.append({
                    "domain": domain,
                    "judgement": ["high/unclear", "low"][bias_pred[0]],
                    "text": high_sents,
                    "indices": high_indices.tolist()
                })
                
            except Exception as e:
                log.error(f"Error processing domain {domain}: {e}")
                results.append({
                    "domain": domain,
                    "judgement": "high/unclear",
                    "text": [],
                    "indices": []
                })
        
        return results