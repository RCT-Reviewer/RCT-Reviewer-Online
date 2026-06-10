import numpy as np
import re
import scipy as sp
from scipy.sparse import diags, lil_matrix, csc_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from rct_reviewer import get_data_path
from rct_reviewer.ml.classifier import MiniClassifier
import logging

log = logging.getLogger(__name__)


class Drugbank:
    """Drug lookup lexicon."""
    def __init__(self):
        from rct_reviewer.config import settings
        
        if settings.use_joblib:
            path = get_data_path('drugbank/drugbank.joblib')
            loader = lambda p: __import__('joblib').load(p)
        else:
            path = get_data_path('drugbank/drugbank.pck')
            loader = lambda p: __import__('pickle').load(open(p, 'rb'))

        if path.exists():
            try:
                self.data = loader(path)
            except Exception as e:
                log.warning(f"Failed to load drugbank: {e}")
                self.data = {}
        else:
            log.warning(f"Drugbank file not found at {path}")
            self.data = {}

    def contains_drug(self, text):
        if not self.data:
            return 0
        tokens = re.split("([^A-Za-z0-9])", text.lower())
        for t in tokens:
            if t in self.data:
                return 1
        return 0


class PICO_vectorizer:
    """Exact feature extraction matching original RobotReviewer."""
    
    def __init__(self):
        self.vectorizer = HashingVectorizer(ngram_range=(1, 2))
        self.dict_vectorizer = DictVectorizer()
    
        self.dict_vectorizer.feature_names_ = [
            'DocumentPositionQuintile0',
            'DocumentPositionQuintile1',
            'DocumentPositionQuintile2',
            'DocumentPositionQuintile3',
            'DocumentPositionQuintile4',
            'DocumentPositionQuintile5',
            'DocumentPositionQuintile6']
        self.dict_vectorizer.vocabulary_ = {k: i for i, k in enumerate(self.dict_vectorizer.feature_names_)}
        self.drugbank = Drugbank()

    def token_contains_number(self, token):
        return any(char.isdigit() for char in token)

    def transform(self, sentences_text, extra_features=None, idf=None):
        """Transform sentences to feature matrix."""

        X_text = self.vectorizer.transform(sentences_text)
        
   
        X_numeric = self.extract_numeric_features(sentences_text)
        

        X_text_norm = normalize(X_text)
        X_text_norm.eliminate_zeros()
        
      
        if extra_features:
            X_extra = self.dict_vectorizer.transform(extra_features)
           
            feature_matrix = sp.sparse.hstack((X_text_norm, X_numeric, X_extra)).tocsr()
        else:
            feature_matrix = sp.sparse.hstack((X_text_norm, X_numeric)).tocsr()
            
        return feature_matrix

    def extract_numeric_features(self, sentences_text, normalize_matrix=False):
        """Extract 12 structural features per sentence."""
        n = len(sentences_text)
        m = 12
        X_numeric = lil_matrix((n, m))
        
        for i, sent in enumerate(sentences_text):
            X_numeric[i, :] = self.extract_structural_features(sent)
        
        X_numeric = X_numeric.tocsc()
        if normalize_matrix:
            X_numeric = normalize(X_numeric, axis=0)
        return X_numeric

    def extract_structural_features(self, sentence):
        """Extract 12 structural features for a single sentence."""
        fv = np.zeros(12)
        
   
        num_new_lines = sentence.count("\n")
        if num_new_lines <= 1:
            fv[0] = 1
        elif num_new_lines < 20:
            fv[1] = 1
        elif num_new_lines < 40:
            fv[2] = 1
        else:
            fv[3] = 1

   
        line_lens = [len(line) for line in sentence.split("\n") if line.strip() != ""]
        if line_lens:
            num_short_lines = sum(1 for l in line_lens if l <= 10)
            frac_short_lines = num_short_lines / len(line_lens)
            if frac_short_lines < .1:
                fv[4] = 1
            elif frac_short_lines <= .25:
                fv[5] = 1
            else:
                fv[6] = 1


        tokens = sentence.split()
        num_numbers = sum(self.token_contains_number(t) for t in tokens)
        
        if num_numbers > 0 and len(tokens) > 0:
            num_frac = num_numbers / len(tokens)
            if num_frac < .2:
                fv[7] = 1
            elif num_frac < .4:
                fv[8] = 1
            else:
                fv[9] = 1


        if tokens:
            avg_token_len = np.mean([len(t) for t in tokens])
            fv[10] = 1 if avg_token_len < 5 else 0


        fv[11] = self.drugbank.contains_drug(sentence)
        
        return fv


class PICORobot:
    def __init__(self, top_k=3, min_k=1):
        log.info("Loading PICO models...")
        self.P_clf = MiniClassifier('pico/P_model.npz')
        self.I_clf = MiniClassifier('pico/I_model.npz')
        self.O_clf = MiniClassifier('pico/O_model.npz')

        self.P_idf = self._load_idf('pico/P_idf.npz')
        self.I_idf = self._load_idf('pico/I_idf.npz')
        self.O_idf = self._load_idf('pico/O_idf.npz')

        self.vec = PICO_vectorizer()
        self.models = [self.P_clf, self.I_clf, self.O_clf]
        self.idfs = [self.P_idf, self.I_idf, self.O_idf]
        self.PICO_domains = ["Population", "Intervention", "Outcomes"]
        self.top_k = top_k
        self.min_k = min_k

    def _load_idf(self, path):
        fpath = get_data_path(path)
        if fpath.exists():
            with open(fpath, 'rb') as f:
                return diags(np.load(f, allow_pickle=True, encoding='latin1').item().todense().A1, 0)
        return None

    def annotate(self, sentences):
        """Annotate sentences with PICO extraction."""
        if not sentences:
            return []
        
        
        sent_texts = [s['text'] for s in sentences]
        

        positional_features = self._get_positional_features(sent_texts)
        
        results = []
        
        for domain, model, idf in zip(self.PICO_domains, self.models, self.idfs):
      
            X = self.vec.transform(sent_texts, extra_features=positional_features, idf=idf)
            
        
            probs = model.predict_proba(X)
            
   
            high_indices = np.argsort(probs)[:-self.top_k-1:-1]
            
            results.append({
                "domain": domain,
                "text": [sent_texts[i] for i in high_indices],
                "indices": high_indices.tolist()
            })
        
        return results

    @staticmethod
    def _get_positional_features(sentences):
        """Generate positional quintile features."""
        num_sents = len(sentences)
        quintile_cutoff = num_sents / 5

        if quintile_cutoff == 0:
            return [{"DocTooSmallForQuintiles": 1} for _ in range(num_sents)]
        
        features = []
        for i in range(num_sents):
            q = int(i / quintile_cutoff)
      
            q = min(q, 6)
            features.append({f"DocumentPositionQuintile{q}": 1})
        
        return features