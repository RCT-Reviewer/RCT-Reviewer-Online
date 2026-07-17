import joblib
import re
import logging
from rct_reviewer import get_data_path
from rct_reviewer.config import settings

log = logging.getLogger(__name__)

class Drugbank:
    def __init__(self):
        # Check config to decide loading method
        if settings.use_joblib:
            path = get_data_path('drugbank/drugbank.joblib')
            loader = lambda p: joblib.load(p)
        else:
            path = get_data_path('drugbank/drugbank.pck')
            loader = lambda p: __import__('pickle').load(open(p, 'rb'))

        if path.exists():
            try:
                self.data = loader(path)
            except Exception as e:
                log.warning(f"Failed to load drugbank from {path}: {e}. Using empty.")
                self.data = {}
        else:
            log.warning(f"Drugbank file not found at {path}")
            self.data = {}

    def contains_drug(self, text):
        tokens = re.split("([^A-Za-z0-9])", text)
        return 1 if self._find_matches(tokens) else 0

    def _find_matches(self, tokens):
        last_buffer = [[]]
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            for blist in last_buffer:
                key = "".join(blist + [token_lower])
                if self.data.get(key):
                    return True
        return False