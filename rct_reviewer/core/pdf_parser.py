import fitz  
import spacy
import logging
from rct_reviewer import config

log = logging.getLogger(__name__)

class PDFParser:
    def __init__(self):
        self.nlp = spacy.load(config.settings.spacy_model)

    def parse(self, pdf_bytes: bytes):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
       
        spacy_doc = self.nlp(text)
        sentences = [{
            "text": sent.text, 
            "start": sent.start_char, 
            "end": sent.end_char
        } for sent in spacy_doc.sents]
        
        return {
            "text": text,
            "sentences": sentences,
            "title": sentences[0]["text"] if sentences else "",
            "abstract": text[:3000] 
        }