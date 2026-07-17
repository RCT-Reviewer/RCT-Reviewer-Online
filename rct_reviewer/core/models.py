from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Annotation(BaseModel):
    text: str
    start_index: int
    end_index: int

class BiasAnnotation(BaseModel):
    domain: str
    judgement: str
    text: List[str]

class PICOAnnotation(BaseModel):
    domain: str
    text: List[str]

class DocumentAnalysis(BaseModel):
    filename: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    rct: Optional[Dict[str, Any]] = None
    pico: List[PICOAnnotation] = []
    bias: List[BiasAnnotation] = []