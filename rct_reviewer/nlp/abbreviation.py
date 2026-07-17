"""
Abbreviation extraction using the Schwartz-Hearst algorithm.

Modernized Python implementation for extracting abbreviation-definition
pairs from biomedical text.

Original algorithm by: Schwartz & Hearst (2003)
Python implementation based on work by: Vincent Van Asch and Phil Gooch
(MIT License)

RCT-Reviewer
Copyright (C) 2026 Vihaan Sahu
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AbbreviationPair:
    """An abbreviation and its definition."""
    abbreviation: str
    definition: str
    start_index: int
    end_index: int
    
    def __str__(self) -> str:
        return f"{self.abbreviation} = {self.definition}"


def extract_abbreviations(
    text: str,
    min_definition_length: int = 3,
    min_abbreviation_length: int = 2,
    max_abbreviation_length: int = 10,
    first_only: bool = True,
) -> dict[str, str]:
    """Extract abbreviation-definition pairs from text.
    
    Uses the Schwartz-Hearst algorithm to find short-form/long-form
    pairs in biomedical text.
    
    Args:
        text: Input text to process
        min_definition_length: Minimum words in definition
        min_abbreviation_length: Minimum characters in abbreviation
        max_abbreviation_length: Maximum characters in abbreviation
        first_only: Only keep first occurrence of each abbreviation
        
    Returns:
        Dictionary mapping abbreviations to definitions
    """
    pairs = extract_abbreviation_pairs(
        text=text,
        min_definition_length=min_definition_length,
        min_abbreviation_length=min_abbreviation_length,
        max_abbreviation_length=max_abbreviation_length,
    )
    
    result: dict[str, str] = {}
    for pair in pairs:
        abbr = pair.abbreviation.lower()
        if first_only and abbr in result:
            continue
        result[abbr] = pair.definition
    
    return result


def extract_abbreviation_pairs(
    text: str,
    min_definition_length: int = 3,
    min_abbreviation_length: int = 2,
    max_abbreviation_length: int = 10,
) -> list[AbbreviationPair]:
    """Extract abbreviation-definition pairs with position info.
    
    Args:
        text: Input text to process
        min_definition_length: Minimum words in definition
        min_abbreviation_length: Minimum characters in abbreviation
        max_abbreviation_length: Maximum characters in abbreviation
        
    Returns:
        List of AbbreviationPair objects
    """
    results: list[AbbreviationPair] = []
    

    pattern = re.compile(
        r'(?P<definition>[A-Za-z][A-Za-z\s,\-]+?)\s*'
        r'\((?P<abbreviation>[A-Za-z][A-Za-z0-9.\-]{1,' + str(max_abbreviation_length - 1) + r'})\)'
    )
    
    for match in pattern.finditer(text):
        definition = match.group("definition").strip()
        abbreviation = match.group("abbreviation").strip()
        
  
        if len(abbreviation) < min_abbreviation_length:
            continue
        
        definition_words = definition.split()
        if len(definition_words) < min_definition_length:
            continue
        
       
        if is_valid_abbreviation(definition, abbreviation):
            results.append(
                AbbreviationPair(
                    abbreviation=abbreviation,
                    definition=definition,
                    start_index=match.start(),
                    end_index=match.end(),
                )
            )
    
    return results


def is_valid_abbreviation(definition: str, abbreviation: str) -> bool:
    """Check if abbreviation is a valid short form of definition.
    
    The Schwartz-Hearst algorithm checks that:
    1. Abbreviation characters appear in order in the definition
    2. At least some characters match the start of definition words
    """
    def_clean = re.sub(r'[^A-Za-z0-9]', '', definition.lower())
    abbr_clean = re.sub(r'[^A-Za-z0-9]', '', abbreviation.lower())
    
    if not abbr_clean or not def_clean:
        return False
    

    def_words = definition.split()
    first_letters = ''.join(w[0].lower() for w in def_words if w)

    if abbr_clean[0] != first_letters[0]:
        return False
    
   
    def_idx = 0
    for char in abbr_clean:
        found = False
        while def_idx < len(def_clean):
            if def_clean[def_idx] == char:
                found = True
                def_idx += 1
                break
            def_idx += 1
        if not found:
           
            if char not in first_letters:
                return False
    
  
    matching_first_letters = sum(1 for c in abbr_clean if c in first_letters)
    if matching_first_letters < len(abbr_clean) * 0.4:  
        return False
    
    return True


def expand_abbreviations(
    text: str,
    abbreviations: dict[str, str] | None = None,
) -> str:
    """Expand abbreviations in text using provided or extracted pairs.
    
    Args:
        text: Input text
        abbreviations: Dict of abbreviations to definitions.
                      If None, extracts from text first.
                      
    Returns:
        Text with abbreviations expanded (first occurrence only)
    """
    if abbreviations is None:
        abbreviations = extract_abbreviations(text)
    
    result = text
    for abbr, definition in abbreviations.items():

        pattern = re.compile(
            r'\b' + re.escape(abbr) + r'\b(?!\s*\()',
            re.IGNORECASE
        )

        if definition.lower() not in result.lower():
            continue
        result = pattern.sub(f'{definition} ({abbr})', result, count=1)
    
    return result


def get_abbreviation_map(text: str) -> dict[str, str]:
    """Get a clean abbreviation map from text.
    
    Returns abbreviations in original case mapped to definitions.
    """
    pairs = extract_abbreviation_pairs(text)
    return {pair.abbreviation: pair.definition for pair in pairs}