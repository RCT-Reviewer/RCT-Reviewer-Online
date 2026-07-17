"""
RCT-Reviewer NLP module.

Text processing utilities including abbreviation expansion,
sentence splitting, and text cleaning.

Copyright (C) 2026 Vihaan Sahu
RCT-Reviewer 
Based on RobotReviewer text processing by Iain Marshall, Joel Kuiper, Byron Wallace
"""

from rct_reviewer.nlp.abbreviation import extract_abbreviations

__all__ = ["extract_abbreviations"]