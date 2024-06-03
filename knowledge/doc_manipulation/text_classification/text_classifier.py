from common import pretty_print
import re
from collections import Counter
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    # Remove special characters and keep only printable ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text


class LineClassifier:
    def __init__(self, line: str, line_number: int, previous_line: Optional[str] = None,
                 next_line: Optional[str] = None, keywords: Optional[List[str]] = None):
        self.line = line
        self.line_number = line_number
        self.previous_line = previous_line
        self.next_line = next_line
        self.keywords = keywords if keywords else []

        # Basic metrics
        self.char_count = len(line)
        self.word_count = len(line.split())
        self.sentence_count = self.count_sentences()
        self.line_length = len(line)

        # Structural metrics
        self.starts_with_digit = line[0].isdigit() if line else False
        self.ends_with_digit = line[-1].isdigit() if line else False
        self.starts_with_special_char = not line[0].isalnum() if line else False
        self.ends_with_special_char = not line[-1].isalnum() if line else False
        self.indentation_level = len(line) - len(line.lstrip())

        # Content-based metrics
        self.uppercase = line.isupper()
        self.lowercase = line.islower()
        self.mixed_case = not (self.uppercase or self.lowercase)

        self.numerical_content = any(char.isdigit() for char in line)
        self.numeric_sum = self.sum_numerical_values()

        self.punctuation_count = self.count_punctuation()

        # Utility metrics
        self.unique_words = len(set(line.split()))
        self.has_dates = self.contains_dates_times()
        self.has_urls = self.contains_urls()
        self.has_emails = self.contains_emails()
        self.has_hashtags = self.contains_hashtags()
        self.shortest_word = self.get_shortest_word()
        self.longest_word = self.get_longest_word()
        self.average_word_length = self.get_average_word_length()

        # Keyword analysis
        self.keyword_analysis = self.analyze_keywords()

    def contains_dates_times(self):
        """Checks if the text contains dates or times."""
        date_time_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates like 12/31/2020 or 1/1/20
            r'\b\d{4}-\d{2}-\d{2}\b',  # Dates like 2020-12-31
            r'\b\d{1,2}:\d{2}(?:AM|PM|am|pm)?\b',  # Times like 12:00, 1:00PM
            r'\b\d{1,2}:\d{2}:\d{2}\b'  # Times like 12:00:00
        ]
        return any(re.search(pattern, self.line) for pattern in date_time_patterns)

    def contains_urls(self):
        """Checks if the text contains URLs or links."""
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(self.line))

    def contains_emails(self):
        """Checks if the text contains email addresses."""
        email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        return bool(email_pattern.search(self.line))

    def contains_hashtags(self):
        """Checks if the text contains hashtags."""
        hashtag_pattern = re.compile(r'\B#\w*[a-zA-Z]+\w*')
        return bool(hashtag_pattern.search(self.line))

    def get_longest_word(self):
        """Finds the longest word in the text and its length."""
        if self.line:
            longest = max(self.line.split(), key=len)
            return longest
        return ""

    def get_shortest_word(self):
        """Finds the shortest word in the text and its length."""
        if self.line:
            shortest = min(self.line.split(), key=len)
            return shortest
        return ""

    def get_average_word_length(self):
        """Calculates the average length of words."""
        if self.line:
            return round(sum(len(word) for word in self.line.split()) / len(self.line))
        return 0.0

    def count_sentences(self) -> int:
        # Maybe incorrect in some cases but will give a rough idea
        return sum(self.line.count(p) for p in '.!?')

    def sum_numerical_values(self) -> int:
        return sum(int(num) for num in re.findall(r'\d+', self.line))

    def count_punctuation(self) -> int:
        return sum(1 for char in self.line if char in '.,;:!?')

    def analyze_keywords(self) -> Dict[str, Dict[str, bool]]:
        analysis = {}
        for keyword in self.keywords:
            analysis[keyword] = {
                "contains": keyword in self.line,
                "starts_with": self.line.startswith(keyword),
                "ends_with": self.line.endswith(keyword),
                "does_not_contain": keyword not in self.line,
                "does_not_start_with": not self.line.startswith(keyword),
                "does_not_end_with": not self.line.endswith(keyword)
            }
        return analysis

    def to_dict(self) -> Dict:
        return {
            "line_number": self.line_number,
            "line": self.line,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "line_length": self.line_length,
            "starts_with_digit": self.starts_with_digit,
            "ends_with_digit": self.ends_with_digit,
            "starts_with_special_char": self.starts_with_special_char,
            "ends_with_special_char": self.ends_with_special_char,
            "indentation_level": self.indentation_level,
            "uppercase": self.uppercase,
            "lowercase": self.lowercase,
            "mixed_case": self.mixed_case,
            "numerical_content": self.numerical_content,
            "numeric_sum": self.numeric_sum,
            "punctuation_count": self.punctuation_count,
            "unique_words": self.unique_words,
            "previous_line": self.previous_line,
            "next_line": self.next_line,
            "keyword_analysis": self.keyword_analysis,
            "has_urls": self.has_urls,
            "has_emails": self.has_emails,
            "has_hashtags": self.has_hashtags,
            "shortest_word": self.shortest_word,
            "longest_word": self.longest_word,
            "average_word_length": self.average_word_length
        }


class TextAnalyzer:
    def __init__(self, text: str, keywords: Optional[List[str]] = None):
        self.text = clean_text(text)
        self.keywords = keywords if keywords else []
        self.lines = self.text.split('\n')
        self.metadata = self.compute_metadata()
        self.analysis = self.analyze_text()

    def compute_metadata(self) -> Dict:
        total_lines = len(self.lines)
        total_chars = sum(len(line) for line in self.lines)
        total_words = sum(len(line.split()) for line in self.lines)
        total_sentences = sum(sum(line.count(p) for p in '.!?') for line in self.lines)

        keyword_counts = {keyword: 0 for keyword in self.keywords}
        for line in self.lines:
            for keyword in self.keywords:
                if keyword in line:
                    keyword_counts[keyword] += 1

        return {
            "total_lines": total_lines,
            "total_characters": total_chars,
            "total_words": total_words,
            "total_sentences": total_sentences,
        }

    def analyze_text(self) -> List[Dict]:
        analysis = []
        for i, line in enumerate(self.lines):
            previous_line = self.lines[i - 1] if i > 0 else None
            next_line = self.lines[i + 1] if i < len(self.lines) - 1 else None
            classifier = LineClassifier(line, i + 1, previous_line, next_line, self.keywords)
            analysis.append(classifier.to_dict())
        return analysis

    def get_analysis(self) -> Dict:
        return {
            "metadata": self.metadata,
            "lines": self.analysis
        }


# This is incomplete for now
class TextManipulation:
    """
    A text manipulation toolkit.

    **Things to know before use:**

    - **Identifiers** :
        An identifier is a structured label or name used to uniquely identify entities such as variables, objects, records, or functions within a given context.
        It enables the organization, reference, and manipulation of data consistently and coherently within a system.

        The identifier structure often includes conditions that determine the validity or selection criteria of the entities it represents.
        These conditions can be combined using logical operations:

        - **AND**: Combines multiple conditions, all of which must be true for the identifier to be valid. Used when every specified criterion needs to be satisfied.
        - **OR**: Combines multiple conditions, any one of which must be true for the identifier to be valid. Used when any one of several criteria being satisfied is sufficient.
        - **NOT**: Negates a condition, requiring it to be false for the identifier to be valid. Used to exclude entities that meet specific criteria.

        This logical structuring allows for flexible and precise control over the identification and selection of entities within the system.

        Example for an identifier:
            identifier = {
                "AND": [
                    {"metric": "startswith", "value": "Fuel", "case_sensitive": True},
                    {"metric": "contains", "value": "diesel", "case_sensitive": True},
                    {"metric": "has_url", "value": True}
                ],
                "OR": [
                    {"metric": "starts_with_special_char", "value": True},
                    {"metric": "ends_with_digit", "value": True},
                ],
                "NOT": [
                    {"metric": "contains", "value": "test", "case_sensitive": True},
                ]
            }
    """

    def __init__(self, document: str):
        self.document = document
        self.updated_document = document
        self.text_cleaning_replacements = \
            {
                # Replace specific Unicode punctuation with ASCII equivalents
                u"\u2018": "'",  # Left single quotation mark
                u"\u2019": "'",  # Right single quotation mark
                u"\u201c": '"',  # Left double quotation mark
                u"\u201d": '"',  # Right double quotation mark
                u"\u2013": "-",  # En dash
                u"\u2014": "--",  # Em dash
                u"\u2026": "..."  # Ellipsis
            }

    def line_qualifies_check(self, line: LineClassifier, line_identifier: dict) -> bool:
        def match_conditions(line_obj, conditions):
            for condition in conditions:

                if condition.get('metric') == "startswith":
                    return line_obj.line.startswith(condition.get('value'))

                elif condition.get('metric') == "contains":
                    return condition.get('value') in line_obj.line

                elif condition.get('metric') == "has_url":
                    return line_obj.has_urls

                elif condition.get('metric') == "has_emails":
                    return line_obj.has_emails

                elif condition.get('metric') == "starts_with_special_char":
                    return line_obj.starts_with_special_char

                elif condition.get('metric') == "ends_with_digit":
                    return line_obj.ends_with_digit

                elif condition.get('metric') == "starts_with_digit":
                    return line_obj.starts_with_digit

                elif condition.get('metric') == "ends_with_special_char":
                    return line_obj.ends_with_special_char

                elif condition.get('metric') == "indentation_level":
                    return line_obj.indentation_level == condition.get('value')

                elif condition.get('metric') == "uppercase":
                    return line_obj.uppercase

                elif condition.get('metric') == "lowercase":
                    return line_obj.lowercase

                elif condition.get('metric') == "mixed_case":
                    return line_obj.mixed_case

                elif condition.get('metric') == "numerical_content":
                    return line_obj.numerical_content

                elif condition.get('metric') == "numeric_sum":
                    return line_obj.numeric_sum == condition.get('value')

                elif condition.get('metric') == "punctuation_count":
                    return line_obj.punctuation_count == condition.get('value')

                elif condition.get('metric') == "unique_words":
                    return line_obj.unique_words == condition.get('value')

                elif condition.get('metric') == "has_hashtags":
                    return line_obj.has_hashtags

                elif condition.get('metric') == "shortest_word":
                    return line_obj.shortest_word == condition.get('value')

                elif condition.get('metric') == "longest_word":
                    return line_obj.longest_word == condition.get('value')

            # Incase the metric is not identified in the if else ladder, then just return False, means condition is not satisfied
            return False

        # Check "AND" conditions: All conditions in the "AND" list must be true for the line to pass.
        # If the "AND" key is in the identifier and any condition in the "AND" list is false, skip this line.
        if "AND" in line_identifier and not match_conditions(line, line_identifier["AND"]):
            return False

        # Check "OR" conditions: At least one condition in the "OR" list must be true for the line to pass.
        # If the "OR" key is in the identifier and none of the conditions in the "OR" list are true, skip this line.
        if "OR" in line_identifier and not any(match_conditions(line, [cond]) for cond in line_identifier["OR"]):
            return False

        # Check "NOT" conditions: All conditions in the "NOT" list must be false for the line to pass.
        # If the "NOT" key is in the identifier and any condition in the "NOT" list is true, skip this line.
        if "NOT" in line_identifier and any(match_conditions(line, [cond]) for cond in line_identifier["NOT"]):
            return False

        return True

    def limit_consecutive_empty_lines(self, document: str, max_empty_lines=2) -> str:
        """
        Limits the number of consecutive empty lines in a given text.

        Args:
            document (str): The input text to be processed.
            max_empty_lines (int): The maximum number of consecutive empty lines allowed.

        Returns:
            str: The cleaned text with limited consecutive empty lines.
        """
        lines = document.split('\n')
        cleaned_lines = []
        empty_count = 0

        for line in lines:
            if not line.strip():  # If the line is empty
                if empty_count < max_empty_lines:
                    cleaned_lines.append(line)
                empty_count += 1
            else:
                cleaned_lines.append(line)
                empty_count = 0  # Reset the count for non-empty lines

        return '\n'.join(cleaned_lines)

    def clean_document(self, document: str, replacements):
        """
        Cleans the input text by replacing certain Unicode characters with their ASCII equivalents
        and removing non-ASCII characters.

        Parameters:
        text (str): The input text to be cleaned.

        Returns:
        str: The cleaned text.
        """

        # Replace specific Unicode punctuation with ASCII equivalents
        if replacements is None:
            replacements = self.text_cleaning_replacements

        # Apply replacements if the characters are present
        for unicode_char, ascii_char in replacements.items():
            if unicode_char in document:
                document = document.replace(unicode_char, ascii_char)

        # Remove non-ASCII characters using regex
        document = re.sub(r'[^\x00-\x7F]+', '', document)

        return document

    def get_line_index_by_identifier(self, line_identifier: dict, after_line: int) -> int:
        """
        :param line_identifier: A basic identifier
        :param after_line: Line index, incase to check after a particular line
        :return: index of the line
        """

    def get_lines_by_document(self, document: str, doc_keywords= None):
        classifier = TextAnalyzer(document, doc_keywords)

        analysis = analyzer.get_analysis()

        lines = analysis.get('lines')

        return lines

    def add_dynamic_markers_single_line(self, document: str, line_identifier: dict,
                                        start_marker_lines: list[str],
                                        start_marker_location: str,
                                        end_marker_lines: list[str],
                                        end_marker_location: str):
        """
        This function targets to capture a **single line** around start and end marker
        Comments by Jatin: The intention behind making was , sometimes a single line needs to be wrapped with start and end markers.
                            Example : Wrapping chapter headers in step 2
        """

        lines = self.get_lines_by_document(document)
        doc_lines = document.split('\n')

        for line_obj in lines:
            line_number = line_obj.get('line_number')
            line_index = line_number - 1

            if self.line_qualifies_check(line_obj, line_identifier):
                new_list = doc_lines[:line_index] + start_marker_lines + doc_lines[line_index:]







    def add_dynamic_markers_multiline(self, document: str, start_line_identifier: dict, end_line_identifier: dict,
                                      add_endmarker_after_lines: int,
                                      start_marker: str, end_marker: str):
        """
        This function targets to capture a **multiline content** around start and end marker
        """
        pass

    def get_content_between_markers(self, start_identifier: list[str], end_identifier: list[str]):
        pass

    def get_content_between_markers_after_excluding(self, start_identifier: list[str], end_identifier: list[str]):
        pass




if __name__ == "__main__":
    # Need to resolve encoding error with ama_raw_text.txt
    with open(r'chapter_text_19.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    keywords = ["Measurement", "Rotation", "example"]
    analyzer = TextAnalyzer(text, keywords)
    analysis = analyzer.get_analysis()

    # Print analysis of the first few lines for demonstration
    for item in analysis["lines"][:1]:
        pretty_print(item)

    # Print metadata
    pretty_print(analysis['metadata'], title="Metadata")

    # Print the full classification
    # pretty_print(analysis)
