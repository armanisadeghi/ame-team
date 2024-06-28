import time
import multidict
from common import pretty_print
import re, random, string, json, copy, uuid
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Any, Union, Literal, Type, get_type_hints
from difflib import SequenceMatcher
from automation_matrix.processing.markdown.classifier import OutputClassifier
import functools
import inspect
from collections import OrderedDict


def clean_text(text: str) -> str:
    # Remove special characters and keep only printable ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text


def get_next_and_previous_line(lines: list, line_obj):
    line_index = line_obj.get('line_number') - 1

    # Calculate previous line index and handle boundary condition
    if line_index > 0:
        previous_line_obj = lines[line_index - 1]
    else:
        previous_line_obj = None

    # Calculate next line index and handle boundary condition
    if line_index < len(lines) - 1:
        next_line_obj = lines[line_index + 1]
    else:
        next_line_obj = None

    return previous_line_obj, next_line_obj


class TextSimilarity:

    def similarity_between_two(self,
                               s1: str,
                               s2: str) -> float:
        """
        Calculates the similarity ratio between two strings using SequenceMatcher from difflib.

        Parameters:
        - s1 (str): First string for comparison.
        - s2 (str): Second string for comparison.

        Returns:
        - float: Similarity ratio between 0.0 and 1.0, where higher values indicate greater similarity.
        """
        matcher = SequenceMatcher(None, s1, s2)

        return round(matcher.ratio(), 2)

    def multi_similarity(self,
                         s1: str,
                         string_list: list,
                         fetch_string_func=None):
        """
        Calculates similarity ratios for a list of strings compared to a reference string and sorts them in descending order.

        Parameters:
        - str1 (str): The reference string.
        - str_list (list of str): The list of strings to compare with str1.

        Returns:
        - list of tuple: List of tuples, each containing a string from str_list and its similarity ratio, sorted by similarity ratio in descending order.
        """

        similarities = [
            (s, self.similarity_between_two(s1, fetch_string_func(s) if fetch_string_func else s))
            for s in string_list
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


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
        self.is_blank = not bool(line.strip())  # bool returns false for a empty line and True for non empty line

        # Structural metrics
        self.starts_with_digit = line[0].isdigit() if line else False
        self.ends_with_digit = line[-1].isdigit() if line else False
        self.starts_with_special_char = not line[0].isalnum() if line else False
        self.ends_with_special_char = not line[-1].isalnum() if line else False

        # Need to consider these cases too
        self.starts_with_digit_after_striping = line.strip()[0].isdigit() if not self.is_blank else False
        self.ends_with_digit_after_striping = line.strip()[-1].isdigit() if not self.is_blank else False
        self.starts_with_special_char_after_striping = not line.strip()[0].isalnum() if not self.is_blank else False
        self.ends_with_special_char_after_striping = not line.strip()[-1].isalnum() if not self.is_blank else False

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
        self.is_digits_and_spaces = all(c.isdigit() or c.isspace() for c in line)
        # Keyword analysis
        self.keyword_analysis = self.analyze_keywords()

        # Markdown
        self.has_markdown = self.contains_markdown()

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
        if not self.is_blank:
            longest = max(self.line.split(), key=len)
            return longest
        return ""

    def get_shortest_word(self):
        """Finds the shortest word in the text and its length."""
        if not self.is_blank:
            shortest = min(self.line.split(), key=len)
            return shortest
        return ""

    def get_average_word_length(self):
        """Calculates the average length of words."""
        if self.line:
            return round(sum(len(word) for word in self.line.split()) / len(self.line))
        return 0.0

    def contains_markdown(self):
        markdown_patterns = [
            r'^(#{1,6})\s(.+)$',  # Headers
            r'^(\*|\+|\-|\d+\.)\s(.+)$',  # Lists
            r'^\>\s(.+)$',  # Blockquotes
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'(\*\*|__)(.*?)\1',  # Bold text
            r'(\*|_)(.*?)\1',  # Italic text
            r'\[([^\]]+)\]\(([^\)]+)\)',  # Links
            r'\!\[([^\]]*)\]\(([^\)]+)\)',  # Images
            r'^(\-{3,}|\*{3,})$',  # Horizontal rules
            r'^\|(.+)\|$',  # Tables
            r'~~(.*?)~~'  # Strikethrough
        ]
        combined_pattern = re.compile('|'.join(markdown_patterns), re.MULTILINE)
        if combined_pattern.search(self.line):
            return True
        return False

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

            "starts_with_digit_after_striping": self.starts_with_digit_after_striping,
            "ends_with_digit_after_striping": self.ends_with_digit_after_striping,
            "starts_with_special_char_after_striping": self.starts_with_special_char_after_striping,
            "ends_with_special_char_after_striping": self.ends_with_special_char_after_striping,

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
            "has_date_time": self.has_dates,
            "has_hashtags": self.has_hashtags,
            "has_markdown": self.has_markdown,
            "shortest_word": self.shortest_word,
            "longest_word": self.longest_word,
            "average_word_length": self.average_word_length,
            "is_digit_and_spaces": self.is_digits_and_spaces
        }


class TextAnalyzer:
    def __init__(self, text: str, keywords: Optional[List[str]] = None):
        self.text = clean_text(text)
        self.keywords = keywords if keywords else []
        self.lines = self.text.split('\n')
        self.metadata = self.compute_metadata()
        self.analysis = self.analyze_text()

    def get_markdown_compatibility(self):
        classifier = OutputClassifier()
        sections = classifier.classify_output_details(self.text, clean_text=False)
        valid_section_count = 0
        plain_text_sections = 0
        # is_markdown = False
        for section_type, section_data in sections:
            if section_type == "plain_text" or section_type == 'other_section_type':
                # is_markdown = True
                plain_text_sections += 1
            else:
                valid_section_count += 1

        return valid_section_count, plain_text_sections

    def compute_metadata(self) -> Dict:
        total_lines = len(self.lines)
        total_chars = sum(len(line) for line in self.lines)
        total_words = sum(len(line.split()) for line in self.lines)
        total_sentences = sum(sum(line.count(p) for p in '.!?') for line in self.lines)
        total_markdown_sections, plain_text_sections = self.get_markdown_compatibility()

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
            "total_plain_text_sections": plain_text_sections,
            "total_markdown_sections": total_markdown_sections,
            # So it is good to consider total markdown sections instead of having something fixed called is_markdown_compatible,
            # as even if there is a single markdown section present we don't want to reach the conclusion that the whole text doc is a markdown. It should be dynamically
            # determined based on the total size and user preferences or some basic threshold.
            # Or instead of having these, we can have a markdown to plain text ratio, which will help how much is plain text and markdown
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


class Metric:

    def __init__(self, name=None, params=None):
        self.name = name
        self.params = params

    def from_dict(self, metric_dict: dict):
        """

        :param metric_dict: eg: {metric: starts_with, case_sensitive: True, strip: True}
        :return:
        """
        self.name = metric_dict.get('metric')
        del metric_dict['metric']
        self.params = metric_dict

    def to_dict(self):
        """
        :return: eg: {metric: starts_with, case_sensitive: True, strip: True}
        """
        ret_obj = self.params
        ret_obj['metric'] = self.name
        return ret_obj

    def int_comparison(self, compare_with, condition):
        if condition.get('greater_than'):
            return compare_with > condition.get('greater_than')

        elif condition.get('less_than'):
            return compare_with < condition.get('less_than')

        elif condition.get('equals'):
            return compare_with == condition.get('equals')

    def get_next_and_previous_line(self,
                                   lines: list,
                                   line_obj):
        line_index = line_obj.get('line_number') - 1

        # Calculate previous line index and handle boundary condition
        if line_index > 0:
            previous_line_obj = lines[line_index - 1]
        else:
            previous_line_obj = None

        # Calculate next line index and handle boundary condition
        if line_index < len(lines) - 1:
            next_line_obj = lines[line_index + 1]
        else:
            next_line_obj = None

        return previous_line_obj, next_line_obj

    def evaluate(self, line_obj, lines):

        # Checking what line it is for
        if self.params.get('check_line'):  # Indicates this might be a previous / next line case
            previous_line_obj, next_line_obj = self.get_next_and_previous_line(lines, line_obj)
            if self.params.get('check_line') == "previous":
                if previous_line_obj is None:
                    return False  # if the condition is to match previous line, and it's not present
                line_obj = previous_line_obj
            elif self.params.get('check_line') == "next":
                if next_line_obj is None:
                    return False  # if the condition is to match next line, and it's not present
                line_obj = next_line_obj
            else:  # if it's not previous or next it has to be the line itself, i.e self
                line_obj = line_obj

        if self.name == "starts_with":
            text_to_match = line_obj.get('line')
            value_to_match = self.params.get('equals')

            if self.params.get('strip'):
                text_to_match = text_to_match.strip()
            if not self.params.get('case_sensitive'):
                text_to_match = text_to_match.upper()
                value_to_match = value_to_match.upper()

            return text_to_match.startswith(value_to_match)

        elif self.name == 'ends_with':
            text_to_match = line_obj.get('line')
            value_to_match = self.params.get('equals')

            if self.params.get('strip'):
                text_to_match = text_to_match.strip()
            if not self.params.get('case_sensitive'):
                text_to_match = text_to_match.upper()
                value_to_match = value_to_match.upper()

            return text_to_match.endswith(value_to_match)

        elif self.name == 'has_markdown':
            return line_obj.get('has_markdown') == self.params.get('equals')

        elif self.name == 'contains_regex_pattern':
            text_to_match = line_obj.get('line')
            pattern = self.params.get('equals')

            if re.match(pattern, text_to_match):
                return True
            return False

        elif self.name == "contains":
            text_to_match = line_obj.get('line')
            value_to_match = self.params.get('equals')

            if self.params.get('strip'):
                text_to_match = text_to_match.strip()
            if not self.params.get('case_sensitive'):
                text_to_match = text_to_match.upper()
                value_to_match = value_to_match.upper()

            return value_to_match in text_to_match

        elif self.name == "has_url":
            return line_obj.get('has_urls') == self.params.get('equals')

        elif self.name == "has_emails":
            return line_obj.get('has_emails') == self.params.get('equals')

        elif self.name == "has_date_time":
            return line_obj.get('has_date_time') == self.params.get('equals')

        # Before removing trailing spaces
        elif self.name == "starts_with_special_char":
            return line_obj.get('starts_with_special_char') == self.params.get('equals')

        elif self.name == "ends_with_digit":
            return line_obj.get('ends_with_digit') == self.params.get('equals')

        elif self.name == "starts_with_digit":
            return line_obj.get('starts_with_digit') == self.params.get('equals')

        elif self.name == "ends_with_special_char":
            return line_obj.get('ends_with_special_char') == self.params.get('equals')

        # After trailing spaces removed
        elif self.name == "starts_with_special_char_after_striping":
            return line_obj.get('starts_with_special_char_after_striping') == self.params.get('equals')

        elif self.name == "ends_with_digit_after_striping":
            return line_obj.get('ends_with_digit_after_striping') == self.params.get('equals')

        elif self.name == "starts_with_digit_after_striping":
            return line_obj.get('starts_with_digit_after_striping') == self.params.get('equals')

        elif self.name == "ends_with_special_char_after_striping":
            return line_obj.get('ends_with_special_char_after_striping') == self.params.get('equals')

        elif self.name == "indentation_level":
            indentation_level = line_obj.get('indentation_level')

            return self.int_comparison(indentation_level, self.params)

        elif self.name == "uppercase":
            return line_obj.get('uppercase') == self.params.get('equals')

        elif self.name == "lowercase":
            return line_obj.get('lowercase') == self.params.get('equals')

        elif self.name == "mixed_case":
            return line_obj.get('mixed_case') == self.params.get('equals')

        elif self.name == "numerical_content":
            return line_obj.get('numerical_content') == self.params.get('equals')

        elif self.name == "numeric_sum":
            numeric_sum = line_obj.get('numeric_sum')

            return self.int_comparison(numeric_sum, self.params)

        elif self.name == "punctuation_count":
            punctuation_count = line_obj.get('punctuation_count')

            return self.int_comparison(punctuation_count, self.params)

        elif self.name == "unique_words":
            unique_words = line_obj.get('unique_words')

            return self.int_comparison(unique_words, self.params)

        elif self.name == "has_hashtags":
            return line_obj.get('has_hashtags') == self.params.get('equals')

        elif self.name == "shortest_word":
            value_to_match = self.params.get('equals')
            text_to_match = line_obj.get('shortest_word')
            if self.params.get('case_sensitive'):
                value_to_match = value_to_match.upper()
                text_to_match = text_to_match.upper()
            return text_to_match == value_to_match

        elif self.name == "longest_word":
            value_to_match = self.params.get('equals')
            text_to_match = line_obj.get('longest_word')
            if self.params.get('case_sensitive'):
                value_to_match = value_to_match.upper()
                text_to_match = text_to_match.upper()
            return text_to_match == value_to_match

        elif self.name == "char_count":
            char_count = line_obj.get('char_count')

            return self.int_comparison(char_count, self.params)

        elif self.name == "word_count":
            word_count = line_obj.get('word_count')

            return self.int_comparison(word_count, self.params)

        elif self.name == "sentence_count":
            sentence_count = line_obj.get('sentence_count')

            return self.int_comparison(sentence_count, self.params)

        elif self.name == "line_length":
            line_length = line_obj.get('line_length')

            return self.int_comparison(line_length, self.params)

        elif self.name == "average_word_length":
            average_word_length = line_obj.get('average_word_length')
            return self.int_comparison(average_word_length, self.params)

        elif self.name == 'is_digit_and_spaces':
            return line_obj.get('is_digit_and_spaces') == self.params.get('equals')

        elif self.name == 'text_similarity_score':
            similarity = TextSimilarity()
            line_content = line_obj.get('line')
            text_to_match = self.params.get('text')
            score = similarity.similarity_between_two(text_to_match, line_content)
            return self.int_comparison(score, self.params)


class Condition:
    def __init__(self,
                 condition_type: str,
                 metrics: list[Metric],
                 ):

        self.condition_type = condition_type
        self.metrics = metrics

    def cached_evaluation(self,
                          line_obj,
                          stored_evaluations: List[Dict[str, Any]],
                          ):
        """
        This is for cached evaluations only, cached means using results from stored metric evaluations
        :param line_obj: Line object
        :param stored_evaluations: The stored evaluations
        :return:

        example of stored evaluations
        stored_evaluations = {"metric1": [1,3,5], "metric2": [1,8,9], "metric3": [1,12, 13]}
        metrics =  {metric1 : {metric: starts_with, equals: The e}...}
        """

        current_line_number = line_obj.get('line_number')
        metric_evaluation_list = []

        line_satisfies_all_metrics = True

        for metric in self.metrics:
            metric_found = False  # Metric does not exist in cache
            for evaluation in stored_evaluations:
                if evaluation.get('metric').to_dict() == metric.to_dict():
                    metric_found = True  # Metric exist in cache
                    if current_line_number not in evaluation.get('passed_lines'):
                        line_satisfies_all_metrics = False

            # Metric does not exist in cache, we cannot
            if not metric_found:
                line_satisfies_all_metrics = False

            metric_evaluation_list.append(line_satisfies_all_metrics)

        if self.condition_type == "AND" and not all(metric_evaluation_list):
            return False

        # Check "OR" conditions: At least one condition in the "OR" list must be true for the line to pass.
        # If the "OR" key is in the identifier and none of the conditions in the "OR" list are true, skip this line.
        if self.condition_type == "OR" and not any(metric_evaluation_list):
            return False

        # Check "NOT" conditions: All conditions in the "NOT" list must be false for the line to pass.
        # If the "NOT" key is in the identifier and any condition in the "NOT" list is true, skip this line.
        if self.condition_type == "NOT" and any(metric_evaluation_list):
            return False

        # Exclusive OR (Only one of the given conditions must be true)
        if self.condition_type == "XOR" and sum(metric_evaluation_list) != 1:
            return False

        # The NAND operation is the negation of the AND operation. It returns True unless both inputs are True.
        if self.condition_type == "NAND" and all(metric_evaluation_list):
            return False

        # The NOR operation is the negation of the OR operation. It returns True only if both inputs are False.
        if self.condition_type == "NOR" and any(metric_evaluation_list):
            return False

        # The XNOR operation is the negation of the XOR (Exclusive OR) operation. It returns True if both inputs are the same (both True or both False).
        if self.condition_type == "XNOR" and sum(metric_evaluation_list) not in [0, len(self.metrics)]:
            return False

        # The next two only excepts two metrics only.

        if self.condition_type == "IMPLICATION":
            # IMPLICATION expects two metrics, metrics[0] implies metrics[1]
            A = metric_evaluation_list[0]
            B = metric_evaluation_list[1]

            if A and not B:  # If first metric is evaluated to True and second is not Implication is failed
                return False

        if self.condition_type == "BICONDITIONAL":
            # BICONDITIONAL expects two metrics, metrics[0] iff metrics[1]
            A = metric_evaluation_list[0]
            B = metric_evaluation_list[1]

            if A != B:
                return False

        return True

    def evaluate(self, line_obj, lines):

        if self.condition_type == "AND" and not all(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # Check "OR" conditions: At least one condition in the "OR" list must be true for the line to pass.
        # If the "OR" key is in the identifier and none of the conditions in the "OR" list are true, skip this line.
        if self.condition_type == "OR" and not any(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # Check "NOT" conditions: All conditions in the "NOT" list must be false for the line to pass.
        # If the "NOT" key is in the identifier and any condition in the "NOT" list is true, skip this line.
        if self.condition_type == "NOT" and any(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # Exclusive OR (Only one of the given conditions must be true)
        if self.condition_type == "XOR" and sum(metric.evaluate(line_obj, lines) for metric in self.metrics) != 1:
            return False

        # The NAND operation is the negation of the AND operation. It returns True unless both inputs are True.
        if self.condition_type == "NAND" and all(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # The NOR operation is the negation of the OR operation. It returns True only if both inputs are False.
        if self.condition_type == "NOR" and any(metric.evaluate(line_obj, lines) for metric in self.metrics):
            return False

        # The XNOR operation is the negation of the XOR (Exclusive OR) operation. It returns True if both inputs are the same (both True or both False).
        if self.condition_type == "XNOR" and sum(metric.evaluate(line_obj, lines) for metric in self.metrics) not in [0,
                                                                                                                      len(self.metrics)]:
            return False

        # The next two only excepts two metrics only.

        if self.condition_type == "IMPLICATION":
            # IMPLICATION expects two metrics, metrics[0] implies metrics[1]
            A = self.metrics[0].evaluate(line_obj, lines)
            B = self.metrics[1].evaluate(line_obj, lines)

            if A and not B:  # If first metric is evaluated to True and second is not Implication is failed
                return False

        if self.condition_type == "BICONDITIONAL":
            # BICONDITIONAL expects two metrics, metrics[0] iff metrics[1]
            A = self.metrics[0].evaluate(line_obj, lines)
            B = self.metrics[1].evaluate(line_obj, lines)

            if A != B:
                return False

        return True


class LineIdentifier:
    """
    Example of a metric list:

        metrics = [
            ('metric1', {'option1': 'value1'}),
            ('metric2', {'option2': 'value2'}),
            ('metric3', {'option3': 'value3'}),
        ]

    **"AND": "Used for ensuring all conditions are true",**

    **"NOT": "Used for ensuring the condition is false",**

    **"OR": "Used for ensuring at least one condition is true",**

    **"XOR": "Used for ensuring exactly one condition is true",**

    **"NAND": "Used for ensuring not all conditions are true",**

    **"NOR": "Used for ensuring all conditions are false",**

    **"XNOR": "Used for ensuring conditions are either all true or all false",**

    **"IMPLICATION": "Used for ensuring if the first condition is true, the second must be true",**

    **"BICONDITIONAL": "Used for ensuring conditions are both true or both false"**
    """

    def __init__(self):
        self.identifier = defaultdict()
        self.idx = None
        self.groups_meanings = {
            "AND": "Used for ensuring all conditions are true",
            "NOT": "Used for ensuring the condition is false",
            "OR": "Used for ensuring at least one condition is true",
            "XOR": "Used for ensuring exactly one condition is true",
            "NAND": "Used for ensuring not all conditions are true",
            "NOR": "Used for ensuring all conditions are false",
            "XNOR": "Used for ensuring conditions are either all true or all false",
            "IMPLICATION": "Used for ensuring if the first condition is true, the second must be true",
            "BICONDITIONAL": "Used for ensuring conditions are both true or both false"
        }
        self.conditions: list[Condition] = []

    def add_condition(self, condition: Condition):
        self.conditions.append(condition)

    def evaluate(self, line_obj, lines, use_saved_metrics=False, saved_metrics: List[Dict[str, Any]] = None):

        if use_saved_metrics and saved_metrics is not None:
            for condition in self.conditions:
                if not condition.cached_evaluation(line_obj, saved_metrics):
                    return False
            return True
        else:

            for condition in self.conditions:
                if not condition.evaluate(line_obj, lines):
                    return False
            return True

    def _add_condition(self, conditional: str, metrics: list):
        conditional_type = conditional
        metrics = [Metric(metric_name, metric_params) for metric_name, metric_params in metrics]

        condition_obj = Condition(conditional_type, metrics)

        self.conditions.append(condition_obj)

    def ADD_AND(self, metrics):
        self._add_condition("AND", metrics=metrics)

    def ADD_NOT(self, metrics):
        self._add_condition("NOT", metrics=metrics)

    def ADD_OR(self, metrics):
        self._add_condition("OR", metrics=metrics)

    def ADD_XOR(self, metrics):
        self._add_condition("XOR", metrics=metrics)

    def ADD_NAND(self, metrics):
        self._add_condition("NAND", metrics=metrics)

    def ADD_NOR(self, metrics):
        self._add_condition("NOR", metrics=metrics)

    def ADD_XNOR(self, metrics):
        self._add_condition("XNOR", metrics=metrics)

    def ADD_IMPLICATION(self, metrics):
        if len(metrics) < 2:
            return
        self._add_condition("IMPLICATION", metrics=metrics[:1])

    def ADD_BICONDITIONAL(self, metrics):
        if len(metrics) < 2:
            return
        self._add_condition("BICONDITIONAL", metrics=metrics[:1])

    def reset(self):
        self.conditions = []

    def validate_structure_from_dict(self, identifier):

        if not isinstance(identifier, dict):
            return False

        if not all([isinstance(x, str) and x.isdigit() for x in identifier.keys()]):
            return False

        if not all([isinstance(x, dict) for x in identifier.values()]):
            return False

        if not all([x.get('type') in self.groups_meanings.keys() for x in identifier.values()]):
            return False

        if not all([isinstance(x.get('metrics'), list) for x in identifier.values()]):
            return False

        return True

    def from_dict(self, identifier: dict):
        # Utility function to load identifier from db (possibly)

        # Validating structure
        if not self.validate_structure_from_dict(identifier):
            return False

        for _, idf in identifier.items():
            metric_type = idf.get('type')
            metrics = idf.get('metrics')

            formatted_metrics = []
            for metric in metrics:
                formatted_metrics.append((metric.get('metric'), metric))

            self._add_condition(metric_type, formatted_metrics)

        return True

    def to_dict(self):
        structure = {}
        for idx, condition in enumerate(self.conditions):
            _type = condition.condition_type
            _metrics = condition.metrics
            _str_idx = str(idx)
            structure[_str_idx] = {
                "type": _type,
                "metrics": []
            }
            for _metric in _metrics:
                name, params = _metric.name, _metric.params

                params['metric'] = name

                structure[_str_idx]['metrics'].append(params)

        return structure


class TextManipulation:
    """
    A text manipulation toolkit.

    **Things to know before use:**

    - **Identifiers** :
        An identifier is a structured label or name used to uniquely identify lines within a given context.
        It enables the  reference, and manipulation of data consistently and coherently within a system.

        This logical structuring allows for flexible and precise control over the identification and selection of lines within the system.

        Example for an identifier:
            identifier = {
                        0: {
                            "type": "AND",
                            "metrics": [
                                {"metric": "startswith", "equals": "Fuel", "case_sensitive": True},
                                {"metric": "contains", "equals": "diesel", "case_sensitive": True},
                                {"metric": "has_url", "equals": True}
                            ]
                        },
                        1: {
                            "type": "OR",
                            "metrics": [
                                {"metric": "starts_with_special_char", "equals": True},
                                {"metric": "ends_with_digit", "equals": True}
                            ]
                        },
                        2: {
                            "type": "NOT",
                            "metrics": [
                                {"metric": "contains", "equals": "test", "case_sensitive": True}
                            ]
                        }
                    }

        Or use LineIdentifier class to build an identifier.
    """

    def __init__(self):
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
        self.by_similarity = TextSimilarity()

    def line_qualifies_check(self,
                             line_obj: LineClassifier,
                             line_identifier: LineIdentifier,
                             lines,
                             ) -> bool:

        return line_identifier.evaluate(line_obj, lines)

    def limit_consecutive_empty_lines(self,
                                      document: str,
                                      max_empty_lines=2) -> str:
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

    def clean_document(self,
                       document: str,
                       replacements=None):
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

    def get_lines_by_document(self,
                              document: str,
                              doc_keywords=None):
        """
        Utility function , to get line objects in supplied document
        """
        classifier = TextAnalyzer(document, doc_keywords)

        analysis = classifier.get_analysis()

        lines = analysis.get('lines')

        return lines

    def insert_breaks(self,
                      document: str,
                      primary_line_identifier: LineIdentifier,
                      ignore_start_identifier: LineIdentifier = None,
                      ignore_end_identifier: LineIdentifier = None,
                      ignore_start_pattern: str = None,
                      ignore_end_pattern: str = None,
                      break_pattern: str = '--break--', ):
        """
        Inserts a specified break pattern into a document based on primary line identifiers,
        while optionally ignoring sections defined by start and end identifiers or patterns.

        Args:
            document (str): The document content to process.
            primary_line_identifier (dict): A dictionary defining the primary line identifier criteria.
            ignore_start_identifier (dict): A dictionary defining the start of the section to ignore.
            ignore_end_identifier (dict): A dictionary defining the end of the section to ignore.
            ignore_start_pattern (str, optional): A regex pattern defining the start of the section to ignore.
            ignore_end_pattern (str, optional): A regex pattern defining the end of the section to ignore.
            break_pattern (str, optional): The break pattern to insert in the document. Defaults to '--break--'.

        Returns:
            str: The modified document with breaks inserted.

        The function processes the document line by line. If ignore_start_pattern and ignore_end_pattern are provided,
        it uses these patterns to identify sections to ignore. Otherwise, it uses ignore_start_identifier and
        ignore_end_identifier for the same purpose. The break_pattern is inserted before lines that match
        primary_line_identifier unless the line is within an ignored section.
        """

        ignore_by_identifier = True

        if ignore_start_pattern and ignore_end_pattern:
            ignore_start_pattern = re.compile(ignore_start_pattern)
            ignore_end_pattern = re.compile(ignore_end_pattern)
            ignore_by_identifier = False
        elif ignore_start_identifier and ignore_end_identifier:
            ignore_by_identifier = True

        lines = self.get_lines_by_document(document)
        within_ignore_section = False
        updated_doc_list = []

        for line in lines:
            line_content = line.get('line', '')
            if ignore_by_identifier:
                if self.line_qualifies_check(line, ignore_start_identifier, lines):
                    within_ignore_section = True

                if within_ignore_section:
                    updated_doc_list.append(line_content)
                    if self.line_qualifies_check(line, ignore_end_identifier, lines):
                        within_ignore_section = False
                else:
                    if self.line_qualifies_check(line, primary_line_identifier, lines):
                        updated_doc_list.append(break_pattern)
                    updated_doc_list.append(line_content)
            else:
                if ignore_start_pattern.match(line_content.strip()):
                    within_ignore_section = True

                if within_ignore_section:
                    updated_doc_list.append(line_content)
                    if ignore_end_pattern.match(line_content.strip()):
                        within_ignore_section = False
                else:
                    if self.line_qualifies_check(line, primary_line_identifier, lines):
                        updated_doc_list.append(break_pattern)
                    updated_doc_list.append(line_content)

        return '\n'.join(updated_doc_list)

    def insert_character_count_markers(self,
                                       document: str,
                                       existing_break_marker: str = '--break--',
                                       max_chars=3000):

        accumulated_text = ''
        updated_text = ''
        char_count_since_last_marker = 0
        total_marker_count = 0
        char_marker_count = 0
        longest_section_length = 0
        lines = self.get_lines_by_document(document)

        for line in lines:
            line_content = line.get('line')
            if accumulated_text:
                accumulated_text += '\n'
            accumulated_text += line_content
            char_count_since_last_marker += len(line_content) + 1  # Including new line character

            if existing_break_marker in line_content:
                total_marker_count += 1
                if char_count_since_last_marker >= max_chars:
                    marker_position = accumulated_text.rfind(existing_break_marker)
                    accumulated_text = accumulated_text[
                                       :marker_position] + f"-- chars {char_count_since_last_marker} --\n" + accumulated_text[
                                                                                                             marker_position:]
                    char_marker_count += 1
                    char_count_since_last_marker = 0
                updated_text += accumulated_text
                longest_section_length = max(longest_section_length, len(accumulated_text))
                accumulated_text = ''
            elif char_count_since_last_marker >= max_chars:
                accumulated_text += f"\n-- chars {char_count_since_last_marker} --\n"
                char_count_since_last_marker = 0
                char_marker_count += 1
                updated_text += accumulated_text
                longest_section_length = max(longest_section_length, len(accumulated_text))
                accumulated_text = ''

        if accumulated_text:
            updated_text += accumulated_text
            longest_section_length = max(longest_section_length, len(accumulated_text))

        updated_text = updated_text.replace(existing_break_marker, '')

        # print(f"Total number of markers found: {total_marker_count}")
        # print(f"Total number of character count markers added: {char_marker_count}")
        # print(f"Length of the longest section: {longest_section_length}")

        return updated_text


class FindLines(TextManipulation):
    """
    This class is used for filtering and finding Line Objects by these ways:
     - Filtering By LineIdentifier
     - Filtering By Regex Pattern
     - Filtering By Text Similarity Score
    """

    def __init__(self):
        super().__init__()

    def filter_lines_by_saved_metrics(self,
                                      classified_lines: list,
                                      line_identifier: LineIdentifier,
                                      saved_metrics: List[Dict[str, Any]]) \
            -> list[Dict]:

        filtered_lines = []
        for line in classified_lines:
            if line_identifier.evaluate(line, classified_lines, use_saved_metrics=True, saved_metrics=saved_metrics):
                filtered_lines.append(line)

        return filtered_lines

    def filter_lines(self,
                     document: str,
                     line_identifier: LineIdentifier) -> list[Dict]:
        """
        Filters lines in a document based on a Line Identifier  and returns a list of line objects.

        This utility function searches through each line in the `document` and checks if the line qualifies
        the identifier check. If a line matches, it is included in the list of line objects returned.

        Parameters:
        - document (str): The text document to filter lines from.
        - line_identifier (dict): The Line identifier used to match lines.

        Returns:
        - list: A list of line objects that match the identifier pattern. Each line object typically contains
          information such as line content, line number, etc.
        """
        lines = self.get_lines_by_document(document)
        filtered_lines = []
        for line in lines:
            if self.line_qualifies_check(line, line_identifier, lines):
                filtered_lines.append(line)

        return filtered_lines

    def filter_lines_by_regex(self,
                              document: Union[List, str],
                              pattern) -> list[Dict]:
        """
        Filters lines in a document based on a regex pattern and returns a list of line objects.

        This utility function searches through each line in the `document` and checks if the line content
        matches the specified `pattern`. If a line matches, it is included in the list of line objects returned.

        Parameters:
        - document (str): The text document to filter lines from.
        - pattern (str): The regex pattern used to match lines.

        Returns:
        - list: A list of line objects that match the regex pattern. Each line object typically contains
          information such as line content, line number, etc.
        """

        if isinstance(document, list):
            lines = document
        else:
            lines = self.get_lines_by_document(document)
        results = []

        for line in lines:
            line_content = line.get('line')
            if re.search(pattern, line_content):
                results.append(line)
        return results

    def filter_similar_lines(self,
                             document: str,
                             lookup_text: str,
                             filter_identifier: LineIdentifier = None,
                             percentage: float = 0.5,
                             ):

        lines = self.get_lines_by_document(document)

        if filter_identifier is not None:
            lines = self.filter_lines(document, filter_identifier)

        ranked_lines = self.by_similarity.multi_similarity(
            lookup_text,
            lines,
            lambda x: x.get('line')
        )

        return [x for x, y in ranked_lines if y >= percentage]


class EditText(TextManipulation):
    """
    This class offers operations like
     - Replace items by line identifier or replacement in whole document
     - Delete lines by patterns or text
     - Modify (delete , edit) lines surrounded by A Identifier within a range
     - Shift lines up or down based on identifiers
    """

    def __init__(self):
        super().__init__()

    def replace_items(self,
                      document: str,
                      replacements: list[dict],
                      line_identifier: LineIdentifier = None):
        """
        Replaces specified items in a document based on given patterns and texts. If no line identifier is provided,
        the replacements will apply to the entire document.

        This function takes a list of replacements where each replacement specifies a pattern and/or text to be
        replaced with a provided replacement string. The replacements can be performed based on regex patterns or
        exact text matches.

        Examples of replacements:
        - `[{ pattern: None, text: "Something", replacement: "Not something" }]`: Replaces all occurrences of the exact text
          "Something" with "Not something".
        - `[{ pattern: r'\\.exe', text: None, replacement: ".py" }]`: Replaces all occurrences of text matching the regex
          pattern `\\.exe` with ".py".
        - `[{ pattern: r'The US Navy has \\.\\d+', text: "Confidential Data", replacement: "[REDACTED]" }]`: Replaces text
          matching the regex pattern `The US Navy has \\.\\d+` and the exact text "Confidential Data" with "[REDACTED]".

        Parameters:
        - document (str): The text document where replacements will be performed.
        - replacements (list): A list of dictionaries, each containing:
          - pattern (str or None): A regex pattern to match text for replacement. If None, an exact text match is used.
          - text (str or None): The exact text to match for replacement. If None, the pattern is used.
          - replacement (str): The replacement text for matched patterns or exact text matches.

        Returns:
        - str: The updated document with specified replacements applied.
        """

        def do_replacement(replacement_config: list[dict], text_data: str):
            updated_text = text_data

            for replacement in replacement_config:
                pattern = replacement.get('pattern')
                search_text = replacement.get('text')
                replacement_text = replacement.get('replacement')

                if pattern:
                    updated_text = re.sub(pattern, replacement_text, updated_text)

                if search_text and search_text in text_data:
                    updated_text = updated_text.replace(search_text, replacement_text)

            return updated_text

        if line_identifier is not None:
            updated_doc_list = []
            lines = self.get_lines_by_document(document)

            for line in lines:
                line_content = line.get('line')
                if self.line_qualifies_check(line, line_identifier, lines):
                    line_content = do_replacement(replacements, line_content)

                updated_doc_list.append(line_content)

            return '\n'.join(updated_doc_list)
        else:
            return do_replacement(replacements, document)

    def delete_lines(self,
                     document: str,
                     deletions: list[dict],
                     line_identifier: LineIdentifier = None):
        """
        Deletes specified items from a document based on given patterns and texts. If no line identifier is provided,
        the deletions will apply to the entire document.

        This function takes a list of deletions where each deletion specifies a pattern and/or text to be removed
        or replaced. The deletions can be performed based on regex patterns or exact text matches.

        Examples of deletions:
        - `[{ pattern: None, text: "Something" }]`: Deletes all occurrences of the exact text "Something".
        - `[{ pattern: r'\\.exe', text: None }]`: Deletes all occurrences of text matching the regex pattern `\\.exe`.
        - `[{ pattern: r'The US Navy has \\.\\d+', text: "Confidential Data" }]`: Replaces text matching the regex pattern
          `The US Navy has \\.\\d+` with "Confidential Data".

        Parameters:
        - document (str): The text document where deletions will be performed.
        - deletions (list): A list of dictionaries, each containing:
          - pattern (str or None): A regex pattern to match text for deletion or replacement. If None, an exact text match is used.
          - text (str or None): The exact text to delete, or the replacement text when used with a pattern. If None, matched text is deleted.

        Returns:
        - str: The updated document with specified deletions and replacements applied.
        """

        def should_delete_line(deletion_config: list[dict], text_data: str):
            for deletion in deletion_config:
                pattern = deletion.get('pattern')
                search_text = deletion.get('text')

                if pattern and re.search(pattern, text_data):
                    return True

                if search_text and search_text in text_data:
                    return True

            return False

        lines = self.get_lines_by_document(document)

        if line_identifier is not None:
            updated_doc_list = []

            for line in lines:
                line_content = line.get('line')
                if self.line_qualifies_check(line, line_identifier, lines):
                    if not should_delete_line(deletions, line_content):
                        updated_doc_list.append(line_content)
                else:
                    updated_doc_list.append(line_content)

            return '\n'.join(updated_doc_list)
        else:
            updated_lines = [line.get('line') for line in lines if not should_delete_line(deletions, line.get('line'))]
            return '\n'.join(updated_lines)

    def modify_surrounding_lines(self,
                                 document: str,
                                 primary_line_identifier: LineIdentifier,
                                 secondary_line_identifier: LineIdentifier,
                                 discover_above_lines: int = 2,
                                 discover_below_lines: int = 2,
                                 action: str = "delete_line",
                                 **kwargs) -> str:
        """
        Modifies lines surrounding a primary identifier line in a document based on specified actions.

        This function performs one of three actions (replace by regex, replace characters, or delete line) on lines
        found within a specified range above or below a primary identifier line.

        Only three actions are supported:
        - replace_by_regex: Requires 'pattern' (str) and 'replace_with' (str).
        - replace_chars: Requires 'character' (str) and 'replace_with' (str).
        - delete_line: Deletes the discovered line.

        Parameters:
        document (str): The document to modify.
        primary_line_identifier (dict): The identifier to locate the primary line.
        secondary_line_identifier (dict): The identifier to locate lines to modify.
        discover_above_lines (int): Number of lines above the primary line to check.
        discover_below_lines (int): Number of lines below the primary line to check.
        action (str): The action to perform on discovered lines ('delete_line', 'replace_chars', 'replace_by_regex').
        kwargs: Additional parameters for actions, such as 'character', 'replace_with', and 'pattern'.

        Returns:
        str: The modified document.
        """
        lines = self.get_lines_by_document(document)
        ignore_line_numbers = []

        def action_delete(line_obj):
            ignore_line_numbers.append(line_obj.get('line_number'))

        def action_replace_characters(line_obj, character, replace_with):
            if character in line_obj.get('line'):
                line_obj['line'] = line_obj.get('line').replace(character, replace_with)

        def action_replace_characters_by_regex(line_obj, pattern, replace_with):
            line_obj['line'] = re.sub(pattern, replace_with, line_obj.get('line'))

        actions = {
            "replace_by_regex": action_replace_characters_by_regex,
            "replace_chars": action_replace_characters,
            "delete_line": action_delete
        }

        for idx, line in enumerate(lines):
            if self.line_qualifies_check(line, primary_line_identifier, lines):
                # Check above lines
                start_idx = max(0, idx - discover_above_lines)
                above_lines = lines[start_idx:idx]

                for i, above_line in enumerate(above_lines):
                    if self.line_qualifies_check(above_line, secondary_line_identifier, lines):
                        if action in actions:
                            actions[action](above_line, **kwargs)

                # Check below lines
                end_idx = min(len(lines), idx + discover_below_lines + 1)
                below_lines = lines[idx + 1:end_idx]

                for i, below_line in enumerate(below_lines):
                    if self.line_qualifies_check(below_line, secondary_line_identifier, lines):
                        if action in actions:
                            actions[action](below_line, **kwargs)

        updated_doc_list = [line.get('line') for line in lines if line.get('line_number') not in ignore_line_numbers]
        return '\n'.join(updated_doc_list)

    def shift_and_join_lines(self,
                             document: str,
                             line_identifier: LineIdentifier,
                             distance: int,
                             direction: str = 'up',
                             strip_before_join: bool = True,
                             keep_line_after_moving: bool = False,
                             join_with: str = ' ',
                             ):
        """
        Moves a specified line in a document either up or down by a given number of lines.

        This function allows you to move a line identified by a unique line identifier a certain number of lines
        upwards or downwards within a document. It can also strip trailing spaces before joining the content of lines
        and decide whether to keep the moved line at its original position or remove it.

        :param document: The entire text document as a string.
        :param line_identifier: A unique identifier to locate the specific line to be moved.
        :param distance: The number of lines to move the identified line, excluding the target line itself.
        :param direction: Specifies the direction to move the line; 'up' for upwards and 'down' for downwards.
        :param strip_before_join: If True, removes trailing spaces before joining the content of the lines.
        :param keep_line_after_moving: If True, retains the original line in its position after moving.
        :param join_with: The character or string to use for joining two lines. Default is a space character.
        :return: The updated document as a string.
        """

        lines = self.get_lines_by_document(document)
        check_after_index = 0
        for idx, line in enumerate(lines):
            if idx < check_after_index:
                continue

            if self.line_qualifies_check(line, line_identifier, lines):
                if direction == "down":
                    join_idx = idx + distance
                else:
                    join_idx = idx - distance

                if 0 <= join_idx < len(lines):
                    join_line = lines[join_idx]

                    if strip_before_join:
                        line['line'] = f"{join_line.get('line').strip()}{join_with}{line.get('line').strip()}"
                    else:
                        line['line'] = f"{join_line.get('line')}{join_with}{line.get('line')}"

                    if not keep_line_after_moving:
                        lines[join_idx]['line'] = ''

                    check_after_index = max(check_after_index, idx + 1) if direction == "down" else idx

        return '\n'.join([x.get('line') for x in lines])

    def shift_and_join_lines_by_primary_identifier(self,
                                                   document: str,
                                                   primary_identifier: LineIdentifier,
                                                   secondary_identifier: LineIdentifier,
                                                   distance: int = 2,
                                                   direction: str = "up",
                                                   keep_line_after_moving: bool = False,
                                                   join_with: str = ' ',
                                                   strip_before_join: bool = True):
        """
        Joins two lines together in a document based on specified criteria.

        This function merges a primary line with another line identified within a given distance and direction.
        If a secondary identifier is not provided, the line at the specified distance is used as the secondary line.
        Trailing spaces can be removed before joining, and the original lines can be kept or removed after merging.

        :param document: The entire text document as a string.
        :param primary_identifier: A unique identifier to locate the primary line to be joined.
        :param secondary_identifier: A unique identifier for the secondary line. If not given, the line at the specified distance is used.
        :param distance: The number of lines to look for the secondary line, excluding the primary line itself.
        :param direction: Specifies the direction to look for the secondary line; 'up' for upwards and 'down' for downwards.
        :param keep_line_after_moving: If True, retains the original lines in their positions after joining.
        :param join_with: The character or string to use for joining the two lines. Default is a space character.
        :param strip_before_join: If True, removes trailing spaces before joining the content of the lines.
        :return: The updated document as a string.
        """

        check_after_index = 0

        lines = self.get_lines_by_document(document)

        for idx, line in enumerate(lines):
            if idx < check_after_index:
                continue

            if self.line_qualifies_check(line, primary_identifier, lines):  # Detected a primary line

                if direction == "down":
                    lookup_lines = lines[
                                   idx + 1: idx + distance + 1]  # As we want to look after the primary identifier
                    # Look till the distance
                else:
                    lookup_lines = lines[max(0, idx - distance):idx]

                # If we have an identifier for the secondary line
                if secondary_identifier:
                    for lookahead_idx, lookahead_line in enumerate(lookup_lines):
                        if self.line_qualifies_check(lookahead_line, secondary_identifier,
                                                     lines):  # Secondary line found

                            if strip_before_join:
                                line[
                                    'line'] = f"{line.get('line').strip()}{join_with}{lookahead_line.get('line').strip()}"
                            else:
                                line['line'] = f"{line.get('line')}{join_with}{lookahead_line.get('line')}"

                            check_after_index = idx + lookahead_idx + 1 if direction == "down" else idx

                            if not keep_line_after_moving:
                                lines[idx + lookahead_idx + 1]['line'] = ''
                            break

                # If we don't have the identifier, use index-based joining
                else:
                    if direction == "down":
                        join_idx = idx + distance
                    else:
                        join_idx = idx - distance

                    if 0 <= join_idx < len(lines):
                        join_line = lines[join_idx]

                        if strip_before_join:
                            line['line'] = f"{line.get('line').strip()}{join_with}{join_line.get('line').strip()}"
                        else:
                            line['line'] = f"{line.get('line')}{join_with}{join_line.get('line')}"

                        if not keep_line_after_moving:
                            lines[join_idx]['line'] = ''
                        check_after_index = max(check_after_index, idx + 1) if direction == "down" else idx

        updated_doc_lines = [x.get('line') for x in lines]

        return '\n'.join(updated_doc_lines)


class ExtractSection(TextManipulation):
    def __init__(self):
        super().__init__()

    def extract_short_sections(self,
                               document: str,
                               line_count: int,
                               char_count: int,
                               content=True):
        """
        Extracts short sections from a document based on line and character count criteria.

        This method scans the document, grouping lines into sections of a specified number of lines (`line_count`).
        If a section's total character count is less than the specified `char_count`, it is extracted and added
        to the list of extracted sections. Remaining lines that do not meet the criteria are retained.

        Parameters:
        document (str): The text document to process.
        line_count (int): The number of lines to group together as a section.
        char_count (int): The maximum number of characters allowed in a section for it to be extracted.
        content (bool): If True, returns the remaining text after extraction. If False, returns the extracted sections.

        Returns:
        str: The remaining text or the extracted sections, depending on the value of `content`.
        """

        lines = self.get_lines_by_document(document)
        extracted_sections = []
        remaining_text = []
        section_lines = []
        section_char_count = 0

        for i, line in enumerate(lines):
            section_lines.append(line.get('line'))
            section_char_count += len(line.get('line'))

            if len(section_lines) == line_count:
                if section_char_count < char_count:
                    extracted_section = "\n".join(section_lines)
                    extracted_sections.append(extracted_section)

                    section_lines = []
                    section_char_count = 0
                else:
                    moved_line = section_lines.pop(0)
                    remaining_text.append(moved_line)
                    section_char_count -= len(moved_line)

            if i == len(lines) - 1:
                remaining_text.extend(section_lines)

        extracted_sections_str = "\n\n".join(extracted_sections)
        remaining_text_str = "\n".join(remaining_text)

        if content:
            return remaining_text_str
        else:
            return extracted_sections_str

    def extract_between_markers(self,
                                document: str,
                                start_marker: str,
                                end_marker: str,
                                content=True):

        """
        Extracts content from a document based on specified start and end markers.

        This function identifies lines containing the start and end markers, then extracts the content between
        these lines. Note that the markers are simple strings and not regex patterns. For regex-based extraction,
        use the `extract_between_identifiers` function.

        Parameters:
        - document (str): The text document to search within.
        - start_marker (str): The string marking the beginning of the content to extract.
        - end_marker (str): The string marking the end of the content to extract.
        - content (bool): If True, returns the content found between the markers. If False, returns the document
          without the content between the markers.

        Returns:
        - str: The extracted content between the start and end markers if `content` is True, otherwise the document
          with the content between the markers removed.
        """

        lines = self.get_lines_by_document(document)
        in_extraction = False  # Flag to indicate whether currently in an extraction section
        extracted_sections = []  # List to store the extracted sections
        remaining_lines = []  # List to store lines not within the extracted sections

        current_section = []  # Temporarily stores lines of the current extracted section

        for line in lines:
            if start_marker in line.get('line'):  # Check for the start marker
                in_extraction = True
                current_section.append(line.get('line'))  # Include the start marker line in the section
                continue  # Skip adding this line to the remaining_lines

            if end_marker in line.get('line') and in_extraction:  # Check for the end marker
                current_section.append(line.get('line'))  # Include the end marker line in the section
                extracted_sections.append('\n'.join(current_section))  # Add the completed section to the list
                current_section = []  # Reset the current section for the next extraction
                in_extraction = False  # Reset the extraction flag
                continue  # Skip adding this line to the remaining_lines

            if in_extraction:
                current_section.append(line.get('line'))  # Add lines between the markers to the current section
            else:
                remaining_lines.append(line.get('line'))  # Add lines outside the markers to the remaining text

        # Combine the extracted sections and remaining lines back into strings
        extracted_text = '\n\n'.join(extracted_sections)  # Separate different sections by an empty line
        remaining_text = '\n'.join(remaining_lines)

        if content:
            return remaining_text
        else:
            return extracted_text

    def extract_between_identifiers(self,
                                    document: str,
                                    start_identifier: LineIdentifier,
                                    end_identifier: LineIdentifier,
                                    content=True):
        """
        Extracts content from a document based on specified start and end identifiers.

        This function searches the document for the given start and end identifiers and extracts the content
        that lies between them.

        Parameters:
        - document (str): The text document to search within.
        - start_identifier (str): The identifier marking the beginning of the content to extract.
        - end_identifier (str): The identifier marking the end of the content to extract.
        - content (bool): If True, returns the content found between the identifiers. If False, returns the
          document without the content between the identifiers.

        Returns:
        - str: The extracted content between the start and end identifiers if `content` is True, otherwise
          the document with the content between the identifiers removed.
        """

        lines = self.get_lines_by_document(document)
        in_extraction = False  # Flag to indicate whether currently in an extraction section
        extracted_sections = []  # List to store the extracted sections
        remaining_lines = []  # List to store lines not within the extracted sections

        current_section = []  # Temporarily stores lines of the current extracted section

        for line in lines:
            if self.line_qualifies_check(line, start_identifier, lines):
                in_extraction = True
                current_section.append(line.get('line'))  # Include the start identifier line in the section
                continue  # Skip adding this line to the remaining_lines

            if self.line_qualifies_check(line, end_identifier, lines) and in_extraction:
                current_section.append(line.get('line'))  # Include the end identifier line in the section
                extracted_sections.append('\n'.join(current_section))  # Add the completed section to the list
                current_section = []  # Reset the current section for the next extraction
                in_extraction = False  # Reset the extraction flag
                continue  # Skip adding this line to the remaining_lines

            if in_extraction:
                current_section.append(line.get('line'))  # Add lines between the markers to the current section
            else:
                remaining_lines.append(line.get('line'))  # Add lines outside the markers to the remaining text

        # Combine the extracted sections and remaining lines back into strings
        extracted_text = '\n\n'.join(extracted_sections)  # Separate different sections by an empty line
        remaining_text = '\n'.join(remaining_lines)

        if content:
            return remaining_text
        else:
            return extracted_text


class DynamicTextMarkers(TextManipulation):

    def __init__(self):
        super().__init__()

    def wrap_line_with_markers(self,
                               document: str,
                               line_identifier: LineIdentifier,
                               start_marker_lines: list[str],
                               end_marker_lines: list[str], ):

        """
        Captures a single line surrounded by start and end markers in a document.

        This function identifies a line (`primary_line_identified_content`) using `line_identifier` and wraps it
        with start and end marker lines (`start_marker_lines` and `end_marker_lines`).

        Parameters:
        - document (str): The text document where the line and markers are located.
        - line_identifier (str): The identifier marking the line to capture.
        - start_marker_lines (list): List of start marker lines to surround the identified line.
        - end_marker_lines (list): List of end marker lines to surround the identified line.

        Returns:
        - str: The updated document with the identified line surrounded by start and end markers.
        """

        lines = self.get_lines_by_document(document)
        updated_doc_list = []

        for line_obj in lines:
            line_content = line_obj.get('line')

            if self.line_qualifies_check(line_obj, line_identifier, lines):
                updated_doc_list.extend(start_marker_lines)
                updated_doc_list.append(line_content)
                updated_doc_list.extend(end_marker_lines)
            else:
                updated_doc_list.append(line_content)

        return '\n'.join(updated_doc_list)

    def add_dynamic_markers_single_line(self,
                                        document: str,
                                        line_identifier,
                                        marker_lines: list[str],
                                        location='before',
                                        ):
        """
        Adds marker lines before or after a specified identifier line in a document.

        This function is useful for inserting marker lines (`marker_lines`) either before or after a line identified
        by `line_identifier` within the `document`.

        Parameters:
        - document (str): The text document where marker lines will be added.
        - line_identifier (str): The identifier marking the line where marker lines will be inserted.
        - marker_lines (list): List of lines to insert as markers.
        - location (str): Specifies whether to insert the `marker_lines` 'before' or 'after' the `line_identifier`.

        Returns:
        - str: The updated document with marker lines inserted before or after the identified line.
        """
        lines = self.get_lines_by_document(document)
        updated_doc_list = []
        for line in lines:
            if self.line_qualifies_check(line, line_identifier, lines):
                if location == "before":
                    updated_doc_list.extend(marker_lines)
                    updated_doc_list.append(line.get('line'))
                else:
                    updated_doc_list.append(line.get('line'))
                    updated_doc_list.extend(marker_lines)
            else:
                updated_doc_list.append(line.get('line'))

        return '\n'.join(updated_doc_list)

    def add_dynamic_markers_multiline(self,
                                      document: str,
                                      start_line_identifier: LineIdentifier,
                                      end_line_identifier: LineIdentifier,
                                      start_marker_list: list[str],
                                      end_marker_list: list[str],
                                      start_marker_position='before',
                                      end_marker_position='after',
                                      max_lines_if_not_found=10,
                                      max_lookup_for_end_identifier=20) -> str:
        """
        Wraps content in a document with start and end markers based on specified identifiers and markers.

        This function identifies a start line using `start_line_identifier`, then searches for an end line using
        `end_line_identifier` within a limited number of lines (`max_lookup_for_end_identifier`). If the end line
        is not found within this limit, a maximum number of lines (`max_lines_if_not_found`) is taken as the end.

        Markers can be added to the start line (`start_marker_list`) or end line (`end_marker_list`) before or after
        the positions specified (`start_marker_position`, `end_marker_position`).

        Parameters:
        - document (str): The text document to modify.
        - start_line_identifier (str): The identifier marking the start line where content will be wrapped.
        - end_line_identifier (str): The identifier marking the end line where content wrapping ends.
        - start_marker_list (list): List of markers to add to the start line.
        - end_marker_list (list): List of markers to add to the end line.
        - start_marker_position (str): Position ('before' or 'after') to add start markers relative to the start line.
        - end_marker_position (str): Position ('before' or 'after') to add end markers relative to the end line.
        - max_lines_if_not_found (int): Maximum number of lines to take as the end if `end_line_identifier` is not found.
        - max_lookup_for_end_identifier (int): Maximum number of lines to search for `end_line_identifier`.

        Returns:
        - str: The updated document with content wrapped between start and end markers.
        """
        lines = self.get_lines_by_document(document)
        updated_doc_list = []
        skip_until_index = 0

        def add_markers(line_content, markers, position):
            if position == 'before':
                return markers + [line_content]
            else:
                return [line_content] + markers

        for idx, line_obj in enumerate(lines):
            if idx < skip_until_index:
                continue

            line_content = line_obj.get('line')
            line_index = line_obj.get('line_number') - 1

            if self.line_qualifies_check(line_obj, start_line_identifier, lines):
                section_lines = []
                end_found = False

                for lookahead_idx in range(line_index + 1,
                                           min(line_index + max_lookup_for_end_identifier + 1, len(lines))):
                    section_lines.append(lines[lookahead_idx].get('line'))

                    if self.line_qualifies_check(lines[lookahead_idx], end_line_identifier, lines):
                        end_found = True
                        skip_until_index = lookahead_idx + 1
                        break

                if not end_found:
                    section_lines = section_lines[:max_lines_if_not_found]
                    skip_until_index = line_index + max_lines_if_not_found

                updated_doc_list.extend(add_markers(line_content, start_marker_list, start_marker_position))
                updated_doc_list.extend(section_lines[:-1])
                updated_doc_list.extend(add_markers(section_lines[-1], end_marker_list, end_marker_position))

            else:
                updated_doc_list.append(line_content)

        return '\n'.join(updated_doc_list)


class FindSections(TextManipulation):
    def __init__(self):
        super().__init__()

    def extract_section_blocks(self,
                               document: str,
                               identifiers: Dict[LineIdentifier, Dict[str, Any]],
                               block_size: int = 5,
                               overlap: bool = True,
                               ):

        """
        All identifiers should fall in each block of given size.
        If the size is 5 , so it will look for presence of all identifiers within 5 lines

        identifiers = {

            idf1: {'optional': False, 'min': 1, 'max': 3},

            idf2: {'optional': False, 'min': 1, 'max': 3},

            idf3: {'optional': False, 'min': 1, 'max': 3},

            idf4: {'optional': True, 'min': 1, 'max': 1}
        }


        :param document: Document as string
        :param identifiers: List of Identifiers
        :param block_size: Lines within we have to find all the identifiers
        :param overlap: To include the last line of the previous block or not (In situations where one block starts after other)
        :return: Document containing blocks or the remaining content
        """

        def check_block_lines_for_identifiers(identifier_map: Dict[LineIdentifier, int], block_lines: list,
                                              all_lines: list):
            for line_obj in block_lines:
                for identifier, _ in identifier_map.items():
                    if identifier.evaluate(line_obj, all_lines):
                        identifier_map[identifier] += 1

            return identifier_map

        lines = self.get_lines_by_document(document)

        detected_blocks = []
        check_after_index = 0
        for idx, line in enumerate(lines):

            if idx < check_after_index:
                continue

            # Using sliding window approach
            end_window_index = idx + block_size

            window_block_lines = lines[idx: end_window_index]

            idf_map = {x: 0 for x in identifiers.keys()}

            matches_map = check_block_lines_for_identifiers(idf_map, window_block_lines, lines)

            criteria_met = True
            for idf_, cfg in identifiers.items():

                freq_of_idf = matches_map[idf_]
                is_optional = cfg.get('optional', False)  # False means that it is required

                if not is_optional:
                    if not cfg.get('min', 1) <= freq_of_idf <= cfg.get('max', float('inf')):
                        criteria_met = False
                        break
                else:
                    if not 0 <= freq_of_idf <= cfg.get('max', float('inf')):
                        criteria_met = False
                        break

            if criteria_met:
                if overlap:
                    check_after_index = end_window_index - 1
                else:
                    check_after_index = end_window_index  # Move past the current block to avoid overlap

                detected_blocks.append({
                    "line_numbers": [x.get('line_number') for x in window_block_lines],
                    "line_indexes": [x.get('line_number') - 1 for x in window_block_lines],
                    "block_content": '\n\n'.join([x.get('line') for x in window_block_lines]),
                })

        return detected_blocks

    def extract_section_lines_by_saved_metrics(self,
                                               lines: list,
                                               identifiers: Dict[LineIdentifier, Dict[str, Any]],
                                               saved_metrics: list[Dict],
                                               block_size: int = 5,
                                               overlap: bool = True, ) -> list:

        """
        All identifiers should fall in each block of given size.
        If the size is 5 , so it will look for presence of all identifiers within 5 lines

        identifiers = {

            idf1: {'optional': False, 'min': 1, 'max': 3},

            idf2: {'optional': False, 'min': 1, 'max': 3},

            idf3: {'optional': False, 'min': 1, 'max': 3},

            idf4: {'optional': True, 'min': 1, 'max': 1}
        }


        :param saved_metrics:
        :param lines: Classified lines as a list of line objects
        :param identifiers: List of Identifiers
        :param block_size: Lines within we have to find all the identifiers
        :param overlap: To include the last line of the previous block or not (In situations where one block starts after other)
        :return: Document containing blocks or the remaining content
        """

        def check_block_lines_for_identifiers(identifier_map: Dict[LineIdentifier, int], block_lines: list,
                                              all_lines: list):
            for line_obj in block_lines:
                for identifier, _ in identifier_map.items():
                    if identifier.evaluate(line_obj, all_lines, use_saved_metrics=True, saved_metrics=saved_metrics):
                        identifier_map[identifier] += 1

            return identifier_map

        detected_blocks_lines = []
        check_after_index = 0
        first_line_number = lines[0].get('line_number') - 1 if len(lines) > 0 else 0

        for idx, line in enumerate(lines, start=first_line_number):

            if idx < check_after_index:
                continue

            # Using sliding window approach
            end_window_index = idx + block_size

            window_block_lines = lines[idx: end_window_index]

            idf_map = {x: 0 for x in identifiers.keys()}

            matches_map = check_block_lines_for_identifiers(idf_map, window_block_lines, lines)

            criteria_met = True
            for idf_, cfg in identifiers.items():

                freq_of_idf = matches_map[idf_]
                is_optional = cfg.get('optional', False)  # False means that it is required

                if not is_optional:
                    if not cfg.get('min', 1) <= freq_of_idf <= cfg.get('max', float('inf')):
                        criteria_met = False
                        break
                else:
                    if not 0 <= freq_of_idf <= cfg.get('max', float('inf')):
                        criteria_met = False
                        break

            if criteria_met:
                if overlap:
                    check_after_index = end_window_index - 1
                else:
                    check_after_index = end_window_index  # Move past the current block to avoid overlap

                detected_blocks_lines.append(window_block_lines)

        return detected_blocks_lines

    def extract_lines_between_markers(self,
                                      lines: list,
                                      start_marker: str,
                                      end_marker: str,
                                      ):

        in_extraction = False  # Flag to indicate whether currently in an extraction section
        extracted_sections = []  # List to store the extracted sections

        current_section = []  # Temporarily stores lines numbers of the current extracted section

        for line in lines:
            if start_marker in line.get('line'):  # Check for the start marker
                in_extraction = True
                current_section.append(line)  # Include the start marker line number in the section
                continue

            if end_marker in line.get('line') and in_extraction:  # Check for the end marker
                current_section.append(line)  # Include the end marker line number in the section
                extracted_sections.append(current_section)  # Add the completed section to the list
                current_section = []  # Reset the current section for the next extraction
                in_extraction = False  # Reset the extraction flag
                continue

            if in_extraction:
                current_section.append(line)  # Add lines between the markers to the current section

        return extracted_sections

    def extract_lines_between_identifiers(self,
                                          lines: list,
                                          start_identifier: LineIdentifier,
                                          end_identifier: LineIdentifier,
                                          saved_metrics: list[Dict],
                                          max_lookup: int = 20,
                                          max_size: int = 20):
        """
        Extracts blocks of lines between start and end identifiers within a specified limit.

        Args:
            lines (list): List of line objects to search through.
            start_identifier (LineIdentifier): Object to evaluate the start condition.
            end_identifier (LineIdentifier): Object to evaluate the end condition.
            saved_metrics (list[Dict]): List of saved metrics for evaluation.
            max_lookup (int): Maximum number of lines to look ahead for the end identifier.
            max_size (int): Maximum number of lines in a block if the end identifier is not found.

        Returns:
            list: A list of blocks, each containing lines between the start and end identifiers.
        """

        all_blocks = []
        skip_until_index = 0

        for idx, line_obj in enumerate(lines):
            if idx < skip_until_index:
                continue

            line_content = line_obj.get('line')
            line_index = line_obj.get('line_number') - 1

            if start_identifier.evaluate(line_obj, lines, use_saved_metrics=True, saved_metrics=saved_metrics):
                section_lines = []
                end_found = False

                for lookahead_idx in range(line_index + 1,
                                           min(line_index + max_lookup + 1, len(lines))):
                    section_lines.append(lines[lookahead_idx])

                    if end_identifier.evaluate(lines[lookahead_idx], lines, use_saved_metrics=True,
                                               saved_metrics=saved_metrics):
                        end_found = True
                        skip_until_index = lookahead_idx + 1
                        break

                if not end_found:
                    section_lines = section_lines[:max_size]
                    skip_until_index = line_index + max_size

                all_blocks.append([line_obj] + section_lines)

        return all_blocks


# Line Operations Specific Classes

class Document:

    def __init__(self, lines=None):
        self.lines = lines

    def get_text(self):
        return "\n".join([x.get('line') for x in self.lines])

    def from_lines(self, lines: list[str]):
        doc = '\n'.join(lines)
        classified_lines = TextAnalyzer(text=doc).get_analysis().get('lines')
        self.lines = classified_lines


class FindOperations:
    """
    Things that can be found possibly are:

    Find Line Numbers with identifiers. Returns list of line numbers

    Find Line Numbers with regex pattern. Returns list of line numbers

    Find Line Numbers with text similarity. Returns list of line numbers

    Find Block Sections with multiple identifiers. Return list of blocks

    Find Content between Markers. Returns list of blocks

    Find Content between Identifiers. Returns list of blocks
    """

    def __init__(self, classified_lines, saved_metrics):
        self.lines = classified_lines
        self.saved_metrics = saved_metrics

    # Find Line Numbers with identifiers. Returns list of line numbers
    def find_lines_by_identifier(self, identifiers: list[LineIdentifier]):
        # Modified
        # Since only one identifier is needed here, so we will take the first one
        identifier = identifiers[0]

        line_finder = FindLines()
        line_objects = line_finder.filter_lines_by_saved_metrics(self.lines,
                                                                 identifier,
                                                                 self.saved_metrics)

        # since we got list of classified, lines we will get the line number and return them

        return [x['line_number'] for x in line_objects]

    # Find Line Numbers with regex pattern. Returns list of line numbers
    def find_lines_by_pattern(self, pattern):
        line_finder = FindLines()
        line_objects = line_finder.filter_lines_by_regex(self.lines, pattern)

        # since we got list of classified, lines we will get the line number and return them

        return [x['line_number'] for x in line_objects]

    # Find Line Numbers with text similarity. Returns list of line numbers
    def find_lines_by_similarity(self):
        raise NotImplementedError("Not important for now.")

    # Find Block Sections with multiple identifiers. Return list of blocks
    def find_blocks_by_identifier(self, identifiers: list[LineIdentifier], config: List[dict[str, Any]], size: int):
        # Modified
        section_finder = FindSections()
        # Converting to `extract_section_lines_by_saved_metrics` friendly format
        identifiers = {identifiers[i]: config[i] for i in range(len(identifiers))}

        detected_blocks = section_finder.extract_section_lines_by_saved_metrics(self.lines,
                                                                                identifiers,
                                                                                self.saved_metrics,
                                                                                size)
        # since detected blocks is List[List]. the inner list is a list of line objects of len: size
        # We only need the line numbers ,so we take first line number and last number in a block
        return [[block[0].get('line_number'), block[-1].get('line_number')] for block in detected_blocks]

    # Find Content between Markers. Returns list of blocks
    def find_blocks_between_markers(self, start_marker, end_marker):
        section_finder = FindSections()
        detected_blocks = section_finder.extract_lines_between_markers(self.lines, start_marker, end_marker)

        # since detected blocks is List[List]. the inner list is a list of line objects of len: size
        # We only need the line numbers ,so we take first line number and last number in a block
        return [[block[0].get('line_number'), block[-1].get('line_number')] for block in detected_blocks]

    # Find Content between Identifiers
    def find_blocks_between_identifiers(self,
                                        identifiers: list[LineIdentifier],
                                        max_lookup: int = 20,
                                        max_size: int = 20):
        # Modified
        start_identifier = identifiers[0]
        end_identifier = identifiers[1]

        section_finder = FindSections()
        detected_blocks = section_finder.extract_lines_between_identifiers(self.lines,
                                                                           start_identifier,
                                                                           end_identifier,
                                                                           self.saved_metrics,
                                                                           max_lookup,
                                                                           max_size)

        return [[block[0].get('line_number'), block[-1].get('line_number')] for block in detected_blocks]


class LineOperations:

    def __init__(self, lines):
        self.classified_lines = lines

    def add_line_before(self, text: str, line_numbers: List[int]):
        new_lines = []

        for num in line_numbers:
            new_lines.append({
                "line_number": num,
                "position": "before",
                "content": text,
            })

        return [{'new_lines': new_lines}]

    def add_line_after(self, text: str, line_numbers: List[int]):
        new_lines = []

        for num in line_numbers:
            new_lines.append({
                "line_number": num,
                "position": "after",
                "content": text,
            })

        return [{'new_lines': new_lines}]

    def delete_line(self, line_numbers: list[int]):

        delete_lines = []

        for num in line_numbers:
            delete_lines.append(num)

        return [{'delete_lines': delete_lines}]

    def move_lines(self, line_numbers_map: List[Dict[Literal["to", "from"], int]], seperator: str = " "):
        # Moving a line is basically  add_content_to_line operation and a delete operation

        delete_lines = []
        edit_lines = []

        for line_map in line_numbers_map:
            to_num = line_map.get('to')
            from_num = line_map.get('from')
            from_line = self.classified_lines[from_num - 1]

            delete_lines.append(from_num)
            edit_lines.append({
                "line_number": to_num,
                "add_content": f"{seperator}{from_line.get('line')}",
            })

        return [{"delete_lines": delete_lines}, {"edit_lines": edit_lines}]

    def replace_regex(self, line_numbers: list[int], pattern, replace_with):
        replace_lines = []

        for num in line_numbers:
            replace_lines.append({
                "line_number": num,
                "replace_pattern": pattern,
                "with": replace_with
            })

        return [{"edit_lines": replace_lines}]

    def replace_characters(self, line_numbers: list[int], content, replace_with):
        replace_lines = []

        for num in line_numbers:
            replace_lines.append({
                "line_number": num,
                "replace": content,
                "with": replace_with
            })

        return [{"edit_lines": replace_lines}]

    def rewrite_line(self, line_numbers: list[int], content):
        replace_lines = []

        for num in line_numbers:
            replace_lines.append({
                "line_number": num,
                "content": content,
            })

        return [{"edit_lines": replace_lines}]

    def add_content_to_line(self, line_numbers: list[int], content):
        replace_lines = []

        for num in line_numbers:
            replace_lines.append({
                "line_number": num,
                "add_content": content,
            })

        return [{"edit_lines": replace_lines}]


def save_state(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)

        # Call the `get_current_state` method to get the dictionary
        state_dict = self.get_current_state()

        # Ensure the result is a dictionary
        if not isinstance(state_dict, dict):
            raise ValueError("The return value of get_current_state must be a dictionary.")

        # Save the dictionary to a JSON file
        filename = 'state_store.json'
        with open(filename, 'w') as f:
            json.dump(state_dict, f, indent=4)

        return result

    return wrapper


def find_operation_requires_identifiers(find_obj, action: str) -> bool:
    if not hasattr(find_obj, action):
        return False

    method = getattr(find_obj, action)

    if not inspect.isfunction(method) and not inspect.ismethod(method):
        return False

    type_hints = get_type_hints(method)

    for param, hint in type_hints.items():
        if hint == list[LineIdentifier]:
            return True
    return False


class ActionProcessor(TextManipulation):

    def __init__(self, document: str):
        super().__init__()
        self.all_metrics: list = []
        self.identifiers: List[Dict[str, Any]] = []

        analyzer = TextAnalyzer(document)
        self.metadata = analyzer.get_analysis()['metadata']
        self.classified_lines = analyzer.get_analysis()['lines']

        self.rounds = []
        self.steps = []

    def index_document(self,
                       document: str,
                       ):

        lines = self.get_lines_by_document(document)

        return lines

    @save_state
    def add_and_evaluate_metric(self,
                                unique_name: str,
                                metric: Union[Metric, Dict[str, Any]],
                                ):

        # Todo for Myself: Incase a new metric is added in middle of round ,
        #  Do we need to reevaluate metrics for all rounds or no??

        if not isinstance(metric, Metric) and isinstance(metric, dict):
            metric_ = Metric()
            metric_.from_dict(metric)
        else:
            metric_ = metric

        metric_info = {
            "unique_name": unique_name,
            "metric": metric_,
            "passed_lines": []
        }

        for idx, line in enumerate(self.classified_lines):
            line_number = line.get('line_number')
            line_number = int(line_number)

            has_passed_evaluation = metric_.evaluate(line, self.classified_lines)

            if has_passed_evaluation:
                metric_info['passed_lines'].append(line_number)

        self.all_metrics.append(metric_info)

    @save_state
    def add_and_store_identifier(self,
                                 unique_name: str,
                                 identifier_comments: str,
                                 conditions: Union[List[Condition], Dict]):
        """
        Add identifiers by either creating a raw structure, or a list of condition objects

        :param unique_name: Unique name for the identifier
        :param identifier_comments: Tell what the identifier is used for
        :param conditions: conditions for the identifier
        :return:
        """

        if isinstance(conditions, Dict):
            idf = LineIdentifier()
            idf.from_dict(conditions)
        else:
            idf = LineIdentifier()
            for condition in conditions:
                idf.add_condition(condition)

        if not any(unique_name == x.get('name') for x in self.identifiers):
            self.identifiers.append({
                "name": unique_name,
                "comments": identifier_comments,
                "identifier": idf
            })
        else:
            random_name_extension = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
            unique_name = f"{unique_name}_{random_name_extension}"

            self.identifiers.append({
                "name": unique_name,
                "comments": identifier_comments,
                "identifier": idf
            })

    # Add rounds and steps
    @save_state
    def add_round(self, desc, id_):
        if not self.steps:
            return False

        self.rounds.append(
            {
                "roundId": id_,
                "description": desc,
                "steps": self.steps,
                "updated_metrics": [],
                "edits": [],
                "saves": [],
            }
        )
        return True

    def add_steps(self, operation, kwargs):
        self.steps.append({
            "action": operation,
            "kwargs": kwargs
        })

    def add_saves(self, name, comments, operation, kwargs):
        # Ask whether to save the things like kwargs, and
        if self.rounds:
            # If there is only one round, use parent lines and metrics
            if len(self.rounds) == 1:
                classified_lines = self.classified_lines
                metrics = self.all_metrics
            else:  # there should be more than one round

                # We are getting the second last round, because the doc to be edited in the current round is from the result of second last round
                classified_lines = self.rounds[-2].get('updated_lines')
                metrics = self.rounds[-2].get('updated_lines')

            find_ops = FindOperations(classified_lines, metrics)

            if getattr(find_ops, operation):
                output = getattr(find_ops, operation)(**kwargs)

                save_obj = {
                    "name": name,
                    "comments": comments,
                    "action": operation,
                    # Do we have to save the things we search,
                    # "kwargs": kwargs,

                }
                # Todo:

                # if isinstance(output, list[list])

    # Utility functions
    def get_metric_evaluations(self) \
            -> list[Dict]:
        return self.convert_metrics()

    def get_identifiers(self):

        return self.convert_identifiers()

    # Conversion functions for db/json friendly format
    def convert_identifiers(self):
        return_identifiers = []
        for idf in self.identifiers:
            return_identifiers.append({
                "name": idf['name'],
                "comments": idf['comments'],
                "identifier": idf['identifier'].to_dict()
            })

        return return_identifiers

    def convert_metrics(self):
        return_metrics = []
        for metric_info in self.all_metrics:
            return_metrics.append({
                "name": metric_info['unique_name'],
                "metric": metric_info['metric'].to_dict(),
                "passed_lines": metric_info['passed_lines']
            })
        return return_metrics

    def normalize_rounds(self):
        return_rounds = []
        for round_ in self.rounds:
            updated_metrics_after_round_processing = round_.get('updated_metrics')

            updated_metrics = []

            for metric_info in updated_metrics_after_round_processing:
                updated_metrics.append({
                    "name": metric_info['name'],
                    "metric": metric_info['metric'].to_dict(),
                    "passed_lines": metric_info['passed_lines']
                })

            return_rounds.append({

                "roundId": round_.get('roundId'),
                "description": round_.get('description'),
                "steps": round_.get('steps'),
                "updated_metrics": updated_metrics,
                "updated_lines": [x.get('line') for x in round_.get('updated_lines')] if round_.get(
                    'updated_lines') else None,
                "edits": round_.get('edits'),
                "saves": round_.get('saves'),
            })

        return return_rounds

    # Get current state of the class
    def get_current_state(self):
        return {
            "identifiers": self.convert_identifiers(),
            "metrics": self.convert_metrics(),
            "classified_lines": [x.get('line') for x in self.classified_lines],
            "metadata": self.metadata,
            "rounds": self.normalize_rounds()
        }

    # Load the stored state into the class.
    def restore_state(self, config):
        """Pending work."""
        pass

    # Processing Rounds, and edits in each round.
    def process_edits(self, round_, classified_lines):
        new_doc_lines = []
        if not round_.get('edits'):
            return classified_lines

        for line in classified_lines:
            line_number = line.get('line_number')
            line_content = line.get('line')

            line_edited = False

            for edit in round_.get('edits'):

                keys = list(edit.keys())
                edit_type = keys[0]
                edit_lines = edit[keys[0]]

                # Incase new lines need to be added
                if edit_type == "new_lines":
                    for line_ in edit_lines:
                        if line_.get('line_number') == line_number:
                            pos = line_.get('position')
                            content = line_.get('content')

                            if pos == "after":
                                # new_doc_lines.append(line_content)
                                new_doc_lines.append(content)

                            elif pos == "before":
                                new_doc_lines.append(content)
                                new_doc_lines.append(line_content)
                            line_edited = True

                # Incase lines need to be deleted
                elif edit_type == "delete_lines":
                    if line_number in edit_lines:
                        line_edited = True

                # Incase lines need to be edited
                elif edit_type == "edit_lines":
                    for line_ in edit_lines:
                        if line_.get('line_number') == line_number:
                            content = line_.get('content')
                            add_content = line_.get('add_content')
                            replace_with = line_.get('with')
                            replace_pattern = line_.get('replace_pattern')
                            replace = line_.get('replace')

                            if replace and replace_with:
                                if replace in line_content:
                                    line_content = line_content.replace(replace, replace_with)

                                new_doc_lines.append(line_content)

                            elif replace_pattern and replace_with:
                                line_content = re.sub(replace_pattern, replace_with, line)
                                new_doc_lines.append(line_content)

                            elif content:
                                new_doc_lines.append(content)

                            elif add_content:
                                new_doc_lines.append(f"{line_content}{add_content}")

                            line_edited = True

            if not line_edited:
                new_doc_lines.append(line_content)
        return new_doc_lines

    @save_state
    def process(self, repeat=False):
        for idx, round_ in enumerate(self.rounds):
            print(f"Processing round {round_.get('roundId')}: {round_.get('description')}")

            if idx == 0:
                classified_lines = self.classified_lines
                saved_metrics = self.all_metrics
            else:
                classified_lines = self.rounds[idx - 1].get('updated_lines')
                saved_metrics = self.rounds[idx - 1].get('updated_lines')

            # If we haven't applied edits to the document for current round or repeat is True
            if not (round_.get('updated_lines') and round_.get('updated_metrics')) or repeat:
                # Resolving steps from LineOperations
                for step in round_.get('steps'):
                    kwargs = step.get('kwargs')
                    lines_ops = LineOperations(classified_lines)

                    if getattr(lines_ops, step.get('action')):
                        output = getattr(lines_ops, step.get('action'))(**kwargs)
                        print("Output from edit maker", output)
                        if isinstance(output, list):
                            round_['edits'].extend(output)

                # Getting document after applying edits
                updated_doc_lines = self.process_edits(round_, classified_lines)

                # Updating classified lines

                doc_ = Document(None)
                doc_.from_lines(updated_doc_lines)

                updated_lines = doc_.lines

                # Updating metric evaluations
                updated_metrics = []
                for metric_obj in saved_metrics:
                    update_obj = {
                        "unique_name": None,
                        "passed_lines": [],
                        "name": None
                    }

                    metric_name = metric_obj.get('unique_name')
                    metric_ = metric_obj.get('metric')

                    update_obj['metric'] = metric_
                    update_obj['name'] = metric_name

                    for line_obj in updated_lines:
                        passes_metric_test = metric_.evaluate(line_obj, updated_lines)
                        if passes_metric_test:
                            update_obj['passed_lines'].append(line_obj.get('line_number'))

                    updated_metrics.append(update_obj)

                round_['updated_lines'] = updated_lines
                round_['updated_metrics'] = updated_metrics

            else:
                print(f"Round already processed, skipping to next round.")


with open('all_metrics.json') as f:
    metric_list = json.loads(f.read())


def generate_random_key():
    return uuid.uuid4().hex


def save_state_by_array(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the original function
        result = func(self, *args, **kwargs)

        # Call the `get_current_state` method to get the dictionary
        state_dict = self.get_state()

        # Save the dictionary to a JSON file
        filename = 'state_store_array.json'
        with open(filename, 'w') as f:
            json.dump(state_dict, f, indent=4)

        return result

    return wrapper


class Node:
    def __init__(self, line_number, content):
        self.line_number = line_number
        self.content = content
        self.prev = None
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.line_map = {}

    def append(self, line_number, content):
        new_node = Node(line_number, content)
        self.line_map[line_number] = new_node
        if not self.head:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def insert_before(self, line_number, content):
        if line_number not in self.line_map:
            return
        existing_node = self.line_map[line_number]
        new_node = Node(f"new_{generate_random_key()}", content)
        new_node.next = existing_node
        new_node.prev = existing_node.prev
        if existing_node.prev:
            existing_node.prev.next = new_node
        existing_node.prev = new_node
        if existing_node == self.head:
            self.head = new_node

    def insert_after(self, line_number, content):
        if line_number not in self.line_map:
            return
        existing_node = self.line_map[line_number]
        new_node = Node(f"new_{generate_random_key()}", content)
        new_node.prev = existing_node
        new_node.next = existing_node.next
        if existing_node.next:
            existing_node.next.prev = new_node
        existing_node.next = new_node
        if existing_node == self.tail:
            self.tail = new_node

    def edit_line(self, line_number, content):
        if line_number not in self.line_map:
            return
        node = self.line_map[line_number]
        node.content = content

    def get_lines(self):
        return_obj = {}
        current = self.head
        while current:
            return_obj[current.line_number] = current.content
            current = current.next

        return return_obj


class ActionProcessorJson:
    def __init__(self, document_lines):

        self.lines = self.get_lines(document_lines)
        self.metric_list = metric_list
        self.document_lines = document_lines
        self.added_metrics = []
        self.metric_results = []
        self.identifiers = []
        self.searches = []
        self.steps = []
        self.updated_document_lines = []

    def get_lines(self, doc_line_list):
        doc_ = Document()
        doc_.from_lines(doc_line_list)
        return doc_.lines

    def get_saved_metrics(self):
        saved_metrics = []
        for results in self.metric_results:
            metric_index = results.get('added_metric')
            passed_lines = results.get('lines')
            metric_dict = self.eval_incoming_metric(self.added_metrics[metric_index])

            metric_obj = Metric()
            metric_obj.from_dict(metric_dict)

            saved_metrics.append({
                "metric": metric_obj,
                "passed_lines": passed_lines
            })
        return saved_metrics

    def eval_incoming_metric(self, metric_structure):
        # Expects a structure like :
        # {
        #     "metric_index" : 0,
        #     "parameters": {
        #         "equals": "Somethings",
        #         "case_sensitive": True
        #     }
        # }
        metric_index = metric_structure["metric_index"]
        metric_name = self.metric_list[metric_index]
        parameters = metric_structure["parameters"]

        metric_dict = {
            "metric": metric_name,
            **parameters
        }

        # Returns something like this
        # {"metric": 'starts_with', 'case_sensitive': True, 'equals': "Somethings"}
        return metric_dict

    @save_state_by_array
    def handle_incoming_metric(self, metric_structure):
        dict_metric = self.eval_incoming_metric(metric_structure)

        self.added_metrics.append(metric_structure)

        total_added_metrics = len(self.added_metrics)

        metric_store_index = total_added_metrics - 1

        # Metric evaluation
        metric_obj = Metric()
        metric_obj.from_dict(dict_metric)

        passed_lines = []
        for line in self.lines:
            line_passes_test = metric_obj.evaluate(line, self.lines)

            if line_passes_test:
                passed_lines.append(line.get('line_number'))

        self.metric_results.append({
            "added_metric": metric_store_index,
            "lines": passed_lines
        })

    def eval_incoming_identifier(self, identifier_structure):
        for k, v in identifier_structure.items():
            metric_indices = v.get('metrics')
            v['metrics'] = []

            for m in metric_indices:
                v['metrics'].append(self.eval_incoming_metric(self.added_metrics[m]))

        return identifier_structure

    @save_state_by_array
    def handle_incoming_identifier(self, identifier_structure):
        self.identifiers.append(identifier_structure)

    @save_state_by_array
    def process_search(self, search_structure):
        todo_action = search_structure.get('action')
        kwargs = search_structure.get('kwargs')
        name = search_structure.get('name', None)

        find_ops = FindOperations(self.lines, self.get_saved_metrics())

        if find_operation_requires_identifiers(find_ops, todo_action):
            # Create a deep copy of kwargs to avoid modifying the original
            kwargs_copy = copy.deepcopy(kwargs)
            identifier_indices = kwargs_copy.get('identifiers', [])

            identifiers = [
                self.eval_incoming_identifier(copy.deepcopy(self.identifiers[i]))
                for i in identifier_indices
            ]

            identifiers_obj_list = []

            for idf_dict in identifiers:
                idf_obj = LineIdentifier()
                idf_obj.from_dict(idf_dict)
                identifiers_obj_list.append(idf_obj)

            kwargs_copy['identifiers'] = identifiers_obj_list

            output = getattr(find_ops, todo_action)(**kwargs_copy)
        else:
            output = getattr(find_ops, todo_action)(**kwargs)

        self.searches.append({
            "name": name,
            "action": todo_action,
            "kwargs": search_structure.get('kwargs'),
            "output": output  # Output generally will be list[int] or list[list[int, int]]
        })

    @save_state_by_array
    def process_steps(self, edit_structure):
        action = edit_structure.get('action')
        kwargs = edit_structure.get('kwargs')

        # This uses the LineOperations Class
        # The LineOperations class just returns a bunch of add, delete, edit set of instructions
        # We are not using the class now, for now just storing the steps
        self.steps.append({
            "action": action,
            "kwargs": kwargs
        })

    def process_by_edits(self, edits: list):

        doc_ = LinkedList()

        for idx, line in enumerate(self.document_lines, start=1):
            doc_.append(idx, line)

        delete_lines = []

        # Line map helps create a structure like this
        # {
        #     "1": "Content",
        #     "2": "Content",
        #     "3": "Content",
        #     "4": "Content",
        #     "5": "Content",
        #     "6": "Content",
        #     "7": "Content",
        #     "8": "Content",
        #     "9": "Content",
        #     "10": "Content"
        # }

        # If we want to create the edits, we just do this. This way we do not edit the keys. But have to keep the order of the edits in order.
        # {
        #     "1": "Content",
        #     "2": "Content",
        #     "rand_co299fna": "--added content--",
        #     "3": "Content",
        #     "4": "Content",
        #     "rand_co29r292": "--added content--",
        #     "5": "Content",
        #     "6": "Content",
        #     "7": "Content",
        #     "8": "Content",
        #     "rand_co29839": "--added content--",
        #     "9": "Content",
        #     "10": "Content"
        # }

        for edit in edits:
            keys = list(edit.keys())  # Get the type of edit and the value
            edit_type = keys[0]
            edit_lines = edit[keys[0]]

            # Incase new lines need to be added, we want to add line with random keys either above or down in order of the edit,
            if edit_type == "new_lines":
                for line in edit_lines:
                    if line['position'] == "before":
                        doc_.insert_before(line['line_number'], line['content'])
                    elif line['position'] == "after":
                        doc_.insert_after(line['line_number'], line['content'])

            # Incase lines need to be deleted
            elif edit_type == "delete_lines":
                delete_lines.extend(edit_lines)

            # Incase lines need to be edited
            elif edit_type == "edit_lines":
                for line_ in edit_lines:
                    line_number = line_.get('line_number')
                    line_content = self.document_lines[line_number - 1]
                    content = line_.get('content')
                    add_content = line_.get('add_content')
                    replace_with = line_.get('with')
                    replace_pattern = line_.get('replace_pattern')
                    replace = line_.get('replace')

                    if replace and replace_with:
                        if replace in line_content:
                            line_content = line_content.replace(replace, replace_with)
                            doc_.edit_line(line_number, line_content)

                    elif replace_pattern and replace_with:
                        line_content = re.sub(replace_pattern, replace_with, line_content)
                        doc_.edit_line(line_number, line_content)

                    elif content:
                        doc_.edit_line(line_number, content)

                    elif add_content:
                        doc_.edit_line(line_number, f"{line_content}{add_content}")

        return [val for num, val in doc_.get_lines().items() if num not in delete_lines]

    def get_snapshot(self, step_index: int):
        steps_to_process = self.steps[0: step_index + 1]
        edits = []

        for step in steps_to_process:
            kwargs = step.get('kwargs')
            lines_ops = LineOperations(self.lines)

            if getattr(lines_ops, step.get('action')):
                output = getattr(lines_ops, step.get('action'))(**kwargs)
                print("Output from edit maker", output)
                if isinstance(output, list):
                    edits.extend(output)

        output = self.process_by_edits(edits)
        with open(f'output_{step_index}.json', 'w') as f:
            f.write(json.dumps(output))
        return output

    def process(self):
        processed_lines = self.get_snapshot(len(self.steps))
        return processed_lines

    def get_state(self):
        return {
            "added_metrics": self.added_metrics,
            "metric_results": self.metric_results,
            "identifiers": self.identifiers,
            "steps": self.steps,
            "searches": self.searches,

        }

    def load_from_state(self, config):
        self.steps = config.get('steps')
        self.added_metrics = config.get('added_metrics')
        self.metric_results = config.get('metric_results')
        self.identifiers = config.get('identifiers')
        self.searches = config.get('searches')


if __name__ == "__main__":
    with open('ama_raw_text.txt', encoding='utf-8') as f:
        doc = f.read()

    doc_lines = doc.split('\n')[:500]

    processor = ActionProcessorJson(doc_lines)

    processor.handle_incoming_metric({"metric_index": 0, "parameters": {"case_sensitive": True, "equals": "Chapter"}})
    processor.handle_incoming_metric({"metric_index": 9, "parameters": {"equals": True}})

    processor.handle_incoming_metric({"metric_index": 27, "parameters": {"less_than": 1}})


    processor.handle_incoming_identifier({
        "0": {
            "type": "AND",
            "metrics": [0, 1]
        }
    })

    processor.handle_incoming_identifier({
        "0": {
            "type": "AND",
            "metrics": [2]
        }
    })

    processor.process_search({
        "action": "find_lines_by_identifier",
        "kwargs": {
            "identifiers": [0]
        }
    })

    processor.process_search({
        "action": "find_blocks_by_identifier",
        "kwargs": {
            "identifiers": [1],
            "config" : [{'optional': False, 'min': 4}],
            "size": 4
        }
    })

    processor.process_steps(
        {"action": "add_line_before", "kwargs": {"line_numbers": processor.searches[0].get('output'), "text": "--start chapter--"}})

    processor.process_steps(
        {"action": "add_line_after",
         "kwargs": {"line_numbers": processor.searches[0].get('output'), "text": "--end chapter--"}})

    processor.process_steps(
        {"action": "delete_line",
         "kwargs": {"line_numbers": [y for x in processor.searches[1].get('output') for y in range(x[0], x[1])]}})

    # processor.process_steps(
    #     {
    #         "action": 'rewrite_line',
    #         "kwargs": {"line_numbers": [y for x in processor.searches[1].get('output') for y in range(x[0], x[1])], "content": "empty_text"}
    #     }
    # )

    # processor.process_steps(
    #     {
    #         "action": 'move_lines',
    #         'kwargs': {
    #             "line_numbers_map": [{"to":1, "from": 500}]
    #         }
    #     }
    # )

    processor.process()
    processor.get_snapshot(0)
    processor.get_snapshot(1)
    processor.get_snapshot(2)

    # # print(processor.metric_passed_info)
    # processor.create_and_store_identifier('startswith The e', 'does nothing', [Condition('AND', [Metric('starts_with', {'equals': 'The e', 'case_sensitive': False})])])
    # print(processor.metric_passed_info)
    # resp = processor.identifiers[0]['identifier'].evaluate(processor.classified_lines[5457-1], processor.classified_lines, use_saved_metrics=False, saved_metrics = processor.metric_passed_info)
    # print(resp)
    # pretty_print(processor.get_current_state())

    # idf = LineIdentifier()

    # Loading Identifier from a dict (from db)
    # idf.from_dict({
    #     0: {
    #         "type": "AND",
    #         "metrics": [
    #             {"metric": 'startswith', 'equals': 'The'}
    #         ]
    #     },
    #     1: {
    #         "type": "OR",
    #         "metrics": [
    #             {"metric": 'endswith', 'equals': "dough"}
    #         ]
    #     },
    #
    # })

    # Loading identifier from raw format
    # processor.create_and_store_identifier("starter_line", "", {
    #     0: {
    #         "type": "AND",
    #         "metrics": [
    #             {"metric": 'startswith', 'equals': 'The'}
    #         ]
    #     },
    #     1: {
    #         "type": "OR",
    #         "metrics": [
    #             {"metric": 'endswith', 'equals': "dough"}
    #         ]
    #     },
    #
    # })

    # Manual Way of Loading
    # idf.add_condition(Condition(condition_type="AND", metrics=[
    #                                     Metric(name="starts_with", params={'equals': "test",'case_sensitive': True, 'strip': True}),
    #                                     Metric(name="ends_with", params={'equals': "test", 'case_sensitive': True, 'strip': True})
    #                                     ]
    #                             )
    #                   )
    # idf.ADD_AND([('starts_with', {'equals': "open", 'strip': True})])
    #
    # print(idf.to_dict())

    # resp = idf.from_dict({0: {'type': 'AND', 'metrics': [{'equals': 'test', 'case_sensitive': True, 'strip': True, 'metric': 'starts_with'},
    #                                               {'equals': 'test', 'case_sensitive': True, 'strip': True, 'metric': 'ends_with'}]},
    #                1: {'type': 'AND', 'metrics': [{'equals': 'open', 'strip': True, 'metric': 'starts_with'}]}})
    # if resp:
    #     print(idf.conditions[0].metrics[0].name,idf.conditions[0].metrics[0].params )
    #     print(idf.to_dict())
    # else:
    #     print("invalid identifier structure") # This will be printed as there are spelling mistakes in the structure.

    # keywords = ["Measurement", "Rotation", "example"]
    # analyzer = TextAnalyzer(text, keywords)
    # analysis = analyzer.get_analysis()

    # Print analysis of the first few lines for demonstration
    # for item in analysis["lines"][:1]:
    #     pretty_print(item)
    #
    # # Print metadata
    # pretty_print(analysis['metadata'], title="Metadata")

    # Print the full classification
    # pretty_print(analysis)
