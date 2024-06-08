from common import pretty_print
import re
from collections import Counter
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    # Remove special characters and keep only printable ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text


# Remaining to add markdown content detection. Code is ready but need to test it thoroughly before pushing it.
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


# have to add more features
# pending features like , check_after_strip in line_qualifies_check, and case_sensitivity too

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

    def line_qualifies_check(self, line: LineClassifier, line_identifier: dict) -> bool:

        def match_conditions(line_obj, conditions):
            for condition in conditions:

                if condition.get('metric') == "startswith":
                    text_to_match = line_obj.get('line')
                    value_to_match = condition.get('equals')

                    if condition.get('strip'):
                        text_to_match = text_to_match.strip()
                    if condition.get('case_sensitive'):
                        text_to_match = text_to_match.upper()
                        value_to_match = value_to_match.upper()

                    return text_to_match.startswith(value_to_match)

                elif condition.get('metric') == 'endswith':
                    text_to_match = line_obj.get('line')
                    value_to_match = condition.get('equals')

                    if condition.get('strip'):
                        text_to_match = text_to_match.strip()
                    if condition.get('case_sensitive'):
                        text_to_match = text_to_match.upper()
                        value_to_match = value_to_match.upper()

                    return text_to_match.endswith(value_to_match)

                elif condition.get('metric') == 'has_markdown':
                    return line_obj.get('has_markdown') == condition.get('equals')

                elif condition.get('metric') == 'contains_regex_pattern':
                    text_to_match = line_obj.get('line')
                    pattern = condition.get('equals')

                    if re.search(pattern, text_to_match):
                        return True
                    return False

                elif condition.get('metric') == "contains":
                    text_to_match = line_obj.get('line')
                    value_to_match = condition.get('equals')

                    if condition.get('strip'):
                        text_to_match = text_to_match.strip()
                    if condition.get('case_sensitive'):
                        text_to_match = text_to_match.upper()
                        value_to_match = value_to_match.upper()

                    return value_to_match in text_to_match

                elif condition.get('metric') == "has_url":
                    return line_obj.get('has_urls') == condition.get('equals')

                elif condition.get('metric') == "has_emails":
                    return line_obj.get('has_emails') == condition.get('equals')

                elif condition.get('metric') == "has_date_time":
                    return line_obj.get('has_date_time') == condition.get('equals')

                elif condition.get('metric') == "starts_with_special_char":
                    return line_obj.get('starts_with_special_char') == condition.get('equals')

                elif condition.get('metric') == "ends_with_digit":
                    return line_obj.get('ends_with_digit') == condition.get('equals')

                elif condition.get('metric') == "starts_with_digit":
                    return line_obj.get('starts_with_digit') == condition.get('equals')

                elif condition.get('metric') == "ends_with_special_char":
                    return line_obj.get('ends_with_special_char') == condition.get('equals')

                elif condition.get('metric') == "indentation_level":
                    indentation_level = line_obj.get('indentation_level')

                    if condition.get('equals'):
                        return indentation_level == condition.get('equals')

                    elif condition.get('greater_than'):
                        return indentation_level > condition.get('greater_than')

                    elif condition.get('less_than'):
                        return indentation_level < condition.get('less_than')

                elif condition.get('metric') == "uppercase":
                    return line_obj.get('uppercase') == condition.get('equals')

                elif condition.get('metric') == "lowercase":
                    return line_obj.get('lowercase') == condition.get('equals')

                elif condition.get('metric') == "mixed_case":
                    return line_obj.get('mixed_case') == condition.get('equals')

                elif condition.get('metric') == "numerical_content":
                    return line_obj.get('numerical_content') == condition.get('equals')

                elif condition.get('metric') == "numeric_sum":
                    numeric_sum = line_obj.get('numeric_sum')

                    if condition.get('equals'):
                        return numeric_sum == condition.get('equals')

                    elif condition.get('greater_than'):
                        return numeric_sum > condition.get('greater_than')

                    elif condition.get('less_than'):
                        return numeric_sum < condition.get('less_than')

                elif condition.get('metric') == "punctuation_count":
                    punctuation_count = line_obj.get('punctuation_count')

                    if condition.get('equals'):
                        return punctuation_count == condition.get('equals')

                    elif condition.get('greater_than'):
                        return punctuation_count > condition.get('greater_than')

                    elif condition.get('less_than'):
                        return punctuation_count < condition.get('less_than')

                elif condition.get('metric') == "unique_words":
                    unique_words = line_obj.get('unique_words')

                    if condition.get('equals'):
                        return unique_words == condition.get('equals')

                    elif condition.get('greater_than'):
                        return unique_words > condition.get('greater_than')

                    elif condition.get('less_than'):
                        return unique_words < condition.get('less_than')

                elif condition.get('metric') == "has_hashtags":
                    return line_obj.get('has_hashtags') == condition.get('equals')

                elif condition.get('metric') == "shortest_word":
                    value_to_match = condition.get('equals')
                    text_to_match = line_obj.get('shortest_word')
                    if condition.get('case_sensitive'):
                        value_to_match = value_to_match.upper()
                        text_to_match = text_to_match.upper()
                    return text_to_match == value_to_match

                elif condition.get('metric') == "longest_word":
                    value_to_match = condition.get('equals')
                    text_to_match = line_obj.get('longest_word')
                    if condition.get('case_sensitive'):
                        value_to_match = value_to_match.upper()
                        text_to_match = text_to_match.upper()
                    return text_to_match == value_to_match

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

    def clean_document(self, document: str, replacements=None):
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

    def filter_lines(self, document: str, line_identifier: dict) -> list[LineClassifier]:

        lines = self.get_lines_by_document(document)
        filtered_lines = []
        for line in lines:
            if self.line_qualifies_check(line, line_identifier):
                filtered_lines.append(line)

        return filtered_lines

    def filter_lines_by_regex(self, document: str, pattern) -> list[LineClassifier]:
        lines = self.get_lines_by_document(document)
        results = []
        for line in lines:
            line_content = line.get('line')
            if re.search(pattern, line_content):
                results.append(line)
        return results

    def get_lines_by_document(self, document: str, doc_keywords=None):
        classifier = TextAnalyzer(document, doc_keywords)

        analysis = classifier.get_analysis()

        lines = analysis.get('lines')

        return lines

    def add_dynamic_markers_single_line(self, document: str, line_identifier: dict,
                                        start_marker_lines: list[str],
                                        end_marker_lines: list[str], ):

        """
        This function targets to capture a **single line** around start and end marker
        """

        lines = self.get_lines_by_document(document)
        updated_doc_list = []

        for line_obj in lines:
            line_content = line_obj.get('line')

            if self.line_qualifies_check(line_obj, line_identifier):
                updated_doc_list.extend(start_marker_lines)
                updated_doc_list.append(line_content)
                updated_doc_list.extend(end_marker_lines)
            else:
                updated_doc_list.append(line_content)

        return '\n'.join(updated_doc_list)

    def add_dynamic_markers_multiline(self,
                                      document: str,
                                      start_line_identifier: dict,
                                      end_line_identifier: dict,
                                      start_marker_list: list[str],
                                      end_marker_list: list[str],
                                      start_marker_position='before',
                                      end_marker_position='after',
                                      max_lines_if_not_found=10,
                                      max_lookup_for_end_identifier=10) -> str:
        """
        This function wraps multiline content with start and end markers based on specified identifiers.
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

            if self.line_qualifies_check(line_obj, start_line_identifier):
                section_lines = []
                end_found = False

                for lookahead_idx in range(line_index + 1,
                                           min(line_index + max_lookup_for_end_identifier + 1, len(lines))):
                    section_lines.append(lines[lookahead_idx].get('line'))

                    if self.line_qualifies_check(lines[lookahead_idx], end_line_identifier):
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

    def extract_between_markers(self, document: str, start_marker: str, end_marker: str, content=True):
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

    def replace_items(self, document: str, replacements: list[dict], line_identifier=None):
        """Use by_line = False, if you need to replace items in the whole document.
        Examples of replacements : [{ pattern : None , text : "Something", replacement: "Not something"},
                                    { pattern : r'\\.exe' , text : None, replacement: ".py"},
                                    { pattern : r'The US Navy has \\.\\d+' , text : "Confidential Data", replacement: "[REDACTED]"}]
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
                if self.line_qualifies_check(line, line_identifier):
                    line_content = do_replacement(replacements, line_content)

                updated_doc_list.append(line_content)

            return '\n'.join(updated_doc_list)
        else:
            return do_replacement(replacements, document)

    def extract_short_sections(self, document: str, line_count: int, char_count: int, content=True):
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

    def join_lines(self,
                   document: str,
                   primary_line_identifier,
                   distance_from_primary_identifier=None,
                   location="after",
                   remove_secondary_line=False,
                   separator=' ',
                   line_to_join_identifier=None):
        """
        :param document: The document string to process
        :param primary_line_identifier: Identifier for primary lines
        :param distance_from_primary_identifier: Distance from the primary line to join another line
        :param location: 'after' or 'before', indicating the relative position of the line to join
        :param remove_secondary_line: Boolean indicating whether to remove the secondary line after joining
        :param separator: The separator to use when joining lines
        :param line_to_join_identifier: Identifier for secondary lines to join
        :return: The formatted document as a string
        """
        updated_doc_lines = []
        check_after_index = 0

        lines = self.get_lines_by_document(document)

        for idx, line in enumerate(lines):
            if idx < check_after_index:
                continue

            if self.line_qualifies_check(line, primary_line_identifier):  # Detected a primary line

                if location == "after":
                    lookup_lines = lines[idx + 1:]
                else:
                    lookup_lines = lines[:idx]

                # If we have an identifier for the secondary line
                if line_to_join_identifier:
                    for lookahead_idx, lookahead_line in enumerate(lookup_lines):
                        if self.line_qualifies_check(lookahead_line, line_to_join_identifier):  # Secondary line found
                            updated_doc_lines.append(f"{line.get('line')}{separator}{lookahead_line.get('line')}")
                            check_after_index = idx + lookahead_idx + 1 if location == "after" else idx
                            if remove_secondary_line:
                                del lines[idx + lookahead_idx + 1]
                            break
                # If we don't have the identifier, use index-based joining
                else:
                    if location == "after":
                        join_idx = idx + distance_from_primary_identifier + 1
                    else:
                        join_idx = idx - distance_from_primary_identifier

                    if 0 <= join_idx < len(lines):
                        join_line = lines[join_idx]
                        updated_doc_lines.append(f"{line.get('line')}{separator}{join_line.get('line')}")
                        if remove_secondary_line:
                            del lines[join_idx]
                        check_after_index = max(check_after_index, idx + 1) if location == "after" else idx
                    else:
                        updated_doc_lines.append(line.get('line'))
            else:
                updated_doc_lines.append(line.get('line'))

        return '\n'.join(updated_doc_lines)


def process_steps(cleaning_steps, document):
    processor = TextManipulation()
    updated_document = document

    for step_info in cleaning_steps:
        step_name = step_info.get("step")
        kwargs = step_info.get("kwargs", {})

        # Class called `TextManipulation` with methods matching the step names
        if hasattr(processor, step_name):
            # Get the method from the class
            method = getattr(processor, step_name)

            # Call the method with the provided keyword arguments
            updated_document = method(document=updated_document, **kwargs)

    return updated_document


if __name__ == "__main__":
    # Need to resolve encoding error with ama_raw_text.txt
    with open(r'chapter_text_19.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    obj = TextManipulation()

    identifier = {
        "AND": [{"metric": "starts_with_digit", "equals": True}]
    }
    identifier2 = {
        "AND": [{"metric": "contains", "equals": "abduction, adduction, elevation, depression"}]
    }

    steps = [
        {"step": "clean_document", "kwargs": {}},
        {"step": "limit_consecutive_empty_lines", "kwargs": {'max_empty_lines': 0}},
    ]

    updated_document = process_steps(steps, document=text)

    print(updated_document)

    # updated_doc = obj.add_dynamic_markers_multiline(text, identifier, identifier2, ['---start---'], ['---end---'], )
    # updated_doc = obj.extract_between_markers(updated_doc, '---start---', '---end---', content=False)
    # updated_doc = obj.replace_items(updated_doc, [{'pattern': r'---(\w+)---', 'replacement':'[REDACTED]', 'text': 'Record movements'}])
    # updated_doc = obj.join_lines(text, identifier, None, line_to_join_identifier=identifier2,
    #                                         remove_secondary_line=True)

    # print(updated_doc)

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
