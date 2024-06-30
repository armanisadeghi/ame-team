from typing import Optional, List, Dict
import re
from utils import clean_text
from automation_matrix.processing.markdown.classifier import OutputClassifier


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
