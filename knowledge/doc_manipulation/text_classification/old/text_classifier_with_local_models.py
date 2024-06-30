"""
**Note**
File by Armani as is. Of no use as of now
"""
import re
from collections import Counter
from typing import List, Optional
from textblob import TextBlob
import spacy
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation


class LineClassifier:
    def __init__(self, line: str, previous_line: Optional[str] = None, next_line: Optional[str] = None):
        self.line = line
        self.previous_line = previous_line
        self.next_line = next_line

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

        # Load models for advanced analysis - Requires model download
        # self.nlp = spacy.load("en_core_web_sm")

        # With added pretrained models (easy to find)
        # self.vectorizer = CountVectorizer()
        # self.lda_model = LatentDirichletAllocation()

    def count_sentences(self) -> int:
        return sum(self.line.count(p) for p in '.!?')

    def sum_numerical_values(self) -> int:
        return sum(int(num) for num in re.findall(r'\d+', self.line))

    def count_punctuation(self) -> int:
        return sum(1 for char in self.line if char in '.,;:!?')

    # Text match methods
    def contains(self, substring: str) -> bool:
        return substring in self.line

    def starts_with(self, substring: str) -> bool:
        return self.line.startswith(substring)

    def ends_with(self, substring: str) -> bool:
        return self.line.endswith(substring)

    def not_contains(self, substring: str) -> bool:
        return substring not in self.line

    def not_starts_with(self, substring: str) -> bool:
        return not self.line.startswith(substring)

    def not_ends_with(self, substring: str) -> bool:
        return not self.line.endswith(substring)

    # Advanced analysis methods
    def sentiment_analysis(self) -> str:
        analysis = TextBlob(self.line)
        return "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"

    def topic_classification(self) -> List[str]:
        transformed_line = self.vectorizer.transform([self.line])
        topic_distribution = self.lda_model.transform(transformed_line)
        topics = [f"Topic {i}" for i in topic_distribution.argmax(axis=1)]
        return topics

    def entity_recognition(self) -> List[str]:
        doc = self.nlp(self.line)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def __str__(self):
        return f"Line: '{self.line}'\n" \
               f"Character Count: {self.char_count}\n" \
               f"Word Count: {self.word_count}\n" \
               f"Sentence Count: {self.sentence_count}\n" \
               f"Line Length: {self.line_length}\n" \
               f"Starts with Digit: {self.starts_with_digit}\n" \
               f"Ends with Digit: {self.ends_with_digit}\n" \
               f"Starts with Special Char: {self.starts_with_special_char}\n" \
               f"Ends with Special Char: {self.ends_with_special_char}\n" \
               f"Indentation Level: {self.indentation_level}\n" \
               f"Uppercase: {self.uppercase}\n" \
               f"Lowercase: {self.lowercase}\n" \
               f"Mixed Case: {self.mixed_case}\n" \
               f"Numerical Content: {self.numerical_content}\n" \
               f"Numeric Sum: {self.numeric_sum}\n" \
               f"Punctuation Count: {self.punctuation_count}\n" \
               f"Unique Words: {self.unique_words}\n" \
               f"Sentiment: {self.sentiment_analysis()}\n" \
               f"Topics: {', '.join(self.topic_classification())}\n" \
               f"Entities: {', '.join([f'{text} ({label})' for text, label in self.entity_recognition()])}\n" \
               f"Previous Line: {self.previous_line}\n" \
               f"Next Line: {self.next_line}\n"


# This one doesn't work, but it was an initial sample to show how we can very easily add local super-fast, specific ai-models that act almost like a function and can provide really great insights.
# Because they're so specialized, it's very easy to get specific data that matters to us.
# I assume there are plenty that can help clean up junk from text for sure, especially if the classifier is feeding them the right amount of text at one time.

line = "This is a test line with 3 sentences. It's for classification! And it ends here."
classifier = LineClassifier(line, previous_line="Previous line.", next_line="Next line.")
print(classifier)
