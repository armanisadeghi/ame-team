from difflib import SequenceMatcher
import re, uuid


def generate_random_key():
    return uuid.uuid4().hex

def clean_text(text: str) -> str:
    # Remove special characters and keep only printable ASCII characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return cleaned_text



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
