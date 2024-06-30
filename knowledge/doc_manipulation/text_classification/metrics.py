import re
from knowledge.doc_manipulation.text_classification.utils import TextSimilarity


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
