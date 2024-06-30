from identifiers import LineIdentifier
from typing import List, Dict, Any
import re


class FindLines:
    """
    This class is used for filtering and finding Line Objects by these ways:
     - Filtering By LineIdentifier
     - Filtering By Regex Pattern
    """

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

    def filter_lines_by_regex(self,
                              lines,
                              pattern) -> list[Dict]:
        results = []

        for line in lines:
            line_content = line.get('line')
            if re.search(pattern, line_content):
                results.append(line)
        return results


class FindSections:

    def extract_section_blocks(self, # Old code no more used
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

