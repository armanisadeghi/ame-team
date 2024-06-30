# The code here is not functional, just keeping it, will be removed later



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

class TextManipulation:

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

