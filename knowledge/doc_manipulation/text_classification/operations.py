from identifiers import LineIdentifier
from sections import FindSections, FindLines
from typing import Any, List, Dict, Literal


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

