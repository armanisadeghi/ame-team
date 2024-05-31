import re
import json
# from common.utils.my_utils import pretty_print_data
# from knowledgebase.utils.add_data import DataManager

def remove_references_original(file_path):
    def is_reference_start(line):
        return line.strip().lower().startswith("references")

    def is_reference_item_start(line):
        stripped_line = line.strip()
        return stripped_line and stripped_line[0] == "1" and stripped_line[1:2] in ["", ".", ")", " "]

    def get_next_expected_number(line, expected):
        stripped_line = line.strip()
        if stripped_line and stripped_line[0].isdigit():
            num_end_index = 1
            while num_end_index < len(stripped_line) and stripped_line[num_end_index].isdigit():
                num_end_index += 1
            current_number = int(stripped_line[:num_end_index])
            if expected - 2 <= current_number <= expected + 2:
                return current_number + 1
        return expected

    def process_file():
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        in_reference_section = False
        reference_sections = []
        current_section = []
        expected_number = 1
        non_matching_lines = 0

        for line in lines:
            if not in_reference_section and is_reference_start(line):
                in_reference_section = True
                current_section = [line]
            elif in_reference_section:
                if is_reference_item_start(line):
                    expected_number = 2  # After finding the first item, expect 2 next
                    current_section.append(line)
                    non_matching_lines = 0
                elif non_matching_lines < 8:
                    next_expected = get_next_expected_number(line, expected_number)
                    if next_expected != expected_number:
                        expected_number = next_expected
                        non_matching_lines = 0
                    else:
                        non_matching_lines += 1
                    current_section.append(line)
                else:
                    # End of the reference section
                    reference_sections.append(current_section)
                    in_reference_section = False
                    expected_number = 1
                    non_matching_lines = 0

        # Remove reference sections from the original text and save them
        for section in reference_sections:
            for line in section:
                lines.remove(line)

        # Write the cleaned text back
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        # Write the reference sections to a separate file
        with open('references.txt', 'w', encoding='utf-8') as ref_file:
            for section in reference_sections:
                ref_file.writelines(section)
                ref_file.write('\n\n')

    process_file()


def normalize_chapter_headers(document):
    lines = document.split('\n')
    new_lines = []
    i = 0

    def is_chapter_line(line):
        return line.strip().lower().startswith("chapter ") and line.strip()[-1].isdigit()

    while i < len(lines):
        line = lines[i]
        if is_chapter_line(line):
            new_lines.append('\n')  # Assuming the 2 blank lines before are already in place
            new_lines.append('\n')
            new_lines.append('=' * 80 + '\n')
            new_lines.append(line + '\n')
            new_lines.append('-' * 65 + '\n')
            new_lines.append('\n')

            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue  # Skip adding the current line again
        new_lines.append(line + '\n')
        i += 1

    updated_document = ''.join(new_lines)
    return updated_document


def clean_blank_lines(document, max_blank_lines):
    lines = document.split('\n')
    cleaned_lines = []
    blank_count = 0

    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count > max_blank_lines:
                continue
        else:
            blank_count = 0
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def extract_references_and_update_document(document):
    lines = document.split('\n')
    all_references = []
    updated_lines = lines[:]

    while True:
        references_start_index = None
        next_chapter_start_index = None

        # Search for the start of the next references section
        for i, line in enumerate(updated_lines):
            if line.strip().lower() == "references":
                references_start_index = i
                break

        # If "References" was found, search for the start of the next chapter
        if references_start_index is not None:
            for i in range(references_start_index + 1, len(updated_lines)):
                if updated_lines[i].startswith('====='):
                    next_chapter_start_index = i
                    break

        # If both "References" and the next chapter start were found, extract and update
        if references_start_index is not None and next_chapter_start_index is not None:
            references = '\n'.join(updated_lines[references_start_index:next_chapter_start_index])
            all_references.append(references)
            # Update the document by removing the references section and ensuring two blank lines
            updated_lines = updated_lines[:references_start_index] + ['\n', '\n'] + updated_lines[next_chapter_start_index:]
        else:
            break  # Exit loop if no more references sections are found

    # Concatenate all references sections separated by two newlines
    concatenated_references = '\n\n'.join(all_references)
    updated_document = '\n'.join(updated_lines)

    return concatenated_references, updated_document


def organize_chapter_sections(document):
    lines = document.split('\n')
    organized_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith('====='):  # Start of a new chapter section
            chapter_line = lines[i + 1]  # Assuming the next line is the chapter line
            main_title = lines[i + 4].strip()  # Assuming the main title is 2 lines down from the chapter line

            # Append the reformatted chapter header
            organized_lines.append(line)  # Keep the "=====" line
            organized_lines.append(f'{chapter_line}: {main_title}')  # Combine chapter line with main title
            organized_lines.append('-' * 65)  # Keep the "-----" line

            # Skip lines to the next section or introduction
            i += 4
            while i < len(lines) and not lines[i].lower().startswith('introduction'):
                section_line = lines[i].strip()
                if section_line:  # Ignore blank lines
                    organized_lines.append(section_line)
                i += 1

            # Append the "Introduction" section and following content without changes
            while i < len(lines) and not lines[i].startswith('====='):  # Until the start of the next chapter
                organized_lines.append(lines[i])
                i += 1
            continue
        else:
            organized_lines.append(line)
        i += 1

    # Join the organized lines to form the updated document
    return '\n'.join(organized_lines)


def add_marker_after_comments(text):
    lines = text.split('\n')
    modified_lines = []

    for line in lines:
        modified_lines.append(line)
        if line.startswith('Comment:'):
            modified_lines.append('---END EXAMPLE---')

    modified_text = '\n'.join(modified_lines)
    return modified_text


def add_marker_before_examples(text):
    lines = text.split('\n')
    modified_lines = []

    for line in lines:
        if line.startswith('Example'):
            modified_lines.append('---START EXAMPLE---')
        modified_lines.append(line)
    modified_text = '\n'.join(modified_lines)

    return modified_text


def add_missing_end_markers(text):
    lines = text.split("\n")  # Split text into lines
    modified_lines = []  # List to store modified lines
    start_index = None  # Keep track of the index of the first "---START EXAMPLE---"

    for i, line in enumerate(lines):
        # Check for "---START EXAMPLE---"
        if "---START EXAMPLE---" in line:
            if start_index is None:  # If it's the first start marker, remember its index
                start_index = i
            else:
                # If a second start marker is found without an end marker within 20 lines, insert the end marker
                if i - start_index <= 20:
                    modified_lines.append("---END EXAMPLE---")
                start_index = i  # Update start_index for the new start marker

        modified_lines.append(line)  # Add the current line to the modified lines

        # If the end marker is found, reset the start_index
        if "---END EXAMPLE---" in line and start_index is not None:
            start_index = None

    # If the text ends with a start marker without a corresponding end marker, add the end marker at the end
    if start_index is not None and len(lines) - start_index <= 20:
        modified_lines.append("---END EXAMPLE---")

    return "\n".join(modified_lines)  # Join the modified lines back into a single string


def fix_examples(text):
    lines = text.split('\n')
    modified_lines = []
    i = 0
    within_example = False
    end_example_pending = False

    while i < len(lines):
        line = lines[i]
        if line == '---START EXAMPLE---':
            if end_example_pending:
                modified_lines.append('---END EXAMPLE---')
                end_example_pending = False
            modified_lines.append(line)
            within_example = True
        elif line == '---END EXAMPLE---':
            if within_example:
                end_example_pending = True
            else:
                modified_lines.append(line)
        else:
            if end_example_pending and (line.strip() == '' or line.startswith('---START EXAMPLE---')):
                modified_lines.append('---END EXAMPLE---')
                end_example_pending = False
                if line.strip() == '':
                    i += 1
                    continue
            modified_lines.append(line)
            within_example = line.strip() != ''
        i += 1
    if end_example_pending:
        modified_lines.append('---END EXAMPLE---')

    return '\n'.join(modified_lines)


def mark_examples(text):
    updated_text = add_marker_after_comments(text)
    updated_text = add_marker_before_examples(updated_text)
    updated_text = add_missing_end_markers(updated_text)
    updated_text = fix_examples(updated_text)
    return updated_text


def extract_and_remove_examples(text, start_marker='---START EXAMPLE---', end_marker='---END EXAMPLE---', max_lines=20):
    lines = text.split("\n")
    remaining_text = []
    examples_text = []
    skip_lines = 0
    current_example = []

    for i, line in enumerate(lines):
        if skip_lines > 0:
            skip_lines -= 1
            current_example.append(line)
            if end_marker in line:
                examples_text.append("\n".join(current_example))
                current_example = []
                skip_lines = 0
            continue

        if start_marker in line:
            current_example = [line]
            skip_lines = max_lines
        else:
            remaining_text.append(line)

    remaining_text_str = "\n".join(remaining_text)
    examples_text_str = "\n".join(examples_text)

    remaining_text_str = remove_end_example_markers(remaining_text_str)
    return remaining_text_str, examples_text_str


def extract_short_sections(text, line_count, char_count):
    lines = text.split("\n")
    extracted_sections = []
    remaining_text = []
    section_lines = []
    section_char_count = 0

    for i, line in enumerate(lines):
        section_lines.append(line)
        section_char_count += len(line)

        if len(section_lines) == line_count:
            if section_char_count < char_count:
                extracted_section = "\n".join(section_lines)
                extracted_sections.append(extracted_section)
                # print(f"--------------------\n{extracted_section}\n--------------------")
                section_lines = []
                section_char_count = 0
            else:
                moved_line = section_lines.pop(0)
                remaining_text.append(moved_line)
                section_char_count -= len(moved_line)
                # print(f"Moved line to remaining text: {moved_line}")

        if i == len(lines) - 1:
            remaining_text.extend(section_lines)

    extracted_sections_str = "\n\n".join(extracted_sections)
    remaining_text_str = "\n".join(remaining_text)

    # print(f"---\nExtracted Sections:\n{extracted_sections_str}")

    return extracted_sections_str, remaining_text_str


def remove_end_example_markers(text):
    lines = text.split("\n")  # Split the text into lines
    cleaned_lines = [line for line in lines if "---END EXAMPLE---" not in line]
    return "\n".join(cleaned_lines)  # Join the remaining lines back into a single string


def eliminate_extra_empty_lines(text, max_consecutive_empty_lines=2):
    lines = text.split("\n")  # Split the text into lines
    cleaned_lines = []  # List to store the cleaned lines
    consecutive_empty_lines = 0  # Counter for consecutive empty lines

    for line in lines:
        if line.strip() == "":  # Check if the line is empty
            consecutive_empty_lines += 1  # Increment the counter for empty lines
        else:
            consecutive_empty_lines = 0  # Reset the counter if a non-empty line is found

        # Keep the line if we haven't exceeded the max number of consecutive empty lines
        if consecutive_empty_lines <= max_consecutive_empty_lines:
            cleaned_lines.append(line)
        # If there are more consecutive empty lines than allowed, this line is skipped

    return "\n".join(cleaned_lines)  # Join the cleaned lines back into a single string


def document_cleanup_orchestrator(document):
    # Step 1: remove excessive blank lines
    cleaned_document = clean_blank_lines(document, 2)
    updated_document = normalize_chapter_headers(cleaned_document)
    references, updated_document = extract_references_and_update_document(updated_document)
    updated_document = organize_chapter_sections(updated_document)

    # Step 5: Wrap and clean examples
    updated_document = mark_examples(updated_document)

    # Step 6: Extract and remove known examples
    updated_document, examples = extract_and_remove_examples(updated_document)

    line_count = 25
    char_count = 100
    low_char_count_sections, updated_document = extract_short_sections(updated_document, line_count, char_count)

    return updated_document, references, examples, low_char_count_sections


def add_dynamic_marker(text, line_start_text, location, text_to_add):
    lines = text.split('\n')
    modified_lines = []
    replacement_count = 0

    for line in lines:
        if line.startswith(line_start_text):
            if location == 'before':
                modified_lines.append(text_to_add)
                modified_lines.append(line)
                replacement_count += 1
            elif location == 'after':
                modified_lines.append(line)
                modified_lines.append(text_to_add)
                replacement_count += 1
        else:
            modified_lines.append(line)

    modified_text = '\n'.join(modified_lines)
    print(f"Total replacements made: {replacement_count}")
    return modified_text


def extract_between_markers(text, start_marker, end_marker):
    lines = text.split('\n')
    in_extraction = False  # Flag to indicate whether currently in an extraction section
    extracted_sections = []  # List to store the extracted sections
    remaining_lines = []  # List to store lines not within the extracted sections

    current_section = []  # Temporarily stores lines of the current extracted section

    for line in lines:
        if start_marker in line:  # Check for the start marker
            in_extraction = True
            current_section.append(line)  # Include the start marker line in the section
            continue  # Skip adding this line to the remaining_lines

        if end_marker in line and in_extraction:  # Check for the end marker
            current_section.append(line)  # Include the end marker line in the section
            extracted_sections.append('\n'.join(current_section))  # Add the completed section to the list
            current_section = []  # Reset the current section for the next extraction
            in_extraction = False  # Reset the extraction flag
            continue  # Skip adding this line to the remaining_lines

        if in_extraction:
            current_section.append(line)  # Add lines between the markers to the current section
        else:
            remaining_lines.append(line)  # Add lines outside the markers to the remaining text

    # Combine the extracted sections and remaining lines back into strings
    extracted_text = '\n\n'.join(extracted_sections)  # Separate different sections by an empty line
    remaining_text = '\n'.join(remaining_lines)

    return extracted_text, remaining_text


def handle_text_starting_with_tab_indent(text):
    lines = text.split('\n')
    result_lines = []

    for i in range(len(lines)):
        if lines[i].startswith('\t'):
            if result_lines:
                result_lines[-1] += ' ' + lines[i].lstrip('\t')
        else:
            result_lines.append(lines[i])

    return '\n'.join(result_lines)


def replace_tabs_with_newlines(text):
    lines = text.split('\n')
    result_lines = []

    for line in lines:
        tab_count = len(line) - len(line.lstrip('\t'))
        if tab_count > 0:
            print(f"Line: {line}, Tab count: {tab_count}")
        stripped_line = line.lstrip('\t')
        if tab_count > 0:
            result_lines.extend(['' for _ in range(tab_count)])
            result_lines.append(stripped_line)
        else:
            result_lines.append(stripped_line)

    return '\n'.join(result_lines)


def replace_multiple_spaces_with_newlines(text, spaces_per_tab=4):
    # Define a tab pattern that can be a tab character or a specified number of spaces
    tab_pattern = '\t' + '|'.join([' ' * i for i in range(spaces_per_tab, 0, -1)])

    result_lines = []  # Initialize an empty list to hold the processed lines
    lines = text.split('\n')  # Split the text into lines

    for line in lines:
        # Initialize a count for additional new lines needed
        new_lines_needed = 0

        # While the line starts with a tab or the equivalent spaces, count and strip them
        while re.match(tab_pattern, line):
            if line.startswith('\t'):
                print(f"Line: {line}, Tab count: 1")
                line = line[1:]  # Strip one tab character
            else:
                line = line[spaces_per_tab:]  # Strip the equivalent number of spaces
            new_lines_needed += 1

        # Add the required new lines and the stripped line to the result
        result_lines.extend(['' for _ in range(new_lines_needed)])
        result_lines.append(line)

    # Join the processed lines back into a single string
    return '\n'.join(result_lines)


def analyze_tabs(line):
    tab_count = line.count('\t')
    if tab_count > 0:
        print(f"Found {tab_count} tab character(s)")

    space_sequences = re.findall(' {2,}', line)
    for seq in space_sequences:
        print(f"Found a sequence of {len(seq)} spaces")


def extract_and_remove_text_sections(
        start_marker,
        end_marker,
        max_lines,
        original_file_path,
        remaining_file_path=None,
        extracted_text_file_path=None):
    def read_text_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def write_text_to_file(file_path, text):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)

    def add_text_to_file(file_path, text):
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write('\n' + text)

    if not remaining_file_path:
        parts = original_file_path.rsplit('.', 1)
        if len(parts) == 2:
            remaining_file_path = f"{parts[0]}_updated.{parts[1]}"
        else:
            remaining_file_path = f"{original_file_path}_updated"

    def remove_extra_end_markers(text):
        lines = text.split("\n")
        cleaned_lines = [line for line in lines if end_marker not in line]
        print(f"Debug: remove_extra_end_markers would have removed {len(lines) - len(cleaned_lines)} lines containing '{end_marker}'")
        # return "\n".join(cleaned_lines)
        return text

    text = read_text_from_file(original_file_path)

    lines = text.split("\n")
    remaining_text = []
    matching_text = []
    skip_lines = 0
    current_example = []
    total_replacements = 0

    for i, line in enumerate(lines, start=1):
        if skip_lines > 0:
            skip_lines -= 1
            current_example.append(line)
            if end_marker in line:
                matching_text.append("\n".join(current_example))
                print(f"Extracted text ending at line number {i}")
                current_example = []
                skip_lines = 0
                total_replacements += 1
            continue

        if start_marker in line:
            current_example = [line]
            print(f"Starting extraction from line number {i}")
            skip_lines = max_lines
        else:
            remaining_text.append(line)

    remaining_text_str = "\n".join(remaining_text)
    examples_text_str = "\n".join(matching_text)

    # I don't think we need this, but keeping it for now
    remaining_text_str = remove_extra_end_markers(remaining_text_str)

    write_text_to_file(remaining_file_path, remaining_text_str)
    add_text_to_file(extracted_text_file_path, examples_text_str)

    print(f"Total number of replacements: {total_replacements}")

    return


def convert_tabs_to_newlines(input_string):
    return input_string.replace('\t', '\n')


def replace_text_pattern(text, old_pattern, new_pattern):
    updated_text = text.replace(old_pattern, new_pattern)
    return updated_text


def delete_lines_starting_with_pattern(text, pattern):
    lines = text.split('\n')
    remaining_lines = []
    deletion_count = 0

    for line in lines:
        if line.startswith(pattern):
            print(f"Deleting line: {line}")
            deletion_count += 1
        else:
            remaining_lines.append(line)

    updated_text = '\n'.join(remaining_lines)
    print(f"Total deletions: {deletion_count}")
    return updated_text


def split_text_by_marker(file_path, marker_text, new_file_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    parts = content.split(marker_text)
    part_count = 0

    for i in range(1, len(parts)):
        part = parts[i].strip()
        if part:
            new_file_path = f"{new_file_name}_{part_count + 1:02d}.txt"
            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                new_content = marker_text.strip() + "\n\n" + part
                content_with_markers = insert_character_count_markers_text(new_content)
                new_file.write(content_with_markers)
                part_count += 1
                print(f"Created: {new_file_path} - Character count: {len(content_with_markers)}")

    print(f"Total documents created: {part_count}")


def remove_pattern_and_numbers(text):
    lines = text.split('\n')
    lines_to_remove = set()
    removed_texts = []

    def is_numbers_and_spaces(line):
        return len(line) < 7 and all(c.isdigit() or c.isspace() for c in line)

    for i, line in enumerate(lines):
        if "Guides to the Evaluation of Permanent Impairment" in line[:55]:
            lines_to_remove.add(i)
            removed_texts.append(line)

            for j in range(max(0, i - 2), i):
                if j not in lines_to_remove and is_numbers_and_spaces(lines[j]):
                    lines_to_remove.add(j)
                    removed_texts.append(lines[j])

            for j in range(i + 1, min(len(lines), i + 3)):
                if j not in lines_to_remove and is_numbers_and_spaces(lines[j]):
                    lines_to_remove.add(j)
                    removed_texts.append(lines[j])

    new_text = '\n'.join(line for i, line in enumerate(lines) if i not in lines_to_remove)

    for removed_text in removed_texts:
        print(f"Removed: {removed_text}")
    print(f"Total instances removed: {len(removed_texts)}")

    return new_text


def insert_breaks(input_file_path, output_file_path):
    chapter_pattern = re.compile(r'^\d+(\.\d+)*[a-d]?')
    ignore_start_pattern = re.compile(r'================================================================================')  # At least one equals sign, but only equals signs
    ignore_end_pattern = re.compile(r'-------INTRODUCTION')  # At least one dash, but only dashes
    ignore_section = False  # Flag to keep track of ignored sections

    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            if ignore_start_pattern.match(line.strip()):  # Use strip() to remove leading/trailing whitespace
                ignore_section = True

            if ignore_section:
                output_file.write(line)
                if ignore_end_pattern.match(line.strip()):  # Use strip() to remove leading/trailing whitespace
                    ignore_section = False
            else:
                stripped_line = line.lstrip()
                if chapter_pattern.match(stripped_line):
                    output_file.write("--break--\n")
                output_file.write(line)


def insert_character_count_markers(input_file_path, output_file_path):
    break_marker = "--break--"
    accumulated_text = ''
    char_count_since_last_marker = 0
    total_marker_count = 0
    char_marker_count = 0
    longest_section_length = 0

    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            accumulated_text += line
            char_count_since_last_marker += len(line)

            if break_marker in line:
                total_marker_count += 1
                if char_count_since_last_marker >= 3000:
                    accumulated_text = accumulated_text.replace(break_marker, f"-- chars {char_count_since_last_marker} --\n{break_marker}")
                    total_marker_count += 1
                    char_count_since_last_marker = 0
                output_file.write(accumulated_text)
                longest_section_length = max(longest_section_length, len(accumulated_text))
                accumulated_text = ''
            elif char_count_since_last_marker >= 3000:
                accumulated_text += f"-- chars {char_count_since_last_marker} --\n"
                char_count_since_last_marker = 0
                char_marker_count += 1
                total_marker_count += 1
                output_file.write(accumulated_text)
                longest_section_length = max(longest_section_length, len(accumulated_text))
                accumulated_text = ''

        if accumulated_text:
            output_file.write(accumulated_text)
            longest_section_length = max(longest_section_length, len(accumulated_text))

    print(f"Total number of markers found: {total_marker_count}")
    print(f"Total number of character count markers added: {char_marker_count}")
    print(f"Length of the longest section: {longest_section_length}")


def insert_character_count_markers_text(text, max_chars=3000):
    break_marker = "--break--"
    accumulated_text = ''
    updated_text = ''
    char_count_since_last_marker = 0
    total_marker_count = 0
    char_marker_count = 0
    longest_section_length = 0
    input_lines = text.split('\n')

    for line in input_lines:
        if accumulated_text:
            accumulated_text += '\n'
        accumulated_text += line
        char_count_since_last_marker += len(line) + 1

        if break_marker in line:
            total_marker_count += 1
            if char_count_since_last_marker >= max_chars:
                marker_position = accumulated_text.rfind(break_marker)
                accumulated_text = accumulated_text[:marker_position] + f"-- chars {char_count_since_last_marker} --\n" + accumulated_text[marker_position:]
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

    updated_text = updated_text.replace(break_marker, '')

    print(f"Total number of markers found: {total_marker_count}")
    print(f"Total number of character count markers added: {char_marker_count}")
    print(f"Length of the longest section: {longest_section_length}")

    return updated_text


def text_to_dict(text, max_chars=3000, test_strings=[]):
    break_marker = "--break--"
    sections = []  # List to hold each section of text along with their char counts
    section_dict = {}  # Dictionary to return
    accumulated_text = ''
    char_count_since_last_marker = 0
    input_lines = text.split('\n')

    # Function to remove lines containing any of the test strings
    def filter_test_strings(lines, test_strings):
        return [line for line in lines if all(test_string not in line for test_string in test_strings)]

    input_lines = filter_test_strings(input_lines, test_strings)

    for line in input_lines:
        if accumulated_text:
            accumulated_text += '\n'
        accumulated_text += line
        char_count_since_last_marker += len(line) + 1  # Including the newline character

        if break_marker in line:
            accumulated_text = accumulated_text.replace(break_marker, '')  # Remove the break marker
            if char_count_since_last_marker >= max_chars:
                # Section reaches the character limit at the break marker
                sections.append((accumulated_text, char_count_since_last_marker))  # Store accumulated text and its char count
                accumulated_text = ''  # Reset accumulated text
                char_count_since_last_marker = 0  # Reset character count
        elif char_count_since_last_marker >= max_chars:
            # Section reaches the character limit without a break marker
            sections.append((accumulated_text, char_count_since_last_marker))  # Store accumulated text and its char count
            accumulated_text = ''  # Reset accumulated text
            char_count_since_last_marker = 0  # Reset character count

    # Add any remaining text as the last section
    if accumulated_text:
        sections.append((accumulated_text, char_count_since_last_marker))

    # Populate the dictionary with sections
    for i, (section, char_count) in enumerate(sections, start=1):
        section_key = f"Section {i} - {char_count} chars"
        section_dict[section_key] = section

    return section_dict


def get_original_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def write_replace_file(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def add_text(file_path, text):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write('\n--- new entry ---' + text)


def remove_pattern_and_line_from_file(input_file_path, output_file_path, pattern):
    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            if pattern not in line:
                output_file.write(line)
            else:
                continue




if __name__ == '__main__':
    import os
    file_path = os.path.join(os.getcwd(),"data","ama_raw_text.txt")
    file_path_1 = os.path.join(os.getcwd(),"data","text_1.txt")
    file_path_2 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_2.txt")
    reference_file = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\references.txt")  # Live
    file_path_3 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_3.txt")
    file_path_4 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_4.txt")
    file_path_5 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_5.txt")
    examples_file = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\examples.txt")  # Live
    file_path_6 = os.path.join(os.getcwd(),"data","text_6.txt")
    file_path_7 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_7.txt")
    low_char_count_text = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\low_char_count_text.txt")
    file_path_8 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_8.txt")
    file_path_9 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_9.txt")
    file_path_10 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_10.txt")
    file_path_11 = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_11.txt")
    file_path_12 = os.path.join(os.getcwd(),"data","text_12.txt")
    index_text = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\index_text.txt")
    glossary_text = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\glossary_text.txt")
    remaining_tables = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\remaining_tables.txt")
    extracted_text = (r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\extracted_text.txt")  # Live

    original_document = get_original_text(file_path)

    # Step 1: remove excessive blank lines
    cleaned_document = clean_blank_lines(original_document,1)

    # print(cleaned_document)
    # Step 2: Normalize chapter headers

    updated_document = normalize_chapter_headers(cleaned_document)

    # Step 3: Extract references and update document
    references, updated_document = extract_references_and_update_document(updated_document)
    # write_replace_file(reference_file, references)


    # Step 4: Organize chapter sections
    updated_document = organize_chapter_sections(updated_document)

    # Step 5: Wrap and clean examples
    updated_document = mark_examples(updated_document)

    # Step 6: Extract and remove known examples
    updated_document, examples = extract_and_remove_examples(updated_document)

    # Write the cleaned document to a new file
    # write_replace_file(file_path_6, updated_document)
    # write_replace_file(examples_file, examples)

    # remove_references(file_path)
    # normalize_chapter_headers_improved(file_path)

    # Step 7: Extract sections with low text content (Did it again towards the end) - 25 lines, 100 characters first time. - second time 8/50, third time 5/14
    line_count = 5
    char_count = 14
    extracted_sections_str, remaining_text_str = extract_short_sections(updated_document, 8, 50)
    # print(extracted_sections_str)

    # add_text(low_char_count_text, extracted_sections_str)
    # write_replace_file(file_path_11, remaining_text_str)

    # Step 8: Eliminate extra empty lines
    updated_document = eliminate_extra_empty_lines(remaining_text_str)

    line_starts_with = 'Index'
    location = 'before'
    text_to_add = '-------- INDEX START -------'
    updated_document = add_dynamic_marker(updated_document, line_starts_with, location, text_to_add)

    # write_replace_file(file_path_9, updated_document)

    # start_marker = '---- GLOSSARY START----'
    # end_marker = '---- GLOSSARY END----'
    # extracted_text, remaining_text = extract_between_markers(original_document, start_marker, end_marker)
    # write_replace_file(glossary_text, extracted_text)

    # Lines starting with tab indent
    # updated_text = handle_text_starting_with_tab_indent(original_document)

    # Identify Introduction
    updated_text = add_dynamic_marker(updated_document, "Introduction", location="before", text_to_add="-------- INTRODUCTION START -------")

    # Replace tabs with newlines
    # updated_text = convert_tabs_to_newlines(original_document)

    # extract text sections dynamically
    start_marker = "---START MANUAL EXAMPLE---"
    end_marker = "---END EXAMPLE---"
    max_lines = 20
    original_file_path = file_path_10
    remaining_file_path = file_path_10
    extracted_text_file_path = examples_file
    # extract_and_remove_text_sections(start_marker, end_marker, max_lines, original_file_path, remaining_file_path, extracted_text_file_path)

    # Identify History (same as Introduction)
    # updated_text = add_dynamic_marker(original_document, "History", location="before", text_to_add="-------- INTRODUCTION START (History)-------")

    # Replace text pattern (Fixing words that were split with a dash to make a word again)
    #file = r"/knowledgebase/utils/doc_manipulation/data/examples.txt"
    #original_document = get_original_text(file)

    #old_pattern = '- '
    #new_pattern = ''
    # updated_text = replace_text_pattern(original_document, old_pattern, new_pattern)
    # write_replace_file(file, updated_text)

    #pattern = '--break--'
    # updated_text = delete_lines_starting_with_pattern(original_document, pattern)

    # write_replace_file(file_path_10, updated_text)

    # Remove pattern and numbers
    updated_text = remove_pattern_and_numbers(updated_document)


    # updated_text = clean_blank_lines(original_document, 1)
    # write_replace_file(file_path_12, updated_text)

    # input_file_path = r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_13.txt"
    # output_file_path = r"D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\text_14.txt"
    # insert_breaks(input_file_path, output_file_path)

    #input_file_path = r"/knowledgebase/utils/doc_manipulation/data/text_14.txt"
    #output_file_path = r"/knowledgebase/utils/doc_manipulation/data/text_15.txt"

    # remove_pattern_and_line_from_file(input_file_path, output_file_path, pattern)

    # insert_character_count_markers(input_file_path, output_file_path)

    # Split text by marker
    #path_to_source = r"/knowledgebase/utils/doc_manipulation/data/chapters/full_text.txt"
    #marker_text = '================================================================================'
    #new_file_name = r"chapter_text"
    # split_text_by_marker(path_to_source, marker_text, new_file_name)

    #final_path = r"/knowledgebase/utils/doc_manipulation/data/text_14.txt"
    #text = get_original_text(final_path)
    test_strings = [
        "================================================================================",
        "-----------------------------------------------------------------",
        "-------INTRODUCTION"
    ]
    #json_path = r"/knowledgebase/utils/doc_manipulation/data/ama_text_2.json"
    #new_text_path = r"/knowledgebase/utils/doc_manipulation/data/ama_text_2.txt"
    #full_dict = text_to_dict(text, max_chars=3000, test_strings=test_strings)
    #save_dict_to_json(full_dict, json_path)
    #extract_json_to_text(json_path, new_text_path)

    #pretty_print_data(full_dict)

