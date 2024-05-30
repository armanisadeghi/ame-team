import re
import subprocess
import os
import tempfile
import time
from common import pretty_print, get_sample_data
from ame_team.settings.base import BASE_DIR, TEMP_DIR
from automation_matrix.processing.processor import Processor


class OutputClassifier(Processor):
    def __init__(self):
        super().__init__()
        self.clean_text = False
        self.separated_sections = []
        self.classified_sections = []
        self.table_tuples = []
        self.patterns = {
            "- ": "bullet",
            " -": "sub_bullet",
            "  -": "sub_bullet",
            "   -": "sub_bullet",
            "| ": "table",
            "|-": "divider",
            "---": "section_break",
            "***": "section_break",
            "#": "header",
            "**": "bold_text",
            "__": "italic_text",
            ">": "quote",
            "[": "link",
            "![": "image",
        }

        self.section_checks = {
            'code_block': self.is_code_block,
            'table': self.is_table,
            'entries_and_values': self.is_entries_and_values,
            'header_with_bullets': self.is_header_with_bullets,
            'header_with_numbered_list': self.is_header_with_numbered_list,
            'header_with_text': self.is_header_with_text,
            'bold_text_with_sub_bullets': self.is_bold_text_with_sub_bullets,
            'plain_text': self.is_plain_text,
        }

    def classify_line(self, line):
        for pattern, category in self.patterns.items():
            if line.startswith(pattern):
                return category

        if re.match(r'\d+\.', line):
            return "numbered_list"
        elif line.endswith(':') and len(line) < 50:
            return "header_text"
        elif line.startswith('**'):
            return "bold_text"
        elif ':' in line and not line.endswith(':'):
            return "entry_and_value"
        elif line.strip() == '':
            return "line_break"
        elif line.startswith('```') and line.strip() == '```':
            return "end_code_block"
        elif line.startswith('```'):
            return "start_code_block"
        elif line.startswith('#'):
            return "header"
        elif line.startswith('__'):
            return "italic_text"
        else:
            return "other_text"

    def process_text(self, text):
        classified_lines = []
        lines = text.split('\n')
        for line in lines:
            category = self.classify_line(line)
            if self.clean_text:
                line = self.text_cleaner(category, line)
            classified_lines.append((category, line))

        return classified_lines

    def text_cleaner(self, category, line):

        # TODO: This was added by Armani for demonstration purposes, but we know it's not ideal because it is removing text from anywhere, including the middle, which we don't want to do.
        # It would be a very quick fix to look specifically and remove starting text and ending text that meet these requirements.

        clean_text = line
        if category == 'header':
            clean_text = line.replace('#', '').strip()
        elif category == 'bold_text':
            clean_text = line.replace('**', '').strip()
        elif category == 'italic_text':
            clean_text = line.replace('__', '').strip()
        elif category == 'bullet':
            clean_text = line.replace('- ', '').strip()
            clean_text = clean_text.replace('**', '').strip()
        elif category == 'sub_bullet':
            clean_text = line.replace(' -', '').strip()
        elif category == 'numbered_list':
            clean_text = line.replace('\d+\.', '').strip()
            clean_text = clean_text.replace('**', '').strip()
        elif category == 'entry_and_value':
            clean_text = line.split(':', 1)[1].strip()
        elif category == 'header_text':
            clean_text = line.split(':', 1)[1].strip()
        elif category == 'other_text':
            clean_text = clean_text.replace('**', '').strip()
            clean_text = clean_text.replace('#', '').strip()

            clean_text = clean_text.strip()

        return clean_text

    def split_text_sections_and_identify_line_types(self, text):
        lines = text.strip().split('\n')
        sections = []
        current_section = []
        in_code_block = False

        for line in lines:
            if (line.strip() == '' or line.strip().startswith('---')) and not in_code_block:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            elif line.strip().startswith('```'):
                if in_code_block:
                    current_section.append(line)
                    sections.append('\n'.join(current_section))
                    current_section = []
                else:
                    if current_section:
                        sections.append('\n'.join(current_section))
                        current_section = []
                    current_section.append(line)
                in_code_block = not in_code_block
            else:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section))

        self.separated_sections = [self.process_text(section) for section in sections]
        # pretty_print(self.separated_sections)

        return self.separated_sections

    def is_header_with_text(self, section):
        return section[0][0] == 'header' and any(entry[0] == 'other_text' for entry in section[1:])

    def is_code_block(self, section):
        return section[0][0] == 'start_code_block' and section[-1][0] == 'end_code_block'

    def is_table(self, section):
        return any(entry[0] == 'table' for entry in section)

    def is_entries_and_values(self, section):
        return all(entry[0] == 'entry_and_value' for entry in section)

    def is_header_with_bullets(self, section):
        return section[0][0] == 'header_text' and all(entry[0] == 'bullet' for entry in section[1:])

    def is_header_with_numbered_list(self, section):
        return section[0][0] == 'header_text' and all(entry[0] == 'numbered_list' for entry in section[1:])

    def is_bold_text_with_sub_bullets(self, section):
        if section[0][0] == 'bold_text' and all(entry[0] == 'sub_bullet' for entry in section[1:]):
            return True
        return False

    def is_plain_text(self, section):
        return all(entry[0] == 'other_text' for entry in section)

    def categorize_sections(self, data):
        for section in data:
            section_type = 'other_section_type'
            for type_name, check_func in self.section_checks.items():
                if check_func(section):
                    section_type = type_name
                    break

            self.classified_sections.append((section_type, section))

        return self.classified_sections

    def table_to_list_of_tuples_final(self, table_text):
        lines = table_text.strip().split('\n')
        updated_tables = []

        for line in lines:
            if line.strip().startswith('|'):
                row = tuple(cell.strip() for cell in line.split('|') if cell.strip())
                if any(cell != '-' and not cell.startswith('---') for cell in row):
                    updated_tables.append(row)

        self.table_tuples.append(updated_tables)
        return

    def classify_output_details(self, text, clean_text=False):
        self.clean_text = clean_text  # Added by Armani

        self.split_text_sections_and_identify_line_types(text)
        self.categorize_sections(self.separated_sections)
        # TODO: Jatin this is where you would add a feature to create structure from the data, if that's what we want next. (Could be optional)

        return self.classified_sections

    def prepare_data_for_google_docs(self, text):
        self.classify_output_details(text)
        self.table_to_list_of_tuples_final(text)

        return self.classified_sections, self.table_tuples


def convert_markdown_content_to_docx(markdown_content, output_docx_file_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmpfile:
        markdown_file_path = tmpfile.name
        tmpfile.write(markdown_content.encode('utf-8'))

    base, ext = os.path.splitext(output_docx_file_path)
    timestamp = int(time.time())
    output_docx_file_path = f"{base}_{timestamp}{ext}"

    try:
        command = ['pandoc', markdown_file_path, '-o', output_docx_file_path]

        subprocess.run(command, check=True)

        print(f'Successfully created {output_docx_file_path}')
    finally:
        os.remove(markdown_file_path)


async def classify_markdown_content(text_data):
    classifiers = OutputClassifier()
    classified_sections = classifiers.classify_output_details(text_data)
    pretty_print(classified_sections)

    classified_sections_dict = {}
    classified_sections_text = {}

    print("\nCategorized sections:")
    for count, (section_type, section) in enumerate(classified_sections, start=1):
        print(f"{count}. {section_type}")
        # Populate dictionary with type as key and section as value
        classified_sections_dict[section_type] = section

    print("\n========================== Content =========================:")
    for section_type, section in classified_sections:
        section_text = []
        print(f"\n----- {section_type} -----")
        for category, line in section:
            print(f"{line}")
            section_text.append(line)
        classified_sections_text[section_type] = "\n".join(section_text)

    classified_content = {
        "classified_sections_dict": classified_sections_dict,
        "classified_sections_text": classified_sections_text
    }

    return classified_content


async def get_classify_markdown_section_list(text_data):
    classifiers = OutputClassifier()
    classified_sections = classifiers.classify_output_details(text_data)

    return classified_sections


async def main(text_data):

    clean_text = True  # Turn this to false to get the old functionality back - NEW FUNCTIONALITY

    classifiers = OutputClassifier()

    classified_sections = classifiers.classify_output_details(text_data, clean_text=clean_text)
    pretty_print(classified_sections)

    print("\nCategorized sections:")
    for count, (section_type, section) in enumerate(classified_sections, start=1):
        print(f"{count}. {section_type}")

    print("\n========================== Content =========================:")
    for section_type, section in classified_sections:
        print(f"\n----- {section_type} -----")
        for category, line in section:
            print(f"{line}")


if __name__ == "__main__":
    import asyncio

    sample_data = get_sample_data(app_name="automation_matrix",
                                  data_name="markdown_content",
                                  sub_app="ama_ai_output_samples",
                                  )

    asyncio.run(main(sample_data))

    # VERY GOOD FOR MARKDOWN TO DOCX CONVERSION
    # output_docx_file_path = os.path.join(BASE_DIR, "temp", "app_outputs", "output.docx")

    # convert_markdown_content_to_docx(sample_data, output_docx_file_path=output_docx_file_path)
