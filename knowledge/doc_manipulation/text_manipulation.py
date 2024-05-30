import django

django.setup()
import re
import json
from common.utils.my_utils import pretty_print_data
from knowledgebase.utils.add_data import DataManager
from google_integrations.cloud_storage.storage_manager import upload_content_to_bucket


def clean_text(text):
    text = text.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201c", '"').replace(u"\u201d", '"')
    text = text.replace(u"\u2013", "-").replace(u"\u2014", "--").replace(u"\u2026", "...")
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def save_dict_to_json(data_dict, file_path):
    cleaned_data = {key: clean_text(value) for key, value in data_dict.items()}
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(cleaned_data, json_file, ensure_ascii=False)


def extract_json_to_text(json_path, text_path):
    with open(json_path, 'r') as json_file:
        data_dict = json.load(json_file)

    with open(text_path, 'w') as text_file:
        for i, (section, content) in enumerate(data_dict.items(), 1):
            text_file.write(content)
            if i < len(data_dict):
                text_file.write('\n--break--\n')


def text_to_dict(text, max_chars=3000, text_strings=[]):
    break_marker = "--break--"
    sections = []  # List to hold each section of text along with their char counts
    section_dict = {}  # Dictionary to return
    accumulated_text = ''
    char_count_since_last_marker = 0
    input_lines = text.split('\n')

    def filter_test_strings(lines, test_strings):
        return [line for line in lines if all(test_string not in line for test_string in test_strings)]

    input_lines = filter_test_strings(input_lines, text_strings)

    for line in input_lines:
        if accumulated_text:
            accumulated_text += '\n'
        accumulated_text += line
        char_count_since_last_marker += len(line) + 1

        if break_marker in line:
            accumulated_text = accumulated_text.replace(break_marker, '')
            if char_count_since_last_marker >= max_chars:
                sections.append((accumulated_text, char_count_since_last_marker))
                accumulated_text = ''
                char_count_since_last_marker = 0
        elif char_count_since_last_marker >= max_chars:
            sections.append((accumulated_text, char_count_since_last_marker))
            accumulated_text = ''
            char_count_since_last_marker = 0

    if accumulated_text:
        sections.append((accumulated_text, char_count_since_last_marker))

    for i, (section, char_count) in enumerate(sections, start=1):
        section_key = f"Section {i} - {char_count} chars"
        section_dict[section_key] = section

    return section_dict


if __name__ == '__main__':
    #upload_or_update_raw_data_from_file('data/raw_data.json', get_raw_data_version())
    manager = DataManager()
    latest_version = manager.get_raw_data_version('AMA Guides Raw Text Without Tables')
    text = latest_version.text_content

    # first_version = get_raw_data_version('AMA Guides Raw Text Without Tables', version_index=1)
    # print(first_version.text_content)

    text_strings = [
        "================================================================================",
        "-----------------------------------------------------------------",
        "-------INTRODUCTION"
    ]

    full_dict = text_to_dict(text, max_chars=3000, text_strings=text_strings)

    clean_data = {
        'name': 'AMA Guides',
        'text_content': text,
        'dict_content': full_dict,
        'source': latest_version.source,
        'status': 'Cleaned',
        'comments': 'Various elements, including tables, references, examples, low character count sections, etc. have been removed.',
    }
    pretty_print_data(manager.add_clean_data(**clean_data))

    #json_path = r"/knowledgebase/utils/doc_manipulation/data/ama_text_2.json"
    #new_text_path = r"/knowledgebase/utils/doc_manipulation/data/ama_text_2.txt"
    #save_dict_to_json(full_dict, json_path)
    #extract_json_to_text(json_path, new_text_path)
