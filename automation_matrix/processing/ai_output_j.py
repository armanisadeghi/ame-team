import re
import ast
from bs4 import BeautifulSoup
from typing import Dict
from collections import defaultdict
from typing import List, Union
import asyncio
from common import vcprint, pretty_print, get_sample_data
from automation_matrix.processing.markdown.organizer import Markdown
from automation_matrix.processing.processor import ProcessingManager
from automation_matrix.processing.markdown.classifier import get_classify_markdown_section_list
verbose = False


# Working Processors:
# - extract_code_snippets - Tested locally, but not in workflow yet.
# - get_markdown_asterisk_structure - Tested separately, but not here and not in workflow yet.


class AiOutput(ProcessingManager):

    def __init__(self, content):
        self.content = content
        self.return_params = {}
        self.processed_content = {}
        self.variable_name = None
        self.source_info = {}
        super().__init__()

    async def process_response(self, return_params):
        if not isinstance(return_params, dict):
            self.return_params = {}
        else:
            self.return_params = return_params

        self.source_info = self.return_params.get('source_info', '')
        self.variable_name = self.return_params.get('variable_name', '')
        if not isinstance(self.variable_name, str):
            self.variable_name = str(self.variable_name)

        self.processed_content = {
            'signature': 'OpenAIWrapperResponse',
            'processing': True,
            "variable_name": self.variable_name,
            'source_info': self.source_info,
            'value': self.content,
            'processed_values': {}
        }

        if self.return_params:
            processors = self.order_processors(self.return_params.get('processors', []))
            await self.process_processors(processors)

        return self.processed_content

    def order_processors(self, processors):
        processor_names = {processor['processor'] for processor in processors}

        # Check each processor to confirm the "depends_on" value is valid. If it's not, or if it's missing, set it to "content".
        for processor in processors:
            if 'depends_on' not in processor or not processor['depends_on']:
                processor['depends_on'] = 'content'

            # Setting the dependency to 'content' -  'content' Is the raw data.
            elif processor['depends_on'] not in processor_names and processor['depends_on'] not in ['content', 'raw_api_response']:
                vcprint(verbose=True, data=f"\n[Warning!] '{processor}' has an invalid dependency on '{processor['depends_on']}'. Changing to 'content'.\n", color='red', style='bold')
                processor['depends_on'] = 'content'

        # ------- Get the processors in teh right order, regardless of what order they were provided in. -------
        # This is a good example of how I like to code. It's MY job to fix a mistake that the other programmer or the user made, if it doesn't hurt anything.
        # Order the processors based on their dependencies - Don't want to run something, before the dependencies are met.
        def add_processor_if_dependency_met(processor, ordered, unprocessed):
            if processor['depends_on'] == 'content' or any(dep['processor'] == processor['depends_on'] for dep in ordered):
                ordered.append(processor)
                unprocessed.remove(processor)
                return True
            return False

        ordered_processors = [proc for proc in processors if proc['depends_on'] == 'raw_api_response']
        unprocessed_processors = [proc for proc in processors if proc not in ordered_processors]

        for proc in list(unprocessed_processors):
            if proc['depends_on'] == 'content':
                ordered_processors.append(proc)
                unprocessed_processors.remove(proc)
        # ------- Up to here is just for ordering the processors based on dependencies. -------

        progress = True

        # Now, we have the ordered processors and we will keep going through them until done.
        while unprocessed_processors and progress:
            progress = False
            for proc in list(unprocessed_processors):
                if add_processor_if_dependency_met(proc, ordered_processors, unprocessed_processors):
                    progress = True

        if unprocessed_processors:
            print("\n[Warning!] Unable to resolve all dependencies without circular references.\n")
            ordered_processors.extend(unprocessed_processors)

        return ordered_processors

    async def process_processors(self, processors):
        processed_names = {}
        count = 0
        for step in processors:
            processor_name = step.get('processor')
            args = step.get('args', {})
            depends_on = step.get('depends_on') or 'content'
            count += 1

            print(f"[processing data {count}] Processor:'{processor_name}' with dependency on: '{depends_on}'")

            input_data = self.content if depends_on in ['content', 'raw_api_response'] else processed_names.get(depends_on)

            processor = getattr(self, processor_name, None)
            if not processor:
                print(f"No processor found for '{processor_name}'")
                continue

            try:
                output_data = await processor(input_data, **args) if asyncio.iscoroutinefunction(processor) else processor(input_data, **args)
            except Exception as e:
                print(f"Error processing '{processor_name}': {e}")
                output_data = "error"

            self.processed_content['processed_values'][processor_name] = {
                'value': output_data,
                'depends_on': depends_on,
                'args': args
            }
            processed_names[processor_name] = output_data

    async def extract_code_snippets(self, text: str) -> Dict[str, List[str]]:
        code_snippets = {}
        pattern = r'(?P<delim>`{3}|\'{3})(?P<lang>[\w#+-]*)\n(?P<code>.*?)\n(?P=delim)'
        matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
        for match in matches:
            language = match.group('lang') or 'no_language'
            code = match.group('code')
            if code_snippets.get(language):
                code_snippets[language].append('\n' + code)
            else:
                code_snippets.setdefault(language, []).append(code)
        return code_snippets

    async def get_markdown_asterisk_structure(self, content: str) -> Dict[str, Union[str, List[str]]]:
        processor = Markdown(style='asterisks')
        asterisk_structure_results = await processor.process_markdown(content)
        # print(f"-------------- DEBUG: Asterisk Structure Results:--------------------------------")
        # pretty_print(asterisk_structure_results)

        return asterisk_structure_results

    # Creates groups from the data. For example, it can combine every two or three items into a group so they go together.
    async def data_groups(self, data, parts_count=2):
        text_value_list = data.get('plain_text', [])
        data_groups = []
        for i in range(0, len(text_value_list), parts_count):
            group = {
                'parts': ['' for _ in range(parts_count)]
            }

            for j in range(parts_count):
                if i + j < len(text_value_list):
                    group['parts'][j] = text_value_list[i + j]
            data_groups.append(group)
        return data_groups

    # The purpose of this is to get rid of "text: " or "Headline: " or "Question: " or "Answer: " or "title: " or "description: " etc. at the start of each item.
    async def clean_groups(self, data):
        pattern = re.compile(r'^.*?:\s')  # Matches any 'text: ' pattern at the start of a string
        cleaned_pairs = []
        for group in data:
            cleaned_parts = []
            for part in group['parts']:
                cleaned_part = re.sub(pattern, '', part)
                cleaned_parts.append(cleaned_part)
            cleaned_pairs.append({
                'parts': cleaned_parts
            })
        return cleaned_pairs

    # This adds manual entries for cases where we need additional, generic parts that are the same for all entries. (e.g. "system message" for fine tune data)
    async def add_parts(self, data, additional_parts=None):
        if isinstance(data, list) and isinstance(additional_parts, list):
            for item in data:
                if isinstance(item, dict) and 'parts' in item and isinstance(item['parts'], list):
                    item['parts'].extend(additional_parts)
                else:
                    continue
        return data

    # This allows the list of "parts" to be converted into a dictionary with named keys. This is useful for making the data more readable and structured.
    async def define_key_value_pairs(self, data, keys=[]):
        if not isinstance(data, list):
            return data

        def generate_generic_keys(length):
            return [f'part {i + 1}' for i in range(length)]

        updated_data = []
        for item in data:
            if isinstance(item, dict) and 'parts' in item:
                parts = item['parts']
                if not keys or len(keys) < len(parts):
                    item_keys = keys + generate_generic_keys(len(parts) - len(keys))
                else:
                    item_keys = keys[:len(parts)]

                keyed_parts = dict(zip(item_keys, parts))
                updated_data.append(keyed_parts)
            else:
                return data
        return updated_data

    async def oai_fine_tuning_data(self, data):
        training_data = []
        for item in data:
            system_content = item.get("system_content", "")
            user_content = item.get("user_content", "")
            assistant_content = item.get("assistant_content", "")

            if not system_content or not user_content or not assistant_content:
                continue

            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    },
                ]
            }
            training_data.append(entry)

        return training_data

    async def process_urls(self, text: str, *args, remove_values: bool = True) -> Union[List[str], str]:
        url_pattern = r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+\b'
        if remove_values:
            return re.sub(url_pattern, '', text)
        else:
            return re.findall(url_pattern, text)

    async def parse_table(self, text: str, *args) -> List[dict]:
        table_pattern = '\\+-[+-]+\\+\\n(?:\\|.*\\|\\n)+\\+-[+-]+\\+'
        match = re.search(table_pattern, text)
        if not match:
            return []
        table = match.group().split('\n')[1:-1]
        headers = [header.strip() for header in table[0].split('|')[1:-1]]
        rows = [dict(zip(headers, [cell.strip() for cell in row.split('|')[1:-1]])) for row in table[1:]]
        return rows

    async def identify_python_dictionaries(self, text: str, *args) -> List[dict]:
        """
        Searches for and interprets Python dictionary structures.

        :param text: String containing the text to be processed.
        :return: List of Python dictionaries extracted from the text.
        """
        dict_pattern = '\\{(?:[^{}]|(?:\\{.*?\\}))*\\}'
        matches = re.findall(dict_pattern, text)
        dictionaries = []
        for match in matches:
            try:
                dictionary = ast.literal_eval(match)
                if isinstance(dictionary, dict):
                    dictionaries.append(dictionary)
            except (SyntaxError, ValueError):
                continue
        print(f"Python Dictionaries:\n{dictionaries}\n")
        return dictionaries

    async def extract_lists(self, text: str, *args) -> List[List[str]]:
        """
        Identifies both bullet-point and numbered lists and extracts them.

        :param text: String containing the text to be processed.
        :return: List of lists extracted from the text.
        """

        # TODO it's identifying lists of LSIs but not capturing the name of what the list is
        list_pattern = '(?:^\\s*(?:[-*]|\\d+\\.)\\s+.*(?:\\n|$))+'
        matches = re.findall(list_pattern, text, re.MULTILINE)
        lists = []
        for match in matches:
            items = [item.strip() for item in re.split(
                '\\n\\s*(?:[-*]|\\d+\\.)\\s+', match.strip()) if item]
            lists.append(items)
        print(f"Lists:\n{lists}\n")
        return lists

    async def parse_html_content(self, text: str, *args) -> List[str]:
        """
        Extracts HTML content sections as they are from the text.

        :param text: String containing the text with HTML.
        :return: List of HTML content sections.
        """
        html_pattern = '<[^>]+>.*?</[^>]+>'
        html_sections = re.findall(html_pattern, text, re.DOTALL)
        print(f"HTML Sections:\n{html_sections}\n")
        return html_sections

    async def extract_markdown_entries(self, text):
        entries = {}
        current_key = None
        current_value = []

        # Splitting the text by lines to process each line individually
        lines = text.split('\n')

        for line in lines:
            # Checking if the line starts with a bolded section
            match = re.match(r'^\s*\d+\.\s+\*\*(.*?):\s*\*\*(.*)', line)
            if match:
                # If there's a current key and value, save them before starting a new section
                if current_key:
                    # Append the current value as a complete entry to the list of entries for the current key
                    if current_key in entries:
                        entries[current_key].append('\n'.join(current_value).strip())
                    else:
                        entries[current_key] = ['\n'.join(current_value).strip()]
                    # Reset for the next entry
                    current_value = []

                # Update the current key with the new section's key
                current_key = match.group(1).strip()
                # Start the new entry's value with the rest of the current line
                current_value.append(match.group(2).strip())
            elif current_key:
                # If the line doesn't start a new section but there's an ongoing section, continue accumulating its content
                current_value.append(line.strip())

        # Add the last entry to the dictionary if there's one
        if current_key and current_value:
            if current_key in entries:
                entries[current_key].append('\n'.join(current_value).strip())
            else:
                entries[current_key] = ['\n'.join(current_value).strip()]

        extracted_markdown_entries = entries
        return extracted_markdown_entries

    async def extract_nested_markdown_entries(selv, text: str) -> Dict[str, List[str]]:
        entries = {}
        current_main_key = None
        buffer = ""

        # Split the text using the '** ' and '- **' markers, while keeping the markers with the split segments
        segments = re.split(r'(\*\* |- \*\*)', text)
        segments.append('')  # Ensure the loop processes the last segment

        for i in range(0, len(segments) - 1, 2):
            # Determine if the current segment is a main entry, nested entry, or text continuation
            current_marker = segments[i]
            current_text = segments[i + 1]

            if current_marker == '**':
                # Save any buffered text to the last nested entry before starting a new main entry
                if buffer and current_main_key and entries[current_main_key]:
                    entries[current_main_key][-1] += buffer
                    buffer = ""

                current_main_key = current_marker + current_text
                entries[current_main_key] = []
            elif current_marker == '- **' and current_main_key:
                # Save any buffered text to the last nested entry before starting a new nested entry
                if buffer and entries[current_main_key]:
                    entries[current_main_key][-1] += buffer

                entries[current_main_key].append(current_marker + current_text)
                buffer = ""  # Reset buffer after starting a new nested entry
            else:
                # Accumulate text continuation in the buffer
                buffer += current_text

        # Append any remaining buffered text to the last nested entry
        if buffer and current_main_key and entries[current_main_key]:
            entries[current_main_key][-1] += buffer

        nested_markdown_entries = entries
        # pretty_print(nested_markdown_entries)
        return nested_markdown_entries

    async def parse_markdown_content(self, text: str, *args) -> List[str]:
        # TODO not sure exactly what markdown content should include but it's including LSIs, but maybe that makes sense
        # If that makes sense, we should make this like a "parent" function that feeds others like extract_lists
        # For a blog section example, it returned the title for each section, but not the heading, which was interesting.
        # Again, useful, but in this case, it would be highly sepcific to the use case
        markdown_pattern = (
            '(?:\\n|^)(?:\\#{1,6}\\s.*|[-*]\\s.*|\\d+\\.\\s.*|\\>\\s.*|```[\\s\\S]*?```)'
        )
        markdown_sections = re.findall(markdown_pattern, text, re.MULTILINE)
        print(f"Before Processing:\n{text}\n\n")
        print(f"Markdown Sections:\n{markdown_sections}\n")
        pretty_print(markdown_sections)
        return markdown_sections

    async def extract_latex_equations(self, text: str, *args) -> List[str]:
        """
        Searches for LaTeX equation patterns (both inline and display mode) and extracts them.

        :param text: String containing the text to be processed.
        :return: List of LaTeX equations.
        """
        latex_pattern = '\\\\\\(.*?\\\\\\)|\\\\\\[.*?\\\\\\]'
        latext_quotations = re.findall(latex_pattern, text)
        print(f"LaTeX Equations:\n{latext_quotations}\n")
        return latext_quotations

    async def extract_plain_text(self, text: str, *args) -> str:
        """
        Extracts plain text content, filtering out any special formatting or structured content.

        :param text: String containing the text to be processed.
        :return: Plain text content.
        """
        # TODO Considering it pulled a list of LSIs as "plain text", I'm not sure this is working as intended
        # I think we need a set of initial checks that essentially breaks up the content into "major cateogries" without any overlap, or very little overlap
        # Then we can have a set of functions that are more specific to each category
        # Otherwise, we might end up with so much duplicate content that it will clutter the results and make it really hard to interpret or save to the db
        text = re.sub('```[\\s\\S]*?```', '', text)
        text = re.sub('\\[.*?\\]\\(.*?\\)', '', text)
        text = re.sub('\\\\\\(.*?\\\\\\)', '', text)
        text = BeautifulSoup(text, 'html.parser').get_text()
        final_text = text.strip()
        print(f"Plain Text:\n{final_text}\n")
        return final_text

    async def identify_fill_in_blanks(self, text: str, *args) -> List[str]:
        """
        Identifies sentences with fill-in-the-blank structures.

        :param text: String containing the text to be processed.
        :return: List of sentences with blanks.
        """
        blank_pattern = '[^\\.\\?\\!]*__+[^\\.\\?\\!]*[\\.|\\?|\\!]'
        final_pattern = re.findall(blank_pattern, text)
        print(f"Fill-in-the-Blanks:\n{final_pattern}\n")
        return final_pattern

    async def parse_multiple_choice_questions(self, text: str, *args) -> List[str]:
        """
        Identifies and extracts multiple-choice questions.

        :param text: String containing the text to be processed.
        :return: List of multiple-choice questions with options.
        """
        mcq_pattern = '(?:\\n|^).*\\?\\n(?:\\s*[a-zA-Z]\\)\\s.*\\n)+'
        final_pattern = [mcq.strip() for mcq in re.findall(mcq_pattern, text, re.MULTILINE)]
        print(f"Multiple-Choice Questions:\n{final_pattern}\n")
        return final_pattern

    async def extract_paragraphs(self, text: str, *args) -> List[str]:
        """
        Extracts paragraph-style text content.

        :param text: String containing the text to be processed.
        :return: List of paragraphs.
        """
        # TODO Not working as intended because it's pulling the list of LSIs, which is definitely not a paragraph. We need to be clear what a paragraph is
        paragraphs = re.split('\\n{2,}', text)
        extracted_paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
        print(f"Paragraphs:\n{extracted_paragraphs}\n")
        return extracted_paragraphs

    async def identify_html_markdown_structured_text(self, text: str, *args) -> Dict[str, List[str]]:
        """
        Specifically looks for and extracts structured text marked as HTML or Markdown.

        :param text: String containing the text to be processed.
        :return: Dictionary with HTML and Markdown content categorized.
        """
        # TODO Even through it actually returned empty results, they still printed, which might be a debugging print, but these should return nothing if they're empty
        # By providing resopnses with empty dictionaries, we're creating a lot more work for later, unless that was the intention - Need to review
        structured_content = {
            'html': [],
            'markdown': []
        }
        html_pattern = '```html\\n([\\s\\S]*?)\\n```'
        markdown_pattern = '```markdown\\n([\\s\\S]*?)\\n```'
        html_matches = re.findall(html_pattern, text)
        structured_content['html'].extend(html_matches)
        markdown_matches = re.findall(markdown_pattern, text)
        structured_content['markdown'].extend(markdown_matches)

        print(f"Structured Content:\n{structured_content}\n")
        return structured_content

    async def extract_prompts_questions(self, text: str, *args) -> List[str]:
        """
        Extracts prompts and questions designed for user interaction.

        :param text: String containing the text to be processed.
        :return: List of prompts and questions.
        """
        prompt_question_pattern = '[^\\.\\?\\!]*\\?\\s*(?:\\n|$)'
        final_pattern = re.findall(prompt_question_pattern, text)
        print(f"Prompts and Questions:\n{final_pattern}\n")
        return final_pattern

    async def find_words_after_triple_quotes(self, text: str, *args) -> Dict[str, int]:
        pattern = re.compile("\\'\\'\\'(\\w+)")
        matches = re.findall(pattern, text)
        word_count = defaultdict(int)
        for word in matches:
            word_count[word] += 1
        print(f"Words After Triple Quotes:\n{word_count}\n")
        return word_count

    async def extract_markdown(self, text: str, *args):
        # Implement markdown extraction logic
        pass

    async def extract_json(self, text: str, *args):
        # Implement JSON extraction logic
        pass

    async def extract_code(self, text: str, *args):
        # Implement code extraction logic
        pass

    async def extract_python_code(self, text: str, *args):
        # Implement Python code extraction logic
        pass

    async def extract_code_remove_comments(self, text: str, *args):
        # Implement logic to remove comments from code
        pass

    async def extract_from_json_by_key(self, text: str, *args):
        # Implement extraction from JSON by key
        pass

    async def extract_from_outline_by_numbers(self, text: str, *args):
        # Implement extraction from an outline by numbers
        pass

    async def new_method(self, text: str, *args):
        # Implement the logic here
        pass

    async def remove_first_and_last_paragraph(self, text: str, *args):
        # Implement the logic here
        pass

    async def local_sample_data_processing(self, sample_content, return_params=None):

        processed_content = {
            'initial_content': sample_content,
            'steps': {},
        }

        if return_params:
            for step in return_params:
                processor_name = step.get('processor')
                args = step.get('args', {})
                depends_on = step.get('depends_on', 'content')
                extraction = step.get('extraction', {})

                if depends_on == 'content':
                    input_data = sample_content
                else:
                    input_data = processed_content['steps'].get(depends_on, {}).get('output')

                processor = getattr(self, processor_name, None)

                if asyncio.iscoroutinefunction(processor):
                    output_data = await processor(input_data, **args)
                else:
                    output_data = await asyncio.to_thread(processor, input_data, **args)

                processed_content['steps'][processor_name] = {
                    'output': output_data,
                    'depends_on': depends_on,
                    'extraction': extraction
                }

        return processed_content

    async def get_classified_markdown(self, markdown):
        from automation_matrix.processing.markdown.classifier import get_classify_markdown_section_list
        sections = await get_classify_markdown_section_list(markdown)
        return sections

    async def get_items_from_classified_markdown_sections(self, sections, **kwargs):
        filtered_sections = []
        for sec in sections:
            for classifier_type, query in kwargs.items():

                if sec[0] == classifier_type: # Since 0th index of each section is the classifier or item_type
                    # Once we know the classified type eg. table or other_section_type or entries_and_values
                    # We will now search for the contain keyword in the section parts if specified
                    for subpart in sec[1]: # sec[1] since item at 1st index is a classification list of each line in the section
                        if query is not None and isinstance(query,str):
                            if query.upper() in subpart[1].upper(): #subpart[1] since , the 0 the item is the classification type of the line,we want to search for the line not the classification type
                                filtered_sections.append(sec)
                        else:
                            filtered_sections.append(sec)

        # print(filtered_sections)

        output_format = {
            "nested_structure": {},
            "plain_text": {}
        }

        for item in filtered_sections:
            for part in item[1]:
                type_ = part[0]
                value_ = part[1]
                if isinstance(output_format['nested_structure'].get(type_), list):
                    output_format['nested_structure'].get(type_).append(value_)
                else:
                    output_format['nested_structure'][type_]  = []
                    output_format['nested_structure'].get(type_).append(value_)

        return output_format


async def local_post_processing(sample_content):

    return_params = {
        'variable_name': 'SAMPLE_DATA_1001',
        'processors':
            [
                {
                    'processor': 'get_classified_markdown',
                    'depends_on': 'content',
                    "extraction": [],
                    'args' : {}
                },
                {
                    'processor': 'get_items_from_classified_markdown_sections',
                    'depends_on': 'get_classified_markdown',
                    "extraction": [{
                            "key_identifier": "nested_structure",
                            "key_index": "1",
                            "output_type": "text",
                        }],
                    'args': {'table': "Impairment Code", 'entries_and_values':'Date of birth',
                             'header_with_bullets': "Occupational Code"}
                }
            ],
    }


    processor = AiOutput(sample_content)
    processed_content = await processor.process_response(return_params)
    # print("========================================== Initial Content ==========================================")
    # print(processed_content['value'])
    # print(processed_content['processed_values'])
    # print("\nProcessed Steps:")
    #
    # for step_name, step_data in processed_content['processed_values'].items():
    #     print(f"\n========================================== {step_name} ==========================================\n")
    #
    #     output = step_data['value']
    #     if isinstance(output, dict):
    #         for key, value in output.items():
    #             print(f"--------- {key}: ---------")
    #             if isinstance(value, list):
    #                 for item in value:
    #                     print(f"\n{item}")
    #             else:
    #                 print(f"\n{value}")
    #     elif isinstance(output, list):
    #         for item in output:
    #             print(f"  - {item}")
    #     else:
    #         print(f"  {output}")
    #
    #     print("-" * 25)
    #     print("\nDepends on:", step_data['depends_on'])
    #     print("\nArgs:", step_data['args'])

    return processed_content


async def access_data_by_reference(reference, data_structure):
    """
    Access a nested dictionary entry based on a reference dictionary and return it as specified (text or dict).

    :param reference: A dictionary with 'data', 'key', and 'output' where 'data' is the key to the main dictionary,
                      'key' is the index to access within the nested dictionary (None for entire structure), and
                      'output' specifies the return format ('text' or 'dict').
    :param data_structure: The main data structure containing nested dictionaries.
    :return: The selected entry in the specified format, or a message if not found.

        # Sample entry to get the 3rd key as text
        reference_text_3rd = {"data": "nested_structure", "key": 1, "output": "text"}

        # Sample entry to get the 2nd key as a dict
        reference_dict_2nd = {"data": "nested_structure", "key": 1, "output": "dict"}

        # Sample entry to get the entire nested structure as text
        reference_entire_text = {"data": "nested_structure", "key": None, "output": "text"}

    """

    nested_key = reference.get('key_identifier')
    index = reference.get('key_index')
    output_format = reference.get('output_type', 'dict')  # Default to 'dict' if not specified
    pretty_print(reference)
    if not index and index != 0:
        nested_dict = data_structure.get(nested_key, {})
        if output_format == 'text':
            result = '\n\n'.join(f'{key}:\n{'\n'.join(value)}' for key, value in nested_dict.items())
        else:
            result = nested_dict
    else:
        index -= 1  # Adjust for 0-based indexing
        if nested_key in data_structure:
            nested_dict = data_structure[nested_key]
            keys = list(nested_dict.keys())
            if 0 <= index < len(keys):
                selected_key = keys[index]
                content = nested_dict[selected_key]
                if output_format == 'text':
                    result = f"{selected_key}:\n{'\n'.join(content)}"
                else:
                    result = {
                        selected_key: content
                    }
            else:
                result = "The specified entry was not found."
        else:
            result = "The specified entry was not found."

    return result


async def handle_OpenAIWrapperResponse(result):
    pretty_print(result)
    core_variable_name = result.get('variable_name', '')
    processed_values = result.get('processed_values', {})
    p_index = 0
    print(f"Processing {core_variable_name}...")
    if result.get('processing'):
        print("Processing is still in progress.")
        for processor, processor_data in processed_values.items():
            p_index += 1
            e_index = 0
            method_name = f"handle_{processor}"
            processor_value = processor_data.get('value', {})

            print(processor_data.get('extraction'))
            if 'extraction' in processor_data:
                print('found')
                for extraction_map in processor_data['extraction']:  # Adjusted to iterate over a list
                    print(f"-------------Processing {core_variable_name} with {method_name}...")
                    e_index += 1
                    variable_name = f"{p_index}_{e_index}_{core_variable_name}"
                    print(f"Processing {variable_name}")

                    try:
                        extraction_result = await access_data_by_reference(extraction_map, processor_value)  # Changed variable name from 'result' to 'extraction_result' to avoid overshadowing
                        pretty_print(extraction_result)
                        print(f"==================================================")
                    except Exception as ex:
                        # Log the error and continue with the next extraction_map
                        print(f"Error processing {variable_name} with {method_name}: {ex}")


async def sample_processor_structure(sample_content):
    return_params = {
        'variable_name': 'SAMPLE_DATA_1001',
        'processors':
            [
                {
                    'processor': 'get_markdown_asterisk_structure',
                    'depends_on': 'content',
                    "extraction": [
                        {
                            "key_identifier": "nested_structure",
                            "key_index": 1,
                            "output_type": "text"
                        },
                        {
                            "key_identifier": "nested_structure",
                            "key_index": "",
                            "output_type": "dict"
                        },
                        {
                            "key_identifier": "nested_structure",
                            "key_index": 3,
                            "output_type": "dict"
                        }
                    ]
                },
            ],
    }

    processor = AiOutput(sample_content)
    processed_content = await processor.process_response(return_params)
    pretty_print(processed_content)


async def main():
    sample_data = get_sample_data(app_name='automation_matrix', data_name='ama_example_new', sub_app='ama_ai_output_samples')
    # print(f"Sample Data:\n{sample_data}\n")
    result = await local_post_processing(sample_data)
    await handle_OpenAIWrapperResponse(result)


if __name__ == "__main__":
    asyncio.run(main())
