import json
import re
import ast
from bs4 import BeautifulSoup
from typing import Dict
from collections import defaultdict
from typing import List, Union
import asyncio

from automation_matrix.processing._dev_simulation import simulate_workflow_after_processing
from common import vcprint, pretty_print, get_sample_data
from automation_matrix.processing.markdown.organizer import Markdown
from automation_matrix.processing.processor import ProcessingManager

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

        # Debug Print 1
        vcprint(verbose=verbose, data=self.return_params, title="Return Params")

        self.source_info = self.return_params.get('source_info', '')
        self.variable_name = self.return_params.get('variable_name', '')
        if not isinstance(self.variable_name, str):
            self.variable_name = str(self.variable_name)

        self.processed_content = {
            'signature': 'AiWrapperResponse',
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
            elif processor['depends_on'] not in processor_names and processor['depends_on'] not in ['content',
                                                                                                    'raw_api_response']:
                vcprint(verbose=True,
                        data=f"\n[Warning!] '{processor}' has an invalid dependency on '{processor['depends_on']}'. Changing to 'content'.\n",
                        color='red', style='bold')
                processor['depends_on'] = 'content'

        # ------- Get the processors in teh right order, regardless of what order they were provided in. -------
        # This is a good example of how I like to code. It's MY job to fix a mistake that the other programmer or the user made, if it doesn't hurt anything.
        # Order the processors based on their dependencies - Don't want to run something, before the dependencies are met.
        def add_processor_if_dependency_met(processor, ordered, unprocessed):
            if processor['depends_on'] == 'content' or any(
                    dep['processor'] == processor['depends_on'] for dep in ordered):
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

        # Debug Print 2:
        vcprint(verbose=verbose, data=ordered_processors, title="Ordered Processors", color='yellow', style='bold')
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

            input_data = self.content if depends_on in ['content', 'raw_api_response'] else processed_names.get(
                depends_on)

            processor = getattr(self, processor_name, None)
            if not processor:
                print(f"No processor found for '{processor_name}'")
                continue

            try:
                # debug print 3
                vcprint(verbose=verbose, data=input_data, title=f"Input Data for '{processor_name}'", color='blue',
                        style='bold')
                vcprint(verbose=verbose, data=args, title=f"Args for '{processor_name}'", color='blue', style='bold')

                output_data = await processor(input_data, args) if asyncio.iscoroutinefunction(
                    processor) else processor(input_data, args)

                # debug print 4
                vcprint(verbose=verbose, data=output_data, title=f"Output Data for '{processor_name}'", color='green',
                        style='bold')

            except Exception as e:
                print(f"Error processing '{processor_name}': {e}")
                output_data = "error"

            self.processed_content['processed_values'][processor_name] = {
                'value': output_data,
                'depends_on': depends_on,
                'args': args,
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

    async def get_markdown_asterisk_structure(self, content: str, args=None) -> Dict[str, Union[str, List[str]]]:
        processor = Markdown(style='asterisks')
        asterisk_structure_results = await processor.process_and_extract(content, args)

        return asterisk_structure_results

    # Creates groups from the data. For example, it can combine every two or three items into a group so they go together.
    async def data_groups(self, data, args=None):
        if args:
            parts_count = args.get('parts_count', 2)
        else:
            parts_count = 2

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
    async def clean_groups(self, data, args=None):
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

    async def extract_urls(self, content: str):
        """This extracts normal urls, those include custom protocols too"""
        from urlextract import URLExtract  # Please add this library (pip install urlextract)
        extractor = URLExtract()
        urls = extractor.find_urls(content)
        return urls

    async def extract_mailto_links(self, content: str):
        pattern = re.compile(r'mailto:[^\s<>]+\b')
        matches = pattern.findall(content)
        return matches

    async def extract_telephone_links(self, content: str):
        pattern = re.compile(r'tel:\+?\d+[\d-]*')
        matches = pattern.findall(content)
        return matches

    async def extract_tables(self, text: str) -> List[dict]:
        tables = text.strip().split('\n\n')
        all_tables = []

        for table in tables:
            lines = table.strip().split('\n')
            header_line_index = next((i for i, line in enumerate(lines) if '|' in line), None)

            if header_line_index is None:
                continue

            headers = [header.strip() for header in lines[header_line_index].split('|')[1:-1]]

            rows = []
            for line in lines[header_line_index + 2:]:
                if '|' in line:
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    if cells:
                        rows.append(dict(zip(headers, cells)))

            all_tables.append(rows)

        return all_tables

    async def extract_python_dicts(self, content: str):
        if not isinstance(content, str):
            content = str(content)

        dicts = []  # return object
        check_after_index = 0

        for index, char in enumerate(content):
            # this step is important because check_after_index
            # tells us at what index to continue looking for a dictionary
            if not index >= check_after_index:
                continue

            # Possible Dict start detected at this point
            if char == "{":
                index_ = index  # create a copy of original index

                for char_ in content[index:]:  # start to check chars after {

                    index_ += 1
                    if char_ == "}":  # check for possible end

                        # Possible end reached
                        try:
                            # Try by evaluating the Possible dict
                            dict_ = ast.literal_eval(content[index:index_].strip())
                            # Checks if the dict is not empty and is a dict not a set
                            if isinstance(dict_, dict) and dict_:
                                dicts.append(dict_)
                            # After a dict is completely detected whether it be a nested one or a normal one
                            # Now we have to update check_after_index here so in the next iterations we ignore the already detected dict.

                            check_after_index = int(index_)
                        except Exception:
                            pass

        return dicts

    # Asked in doc
    async def extract_lists(self, text: str) -> List[List[str]]:
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

    async def extract_html_snippet(self, content: str):
        if not isinstance(content, str):
            content = str(content)

        # Getting possible snippets
        possible_snippets = await self.extract_code_snippets(content)

        if possible_snippets.get('html'):
            return possible_snippets.get('html')

    async def extract_html_from_text(self, content: str):

        from html.parser import HTMLParser
        class HTMLValidator(HTMLParser):
            def __init__(self):
                super().__init__()
                self.open_tags = []

            def handle_starttag(self, tag, attrs):
                # Keep track of open tags
                self.open_tags.append(tag)

            def handle_endtag(self, tag):
                # Remove corresponding open tag
                if self.open_tags and self.open_tags[-1] == tag:
                    self.open_tags.pop()
                else:
                    # If there's no corresponding open tag, HTML is invalid
                    self.error(f"Closing tag '{tag}' found without corresponding opening tag.")

            def error(self, message):
                # Reset the list of open tags on error
                self.open_tags = []

        # initialize empty return list
        html = []
        check_after_index = 0
        if not isinstance(content, str):
            content = str(content)

        for index, char in enumerate(content):
            # this step is important because check_after_index
            # tells us at what index to continue looking for a html
            if not index >= check_after_index:
                continue

                # Possible html/element start detected at this point
            if char == "<":
                index_ = index  # create a copy of original index
                for char_ in content[index:]:  # start to check chars after <
                    index_ += 1
                    if char_ == ">":  # check for possible end
                        # Possible end reached
                        try:
                            parser = HTMLValidator()
                            parser.feed(content[index:index_].strip())
                            if not parser.open_tags:
                                html.append(content[index:index_])  # Append the HTML snippet to the list
                                check_after_index = int(index_)
                                break  # Exit the inner loop once HTML is found
                            # After a html is completely detected
                            # Now we have to update check_after_index here so in the next iterations we ignore the already detected html.
                        except Exception:
                            pass

        html_snippets = html

        return html_snippets

    # Asked in doc
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

    # Asked in doc
    async def extract_nested_markdown_entries(self, text: str) -> Dict[str, List[str]]:
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

    # Asked in doc
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

    # Asked in doc
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

    async def extract_fill_in_blanks(self, text: str, *args) -> List[str]:
        """
        Identifies sentences with fill-in-the-blank structures.

        :param text: String containing the text to be processed.
        :return: List of sentences with blanks.
        """
        blank_pattern = '[^\\.\\?\\!]*__+[^\\.\\?\\!]*[\\.|\\?|\\!]'
        final_pattern = re.findall(blank_pattern, text)
        return final_pattern

    async def extract_multiple_choice_questions(self, text: str, *args) -> List[str]:
        """
        Identifies and extracts multiple-choice questions.

        :param text: String containing the text to be processed.
        :return: List of multiple-choice questions with options.
        """
        mcq_pattern = '(?:\\n|^).*\\?\\n(?:\\s*[a-zA-Z]\\)\\s.*\\n)+'
        final_pattern = [mcq.strip() for mcq in re.findall(mcq_pattern, text, re.MULTILINE)]
        return final_pattern

    # Asked in doc
    async def extract_paragraphs(self, text: str, *args) -> List[str]:
        """
        Extracts paragraph-style text content.

        :param text: String containing the text to be processed.
        :return: List of paragraphs.
        """
        # TODO Not working as intended because it's pulling the list of LSIs, which is definitely not a paragraph.
        #  We need to be clear what a paragraph is
        paragraphs = re.split('\\n{2,}', text)
        extracted_paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
        print(f"Paragraphs:\n{extracted_paragraphs}\n")
        return extracted_paragraphs

    async def extract_html_markdown_structured_text(self, text: str, *args) -> Dict[str, List[str]]:
        """
        Specifically looks for and extracts structured text marked as HTML or Markdown.

        :param text: String containing the text to be processed.
        :return: Dictionary with HTML and Markdown content categorized.
        """
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
        return structured_content

    # Asked in doc
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
        return word_count

    # Not sure what to do
    async def extract_markdown(self, text: str, *args):
        # Implement markdown extraction logic
        pass

    async def extract_json_from_text(self, content: str, *args):
        """
        So the logic here will be to find valid json objects directly from raw plain text.
        """
        # Implement JSON extraction logic
        char_map = {'{': '}', '[': ']'}
        if not isinstance(content, str):
            content = str(content)

        valid_json_objs = []  # return object
        check_after_index = 0

        for index, char in enumerate(content):
            # this step is important because check_after_index
            # tells us at what index to continue looking for a JSON
            if not index >= check_after_index:
                continue

            # Possible Json start detected at this point
            if char == "{" or char == "[":
                end_brac = char_map[char]
                index_ = index  # create a copy of original index

                for char_ in content[index:]:  # start to check chars after { or [
                    index_ += 1
                    if char_ == end_brac:  # check for possible end

                        # Possible end reached
                        try:
                            # Try by evaluating the Possible JSON
                            json_ = json.loads(content[index:index_].strip())
                            valid_json_objs.append(json_)
                            # Now we have to update check_after_index here so in the next iterations we ignore the already detected Json/dict item.
                            check_after_index = int(index_)

                        except Exception:
                            pass

        return valid_json_objs

    async def extract_json_snippet(self, content: str, *args):
        snippets = await self.extract_code_snippets(content)
        if snippets.get('json'):
            return snippets.get('json')

    # Answer in docstring please
    async def extract_code(self, text: str, *args):
        """
        What to do here exactly? Is the language name be going to be specified in the args, or we have to extract direct from AI's response
        if we need to extract from AI response that is already implemented
        """
        # Implement code extraction logic
        pass

    async def extract_python_code_snippet(self, text: str, *args):
        """
        This currently looks for snippets that are identified as python snippets
        """
        python_code_list = []
        # 1 st checkpoint: checks for if there are snippets that can be directly detected as python code
        snippets = await self.extract_code_snippets(text)
        if snippets.get('python'):
            python_code_list.extend(python_code_list)

        return python_code_list

    # Answer in docstring please
    async def extract_python_code_from_text(self, text: str, *args):
        """
        We can identify python code from raw text also , but this doesn't include incomplete code or code with syntax erros
        Please leave your suggestions here
        """
        # Please make sure that the escape characters are unescaped, otherwise may lead to errors in final code

        import parso  # Needs parso library (pip install parso)
        final_code = ""
        parsed = parso.parse(text)
        for module in parsed.children:

            error_lines_identifier = ['error_node', 'error_leaf']
            if str(module.type) in error_lines_identifier:
                continue

            final_code = final_code + module.get_code()

        return final_code

    # Answer in docstring
    async def extract_code_remove_comments(self, code: str, **kwargs):
        """
        What to do here exactly? Is the language name be going to be specified in the args
        Do we create regex patterns for all possible comment patterns or what ?
        I still wrote the function , please recommend if this needs improvement. This function supports the function just below this
        """

        alt_names = {
            "python": ['py', 'python', 'python3'],
            "javascript": ['js'],
            "java": ['java'],
            "go": ['go', 'golang'],
            "c": ['c'],
            "cpp": ['c++', 'cpp', 'cplusplus'],
            "csharp": ['c#', 'csharp'],
            "php": ['php'],
            "ruby": ['ruby'],
            "swift": ['swift'],
            "kotlin": ['kotlin'],
            "rust": ['rust'],
            "typescript": ['ts', 'typescript'],
            "perl": ['perl'],
            "scala": ['scala'],
            "bash": ['bash', 'shell', 'sh'],
            'r': ['r'],
            'matlab': ['matlab'],
            'sql': ['sql'],
            'dart': ['dart'],
        }

        comment_patterns = {
            "python": re.compile(r'#.*|\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\"'),
            "javascript": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "java": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "c": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "cpp": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "csharp": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "php": re.compile(r'//.*|#.*|/\*[\s\S]*?\*/'),
            "ruby": re.compile(r'#.*|=begin[\s\S]*?=end'),
            "swift": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "kotlin": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "rust": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "typescript": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "perl": re.compile(r'#.*|=begin[\s\S]*?=cut'),
            "scala": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "bash": re.compile(r'#.*|:\s*\'.*?\''),  # Bash multi-line comments with : '
            "r": re.compile(r'#.*'),  # R only has single-line comments
            "matlab": re.compile(r'%.*|%\{[\s\S]*?%\}'),
            "sql": re.compile(r'--.*|/\*[\s\S]*?\*/'),
            "dart": re.compile(r'//.*|/\*[\s\S]*?\*/'),
            "_general": re.compile(r'//.*|/\*[\s\S]*?\*/'),  # General pattern for unrecognized languages
        }

        language = kwargs.get('lang')

        if language is not None:
            language = language.lower()
            if language not in alt_names.keys():
                # finding the key incase not found
                for lang, alt_name in alt_names.items():
                    if language in alt_name:
                        language = lang
                        break

        if language is not None:
            pattern = comment_patterns.get('_general')
        else:
            pattern = comment_patterns.get(language)

        result_code = pattern.sub("", code)

        return result_code.strip()

    async def extract_code_remove_comments_from_snippets(self, text: str):
        """
        :param text: Direct code from AI response. The language is auto-detected and comments are removed
        :return: returns a dict of various languages, values are list of code (after removing comments)
        """
        snippets = await self.extract_code_snippets(text)
        output = defaultdict(list)

        for lang, code_list in snippets.items():
            kwargs = {'lang': lang}
            for code in code_list:
                code_without_comments = await self.extract_code_remove_comments(code, **kwargs)
                output[lang].append(code_without_comments)

        return output

    # Answer in docstring
    async def extract_from_json_by_key(self, text: str, **kwargs):
        """
        What to do here exactly . Extract json from either AI response or from raw text and then find the key
        Please clarify here. (Json from text and snippet is already implemented)
        I still wrote the function , Please make recommendations for this
        """
        json_snippets = []
        key = kwargs.get('key')

        if key is None:
            return []

        # 1 st checkpoint: Checked whether we can get any json by extracting code snippets
        snippets = await self.extract_code_snippets(text)
        if snippets.get('json') is not None:
            json_snippets.extend(snippets.get('json'))

        # 2 nd checkpoint: Tries to load JSON by text , if json is not a snippet
        text_parsed_json_snippets = await self.extract_json_from_text(text)
        if snippets:
            json_snippets.extend(text_parsed_json_snippets)

        # 3 rd part , here the actual search happens tries to get results by the key
        results = []

        def _search_key(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == key:
                        results.append(v)
                    elif isinstance(v, (dict, list)):
                        _search_key(v)
            elif isinstance(obj, list):
                for item in obj:
                    _search_key(item)

        for snippet in json_snippets:
            try:
                snippet_dict = json.loads(snippet)
                _search_key(snippet_dict)
            except json.JSONDecodeError:
                pass

        return results

    # Answer in docstring
    async def extract_from_outline_by_numbers(self, text: str, *args):
        """
        what to do here exactly ? Please try to give an example here
        """
        # Implement extraction from an outline by numbers
        pass

    # Answer in docstring
    async def remove_first_and_last_paragraph(self, text: str, *args):
        """
        What's a paragraph exactly?  Are there any rules to check for. Try to give examples of input and output
        """
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

    async def get_classified_markdown(self, content, args):
        from automation_matrix.processing.markdown.classifier import get_classify_markdown_section_list
        sections = await get_classify_markdown_section_list(content)
        structure = {
            "sections": sections,
            "brokers": []
        }
        processor = Markdown(style='asterisks')
        results = processor.handle_extraction(structure, args.get('extractors'))
        return results


def print_initial_and_processed_content(processed_content):
    print("========================================== Initial Content ==========================================")
    print(processed_content['value'])
    print(processed_content['processed_values'])
    print("\nProcessed Steps:")

    for step_name, step_data in processed_content['processed_values'].items():
        print(f"\n========================================== {step_name} ==========================================\n")

        output = step_data['value']
        if isinstance(output, dict):
            for key, value in output.items():
                print(f"--------- {key}: ---------")
                if isinstance(value, list):
                    for item in value:
                        print(f"\n{item}")
                else:
                    print(f"\n{value}")
        elif isinstance(output, list):
            for item in output:
                print(f"  - {item}")
        else:
            print(f"  {output}")

        print("-" * 25)
        print("\nDepends on:", step_data['depends_on'])
        print("\nArgs:", step_data['args'])


async def local_post_processing(sample_api_response, sample_return_params):
    pretty_print(sample_api_response)
    print('==========' * 10)
    processor = AiOutput(
        sample_api_response)  # This needs to be the sample content from the API response (Not just plain content, but processed content. I can get more examples, if needed.)
    processed_content = await processor.process_response(
        sample_return_params)  # This is the end result of all of the processing steps.

    vcprint(verbose=True, data=processed_content, title="Processed Content", color='blue', style='bold')

    return processed_content


# I added this as a way to simulate the AI Wrapper response. This is a way to test the processing without having to run the AI Wrapper.
# Use this if you want to just get basic content and call the system. All it does is put "content" into the "value" key of the response.
# This, plus the return_params is all you need to test the processing. (But if the content is already formatted like this, you don't need it.)
def format_non_formatted_data(content):
    simulated_ai_wrapper_response = {
        'signature': 'AIWrapperResponse',
        'processing': False,
        'value': content,
        'processed_values': {}
    }
    return simulated_ai_wrapper_response


if __name__ == "__main__":
    sample_api_response = get_sample_data(app_name='automation_matrix', data_name='ama_medical_report_sample',
                                          sub_app='ama_ai_output_samples')  # Get sample API response
    # sample_return_params = get_sample_data(app_name='automation_matrix', data_name='blog_processing_asterisk_sample', sub_app='processor_settings_samples')  # Get sample return params structure
    sample_return_params = get_sample_data(app_name='automation_matrix', data_name='ama_medical_processing_sample',
                                           sub_app='processor_settings_samples')  # Get sample return params structure

    # This is the final results from all processors and all extractors used.
    processing_and_extraction_results = asyncio.run(local_post_processing(sample_api_response, sample_return_params))

    # This is where the processor is done, but to simulate the next steps, I've created some functions that mimic what the workflow does with the data.
    # But in reality, as long as your brokers are EXACTLY where they're supposed to be, there is nothing else to do.
    wf_simulation_results = simulate_workflow_after_processing(processing_and_extraction_results)
    pretty_print(wf_simulation_results)
