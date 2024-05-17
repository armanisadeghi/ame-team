# Marked as TODOS in ai_ouptut.py

**access_data_by_reference** , this needs to be removed , its a duplicate , currently in organizer.py

**handle_OpenAIWrapperResponse**: I think this one is no longer needed , as it was for simulating the extraction with return params


# Processors

**extract_code_snippets** : works

**get_markdown_asterisk_structure** : works for three styles asterisks, question_asterisks,  hashes

**data_groups**  and  **clean_groups** and **add_parts**: Works but the basic idea is to convert the list of plain_text to group of supplied part_count , default is 2. No idea of the exact use case then # Matches any 'text: ' pattern at the start of a string with clean groups , finally add common parts

**define_key_value_pairs** and **oai_fine_tuning_data** : The first one is to define key value pairs from the given keys and parts 

not sure what oai_fine_tuning does


**process_urls** : this works for generic urls with https:// or subdomain.x.com or x.com structure , currently missing many formats like window file paths , doesnt take the url params or paths, torrent urls , custom protocols , data urls, tel links, mailto links

**parse_table** doesn't extract tables

**identify_python_dictionaries** : this doesn't parse nested dictionaries. Also ignores values sometimes. Cannot process complex dictionaries

**extract_lists**: works but doesn't differentiate nested lists of different or same types

**parse_html_content** fails to extract properly in some simple html


**extract_markdown_entries** and **extract_nested_markdown_entries** not sure what it does, doesnt give any resp to me


**extract_latex_equations** works

**extract_plain_text** not yet tested


**identify_fill_in_blanks**: Works mostly sometimes print invalid items too, needs some improvement

**parse_multiple_choice_questions** works mostly

**extract_markdown** not yet checked

**identify_html_markdown_structured_text** not checked yet


**extract_prompts_questions** how does the input look like

**find_words_after_triple_quotes** whats the purpose for this?

**rest of the processors are not completed in ai_output file**  
