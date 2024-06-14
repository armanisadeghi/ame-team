from knowledge.doc_manipulation.text_classification.text_classifier import (
                                                                            TextManipulation,
                                                                            process_steps,
                                                                            LineIdentifier
                                                                            )

# Identifying Our Identifiers

idf = LineIdentifier()

# Forming identifier for starting chapter headers
idf.ADD_AND([
    ('startswith', {'equals': 'chapter ', 'case_sensitive': True}),
    ('ends_with_digit', {'equals': True})
])
chapter_line_identifier = idf.identifier
idf.reset_identifier()

# Forming identifier for Extracting References
idf.ADD_AND([
    ('startswith', {'strip': True, 'equals': 'references', 'case_sensitive': True}),
    ('endswith', {'strip': True, 'equals': 'references', 'case_sensitive': True})
])
start_reference_idf = idf.identifier
idf.reset_identifier()

idf.ADD_AND([
    ('startswith', {'equals': '====='})
])
end_reference_idf = idf.identifier
idf.reset_identifier()

# Forming identifier for chapter start
idf.ADD_AND([
    ('startswith', {'equals': '====='})
])
chapter_start_idf = idf.identifier
idf.reset_identifier()

# Forming Examples start and end identifiers
idf.ADD_AND([
    ('startswith', {'equals': 'Example'})
])
example_start_idf = idf.identifier
idf.reset_identifier()

idf.ADD_AND([
    ('startswith', {'equals': 'Comment:'})
])
example_end_idf = idf.identifier
idf.reset_identifier()

# Forming Index start
idf.ADD_AND([
    ('startswith', {'equals': 'Index'})
])
add_index_line_marker_idf = idf.identifier
idf.reset_identifier()

# Forming Line starting with tab
idf.ADD_AND([
    ('startswith', {'equals': '\t'})
])
line_starting_with_tab_idf = idf.identifier
idf.reset_identifier()

# Forming Introduction start identifier
idf.ADD_AND([
    ('startswith', {'equals': 'Introduction', 'case_sensitive': True})
])
intro_start_idf = idf.identifier
idf.reset_identifier()

# Forming History start identifier
idf.ADD_AND([
    ('startswith', {'equals': 'History', }),
    ('endswith', {'equals': 'History',})
])
hist_start_idf = idf.identifier
idf.reset_identifier()

# Forming identifier for if line is numbers and spaces
idf.ADD_AND([
    ('char_count', {'less_than': 7, }),
    ('is_digit_and_spaces', {'equals': True})
])
numbers_and_spaces_idf = idf.identifier
idf.reset_identifier()


# Forming identifier for if line has guides to the eval .....
idf.ADD_AND([
    ('contains', {'equals': "Guides to the Evaluation of Permanent Impairment", }),
])

primary_line_idf_guides = idf.identifier
idf.reset_identifier()

# Forming header for chapter line
idf.ADD_AND([
    ('contains_regex_pattern', {'equals': r'^\d+(\.\d+)*[a-d]?'})
])
chapter_headline_idf = idf.identifier

idf.reset_identifier()


steps = [{"step": "clean_document"},
         {"step": "limit_consecutive_empty_lines"},
         {'step': 'wrap_line_with_markers', 'kwargs': {'line_identifier': chapter_line_identifier, 'start_marker_lines':['\n', '\n', "="*80], 'end_marker_lines':["-"*65, '\n','\n']}},
         {"step": "limit_consecutive_empty_lines", 'kwargs': {'max_empty_lines': 1}},
         {'step': 'add_dynamic_markers_multiline', 'kwargs': {'start_line_identifier': start_reference_idf, 'end_line_identifier': end_reference_idf, 'start_marker_list': ['--reference--', '\n'], 'end_marker_list': ['\n', '--end reference--', '\n'], 'end_marker_position': 'before'}},
         {'step': 'extract_between_markers', 'kwargs': {'start_marker': '--reference--', 'end_marker': '--end reference--'}},
         {'step': 'shift_and_join_lines_by_primary_identifier', 'kwargs': {'primary_identifier': chapter_line_identifier, "distance": 3, 'direction': 'down', 'keep_line_after_moving': False, 'join_with' : ': '}},
         {'step': 'add_dynamic_markers_multiline', 'kwargs': {'start_line_identifier': example_start_idf, 'end_line_identifier': example_end_idf, 'start_marker_list': ['--example--', '\n'], 'end_marker_list': ['\n', '--end example--', '\n'], 'end_marker_position': 'after', 'max_lookup_for_end_identifier': 20}},
         {'step': 'extract_between_markers', 'kwargs': {'start_marker': '--example--' , 'end_marker':'--end example--' }},

         {'step': "extract_short_sections", 'kwargs': {'line_count': 25, "char_count": 100}},
         {'step': "extract_short_sections", 'kwargs': {'line_count': 8, "char_count": 50}},
         {'step': "extract_short_sections", 'kwargs': {'line_count': 5, "char_count": 14}},

         {'step': "add_dynamic_markers_single_line", 'kwargs': {'line_identifier': add_index_line_marker_idf, "marker_lines": ['-------- INDEX START -------'], }},
         {'step': "shift_and_join_lines", 'kwargs': {'line_identifier': line_starting_with_tab_idf, 'distance': 1, 'direction': 'up', 'join_with' : "--joined--"}},
         {'step': "add_dynamic_markers_single_line", 'kwargs': {'line_identifier': intro_start_idf, "marker_lines": ['-------- INTRODUCTION START -------'], }},

         {'step': 'replace_items', 'kwargs': {'replacements': [{'text':'\t', 'replacement': '\n'}]}},
         {'step': "add_dynamic_markers_single_line", 'kwargs': {'line_identifier': hist_start_idf, "marker_lines": ['-------- INTRODUCTION START (History)-------'], }},
         {'step': 'replace_items', 'kwargs': {'replacements': [{'text':'- ' , 'replacement': ''}]}},
         {'step': 'modify_surrounding_lines', 'kwargs': {'primary_line_identifier': primary_line_idf_guides, 'secondary_line_identifier': numbers_and_spaces_idf, 'discover_above_lines':2, 'discover_below_lines': 2 , 'action': 'delete_line'}},
         {'step': 'insert_breaks', 'kwargs': {'primary_line_identifier': chapter_headline_idf, 'ignore_start_pattern': r'================================================================================' ,'ignore_end_pattern':r'-------INTRODUCTION'}}
         ]

if __name__ == "__main__":
    with open(r'C:\Users\jatin\OneDrive\Desktop\work 2\2024\armani\ame-team\knowledge\doc_manipulation\data\ama_raw_text.txt', 'r', encoding='utf-8') as file:
        document = file.read()
    updated = process_steps(steps, document)
    print(updated)

    # 1.2
    # Impairment, Disability, and
