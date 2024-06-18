from knowledge.doc_manipulation.text_classification.text_classifier import (
                                                                            TextManipulation,
                                                                            process_steps,
                                                                            LineIdentifier
                                                                            )

# Identifying Our Identifiers

idf = LineIdentifier()

# Forming identifier for starting chapter headers
idf.ADD_AND([
    ('starts_with', {'equals': 'Chapter ', 'case_sensitive': True}),
    ('ends_with_digit', {'equals': True})
])

steps = [{"step": "clean_document"},
         {"step": "limit_consecutive_empty_lines"},
         {'step': 'wrap_line_with_markers', 'kwargs': {'line_identifier': idf, 'start_marker_lines': ['\n', '\n', "="*80], 'end_marker_lines': ["-"*65, '\n','\n']}},
         ]

if __name__ == "__main__":
    with open(r'C:\Users\jatin\OneDrive\Desktop\work 2\2024\armani\ame-team\knowledge\doc_manipulation\data\ama_raw_text.txt', 'r', encoding='utf-8') as file:
        document = file.read()
    updated = process_steps(steps, document)
    print(updated)

    # 1.2
    # Impairment, Disability, and
