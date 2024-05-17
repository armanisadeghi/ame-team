
blog_processing_asterisk_sample = {
    'variable_name': 'SAMPLE_DATA_1001',
    'processors':
        [
            {
                'processor': 'get_markdown_asterisk_structure',
                'depends_on': 'content',
                'args': {
                    "primary_broker": "SAMPLE_DATA_1001",
                    "extractors": [
                        {
                            "name": "access_data_by_reference",
                            "broker": "BLOG_IDEA_1",
                            "key_identifier": "nested_structure",
                            "key_index": 1,
                            "output_type": "text",
                        },
                        {
                            "name": "access_data_by_reference",
                            "key_identifier": "nested_structure",
                            "key_index": 2,
                            "output_type": "text",
                            "broker": "BLOG_IDEA_2",
                        },
                        {
                            "name": "access_data_by_reference",
                            "key_identifier": "nested_structure",
                            "key_index": 3,
                            "output_type": "text",
                            "broker": "BLOG_IDEA_3",
                        },
                        {
                            "name": "access_data_by_reference",
                            "broker": "BLOG_IDEA_4",
                            "key_identifier": "nested_structure",
                            "key_index": 4,
                            "output_type": "text",

                        },
                        {
                            "name": "access_data_by_reference",
                            "broker": "BLOG_IDEA_5",
                            "key_identifier": "nested_structure",
                            "key_index": 5,
                            "output_type": "text",
                        }
                    ]
                }
            },

        ],
}

asterisk_processing_nested_key_extraction = {
    'variable_name': 'SAMPLE_DATA_1001',
    'processors':
        [
            {
                'processor': 'get_markdown_asterisk_structure',
                'depends_on': 'content',
                "extractors": [
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

ama_medical_processing_sample = {
        'variable_name': 'SAMPLE_DATA_1001',
        'processors':
            [
                {
                    'processor': 'get_classified_markdown',
                    'depends_on': 'content',
                    'args': {
                        "primary_broker": "SAMPLE_DATA_1001",
                        "extractors": [
                            {
                                "name": "get_items_from_classified_markdown_sections",
                                "broker": "IMPAIRMENT_TABLE_RAW",
                                "classified_line_type": "table",
                                "search": "Impairment Code"
                            },
                            {
                                "name": "get_items_from_classified_markdown_sections",
                                "broker": "DATES_RAW",
                                "classified_line_type": "entries_and_values",
                                "search": "Date of birth"
                            },
                            {
                                "name": "get_items_from_classified_markdown_sections",
                                "broker": "OCCUPATION_RAW",
                                "classified_line_type": "header_with_bullets",
                                "search": "Occupational Code"
                            },
                            {
                                "name": "extract_list_table",
                                "broker": "IMPAIRMENTS_TABLE_CLEANED",
                                "column_index": 0,
                                "row_index_start": 2,
                            },
                            {
                                "name": "extract_integers_from_string",
                                "broker": "OCCUPATION_CODE",
                                "args": {"broker": True, "broker_name": "OCCUPATION_RAW",
                                         'target_index_or_name_in_broker': 1},
                                "value_to_be_processed": None,
                                "result_type": 'single'
                            },
                            {
                                'name': 'transform_table_data_format',
                                'broker': 'IMPAIRMENT_NUMBERS',
                                'args': {"broker": True, "broker_name": "IMPAIRMENTS_TABLE_CLEANED"},
                                'table': None,
                                'field_mappings': {"Impairment Code": "impairment_number",
                                                   "Whole Person Impairment": "wpi", "Industrial %": "industrial"},
                                'default_values': {"side": None, "ue": None, "digit": None, "le": None, "pain": 0},
                                'field_type_conversion': {'wpi': int, "industrial": int}
                                # This is for extracting the specified data type from the field value
                            },
                            {
                                "name": "extract_key_pair_from_string",
                                "broker": "AGE",
                                "args": {"broker": True, "broker_name": "DATES_RAW",
                                         'target_index_or_name_in_broker': 0},
                                "value_to_be_processed": None,
                                "result_datatype": int
                            },
                            {
                                'name': 'match_dates_from_string',
                                'broker': 'DOB',
                                'args': {"broker": True, "broker_name": "DATES_RAW",
                                         'target_index_or_name_in_broker': 0},
                                "result_type": 'single',

                            },
                            {
                                'name': 'match_dates_from_string',
                                'broker': 'DOI',
                                'args': {"broker": True, "broker_name": "DATES_RAW",
                                         'target_index_or_name_in_broker': 1},
                                "result_type": 'single'
                            }
                        ]
                    }
                },

            ],
    }