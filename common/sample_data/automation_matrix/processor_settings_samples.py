
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
