import json
# from docx import Document
from common import pretty_print


def get_cell_text(cell):
    """Extract all text from a cell, including from nested tables."""
    text_parts = []
    for paragraph in cell.paragraphs:
        text_parts.append(paragraph.text)
    for table in cell.tables:
        for row in table.rows:
            for cell in row.cells:
                text_parts.append(get_cell_text(cell))
    return ' '.join(text_parts).strip()


def tables_to_json(doc_path, json_path, table_limit=None):
    doc = Document(doc_path)

    tables_dict = {}
    actual_table_count = len(doc.tables)
    if table_limit is None or table_limit > actual_table_count:
        table_limit = actual_table_count

    for table_index, table in enumerate(doc.tables[:table_limit]):
        rows_as_keys = f"Table_{table_index + 1}_RowsAsKeys"
        cols_as_keys = f"Table_{table_index + 1}_ColsAsKeys"

        tables_dict[rows_as_keys] = {}
        tables_dict[cols_as_keys] = {}

        headers = [get_cell_text(cell) for cell in table.rows[0].cells]
        data_rows = table.rows[1:] if headers else table.rows
        if not headers:
            headers = [f"Column_{i}" for i in range(len(table.rows[0].cells))]

        for row in data_rows:
            row_key = get_cell_text(row.cells[0])
            row_data = {"Activity": row_key}
            for cell_index, cell in enumerate(row.cells[1:], start=1):
                cell_text = get_cell_text(cell)
                row_data[headers[cell_index]] = cell_text
            tables_dict[rows_as_keys][row_key] = row_data

        for header_index, header in enumerate(headers):
            tables_dict[cols_as_keys][header] = []
            for row in data_rows:
                cell_text = get_cell_text(row.cells[header_index])
                tables_dict[cols_as_keys][header].append(cell_text)

    json_data = json.dumps(tables_dict, indent=4, ensure_ascii=False)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)

    return json_data

def extract_rows_as_keys(doc, table_limit=None):
    tables_dict = {}
    actual_table_count = len(doc.tables)
    if table_limit is None or table_limit > actual_table_count:
        table_limit = actual_table_count

    for table_index, table in enumerate(doc.tables[:table_limit]):
        table_key = f"Table_{table_index + 1}_RowsAsKeys"
        tables_dict[table_key] = {}

        headers = [get_cell_text(cell) for cell in table.rows[0].cells]
        data_rows = table.rows[1:] if headers else table.rows
        if not headers:
            headers = [f"Column_{i}" for i in range(len(table.rows[0].cells))]

        for row in data_rows:
            row_key = get_cell_text(row.cells[0])
            row_data = {"Activity": row_key}
            for cell_index, cell in enumerate(row.cells[1:], start=1):
                cell_text = get_cell_text(cell)
                row_data[headers[cell_index]] = cell_text
            tables_dict[table_key][row_key] = row_data

    return tables_dict

def extract_columns_as_keys(doc, table_limit=None):
    tables_dict = {}
    actual_table_count = len(doc.tables)
    if table_limit is None or table_limit > actual_table_count:
        table_limit = actual_table_count

    for table_index, table in enumerate(doc.tables[:table_limit]):
        table_key = f"Table_{table_index + 1}_ColsAsKeys"
        tables_dict[table_key] = {}

        headers = [get_cell_text(cell) for cell in table.rows[0].cells]
        data_rows = table.rows[1:] if headers else table.rows
        if not headers:
            headers = [f"Column_{i}" for i in range(len(table.rows[0].cells))]

        for header_index, header in enumerate(headers):
            tables_dict[table_key][header] = []
            for row in data_rows:
                cell_text = get_cell_text(row.cells[header_index])
                tables_dict[table_key][header].append(cell_text)

    return tables_dict


def tables_to_json_rows(doc_path, json_path_rows, table_limit=None):
    doc = Document(doc_path)

    # Extract Rows as Keys
    rows_dict = extract_rows_as_keys(doc, table_limit)
    json_data_rows = json.dumps(rows_dict, indent=4, ensure_ascii=False)
    with open(json_path_rows, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data_rows)

    return json_data_rows


def tables_to_json_columns(doc_path, json_path_columns, table_limit=None):
    doc = Document(doc_path)

    columns_dict = extract_columns_as_keys(doc, table_limit)
    json_data_columns = json.dumps(columns_dict, indent=4, ensure_ascii=False)
    with open(json_path_columns, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data_columns)

    return json_data_columns


doc_path = r"D:\a_starter\ama\ama_guide_2.docx"
json_path_rows = r'/knowledgebase/utils/doc_manipulation/data/ama_guides_rows.json'
json_path_columns = r'D:\OneDrive\dev\PycharmProjects\aidream\common\utils\doc_manipulation\data\ama_guides_columns.json'

rows_json = tables_to_json_rows(doc_path, json_path_rows, table_limit=None)
