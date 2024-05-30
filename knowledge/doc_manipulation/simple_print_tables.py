from docx import Document


def extract_tables(doc_path):
    doc = Document(doc_path)
    for i, table in enumerate(doc.tables):
        print(f"Table {i + 1}:")

        for row in table.rows:
            row_data = []

            for cell in row.cells:
                row_data.append(cell.text.strip())

            print(", ".join(row_data))

        print("-" * 40)


doc_path = r"D:\a_starter\ama\ama_guide_2.docx"

extract_tables(doc_path)
