import subprocess
import os

# Function to convert Markdown file to DOCX
def convert_markdown_to_docx(markdown_file_path, output_docx_file_path):
    try:
        # Build the Pandoc command
        command = ['pandoc', markdown_file_path, '-o', output_docx_file_path]

        # Execute the command
        subprocess.run(command, check=True)

        print(f'Successfully converted {markdown_file_path} to {output_docx_file_path}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred while converting {markdown_file_path} to {output_docx_file_path}: {e}')

# Example usage
markdown_file_path = 'example.md'  # Path to your markdown file
output_docx_file_path = 'output.docx'  # Desired output DOCX file path

# Convert the file
convert_markdown_to_docx(markdown_file_path, output_docx_file_path)
