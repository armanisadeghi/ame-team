import asyncio
import tabula  # Ensure tabula-py is installed and Java is installed
import os
from google_integrations.cloud_storage.storage_manager import upload_content_to_bucket

java_home = r"C:\Program Files\Java\jdk-21"  # Update this path to your JDK installation
os.environ["JAVA_HOME"] = java_home

# Add the path to the 'bin' directory of your Java installation to the PATH environment variable
os.environ["PATH"] = f"{java_home}\\bin;{os.environ['PATH']}"

output_dir = "table_output"
os.makedirs(output_dir, exist_ok=True)


def extract_tables_to_csv(filepath):
    try:
        # Output CSV files will be saved in the specified directory with the same base name as the PDF
        output_csv_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filepath))[0])
        tabula.convert_into(filepath, f"{output_csv_path}_table.csv", output_format="csv", pages='all')
        print(f"Tables extracted to CSV for {filepath}")
        return [f"{output_csv_path}_table.csv"]
    except Exception as e:
        print(f"Error extracting tables from {filepath}: {e}")
        return []


def handle_destination(filepath, destination_type, destination_folder=None):
    # Extract the basename and replace spaces with underscores
    filename = os.path.basename(filepath).replace(' ', '_')

    if destination_type == "bucket":
        pass  # Disabled for this purpose.
        # with open(filepath, 'rb') as file:
        # Make sure the bucket name is correct and does not include 'gs://' prefix
        # upload_content_to_bucket(file, "ai-matrix-engine-text", f"table_output/{filename}", content_type='text/plain')
        # print(f"Uploaded {filename} to Firebase")
    elif destination_type == "local_folder" and destination_folder is not None:
        local_path = os.path.join(destination_folder, filename)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure the destination folder exists
        with open(filepath, 'rb') as f_read, open(local_path, 'wb') as f_write:
            f_write.write(f_read.read())
            # Use the format for the clickable path, ensuring spaces in paths are handled
            clickable_path = f"file:///{local_path.replace('\\', '/')}"
            print(f"Data saved to {clickable_path}")


def process_pdf_for_tables(filepath, destination_type, destination_folder=None):
    csv_files = extract_tables_to_csv(filepath)
    for csv_file in csv_files:
        handle_destination(csv_file, destination_type, destination_folder)


if __name__ == "__main__":
    pdf_paths = [
        r"D:\OneDrive\Downloads\PD Rating Schedule.pdf",  # Example path
    ]

    destination_folder = r"D:\OneDrive\dev\PycharmProjects\aidream\photo_editing\utils_pdf\text_tables_output"

    for pdf_path in pdf_paths:
        process_pdf_for_tables(pdf_path, "local_folder", destination_folder)  # Change "local_folder" to "bucket" to upload to Firebase

# vasaro account: https://chat.openai.com/share/e/5ec5788d-8151-4c7e-ab1e-68d84a80b79f (But not that good)
