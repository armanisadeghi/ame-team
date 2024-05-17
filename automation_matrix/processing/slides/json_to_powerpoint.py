import json
import os
import uuid
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from ame_team.settings.base import BASE_DIR
from common import get_sample_data, pretty_print, print_link

output_temp_dir = "temp/app_outputs/slides"
output_dir = os.path.join(BASE_DIR, output_temp_dir)
short_uuid_part = lambda: str(uuid.uuid4())[:8]


def json_to_presentation(json_data, file_name_root='output', output_dir='.'):
    """A simple slide creator that expects the json_data to be a list of dictionaries with 'title' and 'points' keys."""
    prs = Presentation()

    for item in json_data:
        slide = prs.slides.add_slide(prs.slide_layouts[1])  # Assumes layout 1 has a title and content placeholder
        title_shape = slide.shapes.title
        title_shape.text = item['title']

        # Customize the title font
        title_text_frame = title_shape.text_frame
        title_text_frame.text = item['title']
        for paragraph in title_text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(32)

        content_shape = slide.placeholders[1]  # This assumes placeholder 1 is the content placeholder for bullet points
        text_frame = content_shape.text_frame
        text_frame.clear()  # Clear existing content in the placeholder

        # Add bullet points and customize font
        for point in item['points']:
            p = text_frame.add_paragraph()
            p.text = point
            p.level = 0  # Set bullet point level (0 for top level)
            p.font.size = Pt(24)

        # Set a light blue background color for the slide
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(r=173, g=216, b=230)  # Light blue color, similar to 'lightblue' HTML color

    output_file_name = f"{file_name_root}_{short_uuid_part()}.pptx"
    save_path = os.path.join(output_dir, output_file_name)

    prs.save(save_path)
    print_link(save_path)
    return save_path


if __name__ == '__main__':
    # sample_slide_content_json
    sample_json = get_sample_data(app_name='automation_matrix', data_name='small_2_slide_sample', sub_app='slide_source_samples')  # Get sample API response
    pretty_print(sample_json)

    save_path = json_to_presentation(sample_json)
