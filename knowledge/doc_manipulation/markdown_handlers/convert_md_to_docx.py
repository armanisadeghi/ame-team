import subprocess
import sys
import tempfile
import os


def convert_markdown_content_to_docx(markdown_content, output_docx_file_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmpfile:
        markdown_file_path = tmpfile.name
        tmpfile.write(markdown_content.encode('utf-8'))

    try:
        command = ['pandoc', markdown_file_path, '-o', output_docx_file_path]

        subprocess.run(command, check=True)

        print(f'Successfully created {output_docx_file_path}')
    finally:
        os.remove(markdown_file_path)

markdown_content = """
**Final Report: Summary of Impairment Evaluation and Injury Information**

**Patient Information:**
- Age: 46 years old
- Occupational Code: Driver (Specifically, Occupation Designator 5, "Truck Driver")
- Injury Circumstances: The patient was involved in a head-on collision while driving a company truck.

**Medical and Injury Details:**
- **MRI & X-Ray Evaluations:** Performed on the cervical spine, bilateral shoulders, and lumbar spine.
- **Medical History:** History of ACL reconstruction surgery on the left knee approximately 20 years ago. No prior similar injuries, other industrial injuries, or automobile accidents reported.
- **Medications:** The patient has been taking meloxicam and other pain medications.
- **Activities of Daily Living:** Reported difficulties due to symptoms.

**Impairment Summary:**

| Body Part       | DRE Category | Whole Person Impairment Percentage | Industrial % | Non-Industrial % | Specific Evidence Used                         |
|-----------------|--------------|------------------------------------|--------------|------------------|------------------------------------------------|
| Cervical Spine  | II           | 6%                                 | 85%          | 15%              | Tenderness, non-verifiable radicular complaints, MRI findings, difficulties with ADL |
| Right Shoulder  | N/A          | 5%                                 | 100%         | 0%               | Tenderness, loss of range of motion, MRI findings            |
| Left Shoulder   | N/A          | 6%                                 | 100%         | 0%               | Tenderness, loss of range of motion, MRI findings            |
| Lumbar Spine    | II           | 7%                                 | 85%          | 15%              | Tenderness, non-verifiable radicular complaints, MRI findings, difficulties with ADL |

**Apportionment and Methodology:**
- **Cervical and Lumbar Spine Impairments:** 85% attributed to the industrial injury, with the remaining 15% considered non-industrial based on unspecified factors.
- **Shoulder Impairments:** 100% attributed to the industrial injury.

Methodology reflected relies on the DRE (Diagnosis-Related Estimates) method for spinal impairments and range of motion assessments for shoulder impairments. However, specifics such as the edition of the AMA Guides used for evaluation, detailed ranges of motion, specific nerves affected by radicular complaints, or a thorough narrative on establishing impairments are not provided within the summary.

**Observations and Missing Information:**

1. **Edition of AMA Guides:** Not specified, crucial for ensuring adherence to current evaluation standards.
2. **Mechanism of Injury Details:** Lacks explicit connection between the collision’s dynamics and the resultant conditions.
3. **Shoulder Range of Motion:** Absence of specific measurements hinders the verification of impairment ratings accuracy.
4. **Radicular Complaints:** Lack of detail about affected nerves and correlation with MRI scans.
5. **Clinical Examinations:** It's unclear to what extent they informed the impairment ratings beyond imaging studies.
6. **Impact of Left Knee History:** Not assessed in relation to current impairments.
7. **Functional Limitations Details:** Specific difficulties with activities of daily living due to impairments are mentioned but not detailed.
8. **Consideration of Non-Industrial Factors:** During the apportionment process, the specific non-industrial factors considered are not clarified.
9. **Narrative Explanations for Specific Impairment Percentages:** The rationale behind choosing specific impairment percentages within the provided ranges lacks detail.

**Conclusion:**
This report compiles available impairment and injury-related information following an industrial accident involving a truck driver. While it identifies specific impairments with their percentage ratings and causative correlations to the industrial incident, it lacks several descriptive and methodological details essential for a comprehensive evaluation. Further clarification and additional evaluations might be needed for a complete understanding of the patient's condition and the impairment's extents, as well as to ensure the fairness and accuracy of the worker's compensation claim process.
"""

if __name__ == "__main__":
    output_docx_file_path = r"D:\OneDrive\dev\PycharmProjects\aidream\temp\app_outputs\output1.docx"
    convert_markdown_content_to_docx(markdown_content, output_docx_file_path)

