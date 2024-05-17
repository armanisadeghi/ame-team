
# ------------------------------- IMPORTANT NOTE -------------------------------
# IMPORTANT: Please do not delete or remove samples, unless you added them.
# You can copy one and modify it as a new entry, but do not modify what is already here. Thank you.
# There is one primary function to get you any sample on this page. You can import "get_sample_data" and name the content you want.
# -----------------------------------------------------------------------------

ama_sample_1 = '''#Medical Report Review
Short summary of the patient's medical report and impairments.

Date of Birth or Age: 07/26/1972
Date of Injury: 01/24/2023

| Body Part | DRE Category | Percentage Range | Whole Person Impairment Percentage | Method of Rating Used | Industrial % | Non-Industrial % | Specific Evidence Used by Doctor |
|-----------|--------------|------------------|-------------------------------------|----------------------|--------------|------------------|-----------------------------------|
| Cervical Spine | - | - | 16% | Range of Motion | 69% | 31% | Multilevel disc bulges at C3-C4, C4-C5, and C5-C6 |
| Lumbar Spine | DRE III | 10-13% | 13% | DRE Method | 62% | 38% | Radiculopathy verified by EMG/NCV |
| Thoracic Spine | DRE II | 5-8% | 8% | DRE Method | 100% | 0% | Paraspinal muscle spasms and asymmetric loss of motion |
| Left Knee | - | - | 3% | Arthritis Impairment | 70% | 30% | Mild degenerative changes on MRI |
| Right Knee | - | - | 3% | Arthritis Impairment | 70% | 30% | Mild degenerative changes on MRI |

Injury & Job Information:
- Occupational Code: Cashier/Customer Service/Stocker
- Injury Circumstances: Cumulative trauma from repetitive work duties, including manually opening and closing heavy automatic doors.

Additional observations:
- The patient had prior degenerative changes in the cervical and lumbar spine, but they were non-labor disabling.
- The patient was able to work at CVS as a Cashier/Customer Service/Stocker from November 2021 until the date of injury on January 24, 2023.
- The doctor provided an add-on of 3% for chronic pain related to the cervical, thoracic, and lumbar spine.
- The doctor did not believe the Combined Values Chart accurately reflected the patient's disability and instead added the impairments for the bilateral knees.

Questions to address about the report:
1. In the doctor's report, does it mention which specific edition of the AMA Guides was used for evaluating the patient's impairments?
2. Regarding the mechanism of injury, does the report provide any additional details about how the head-on collision led to the diagnosed conditions and impairments?
3. What are the specific ranges of motion measured for the shoulders, and are they within normal limits or did they reveal any restrictions?'''
ama_sample_2 = """Date of Birth or Age: 07/26/1972
Date of Injury: 01/24/2023

**Headline:** "Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance"
   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.
   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.
   - **Unique Angle:** This piece offers a fresh perspective by examining not just the aesthetic appeal of the mini dress but its socio-cultural impact, weaving in untold stories of resistance, liberation, and identity that surround this iconic garment.
   - **Value Proposition:** Readers will gain insightful narratives that paint the mini dress not merely as a fashion item but as a beacon of societal change, challenging conventional narratives and enriching their understanding of fashion history.
   - **Key Takeaways:** An exploration of the mini dress's roots, the visionaries who propelled it to fame, and its enduring symbol as an instrument of empowerment and fashion innovation.
   - **Engagement Triggers:** Invitations for readers to contribute personal anecdotes or reflections on how the mini dress has influenced their perception of style, identity, or liberation.
   - **SEO Keywords:** "Mini dress evolution," "influence of mini dresses in society," "fashion icons and the mini dress."
   - **Tone and Style:** Richly narrative and thought-provoking, blending rigorous historical insights with captivating tales of fashion revolutions.

---

In crafting this article, imagine yourself not just as a writer but as an architectural historian of fabric and societal norms, reconstructing the story of a garment that did more than just shorten the hemline—it redefined the silhouette of freedom, rebellion, and identity. You have the power to weave a tapestry that combines the threads of fashion, culture, and history into a compelling narrative that challenges the reader to view the mini dress not only as a piece of clothing but as a cultural milestone that marked a significant shift in societal attitudes towards women's autonomy and expression. Through rich storytelling, detailed historical context, and engaging personal reflections, inspire your readers to see the mini dress as a herald of change, a symbol of progress, and an emblem of modern elegance that continues to influence fashion trends today. This isn't just a blog post; it's a tribute to a revolutionary garment that captured the spirit of an era and continues to inspire those who dare to defy convention.

| Body Part | DRE Category | Percentage Range | Whole Person Impairment | Method Used | Industrial % | Non-Industrial % | Evidence Used |
|-----------|--------------|------------------|-------------------------|-------------|--------------|------------------|---------------|
| Cervical Spine | - | - | 16% | Range of Motion | 69% | 31% | Multilevel disc bulges at C3-C4, C4-C5, and C5-C6 |
| Lumbar Spine | DRE III | 10-13% | 13% | DRE Method | 62% | 38% | Radiculopathy verified by EMG/NCV |
| Thoracic Spine | DRE II | 5-8% | 8% | DRE Method | 100% | 0% | Paraspinal muscle spasms and asymmetric loss of motion |
| Left Knee | - | - | 3% | Arthritis Impairment | 70% | 30% | Mild degenerative changes on MRI |
| Right Knee | - | - | 3% | Arthritis Impairment | 70% | 30% | Mild degenerative changes on MRI |
---
Injury & Job Information:
- Occupational Code: Cashier/Customer Service/Stocker
- Injury Circumstances: Cumulative trauma from repetitive work duties, including manually opening and closing heavy automatic doors.

```python
table_as_list_of_tuples = [('text 1', 'text 2', 'text 3'), ('text 4', 'text 5', 'text 6'), ('text 7', 'text 8', 'text 9')]
lengths = [len(text) for text in table_as_list_of_tuples[0]]
```
**Headline:** \"Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance\"\n   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.\n   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.\n   - **Unique Angle:** This piece offers a fresh perspective by examining not just the aesthetic appeal of the mini dress but its socio-cultural impact, weaving in untold stories of resistance, liberation, and identity that surround this iconic garment.\n   - **Value Proposition:** Readers will gain insightful narratives that paint the mini dress not merely as a fashion item but as a beacon of societal change, challenging conventional narratives and enriching their understanding of fashion history.\n   - **Key Takeaways:** An exploration of the mini dress's roots, the visionaries who propelled it to fame, and its enduring symbol as an instrument of empowerment and fashion innovation.\n   - **Engagement Triggers:** Invitations for readers to contribute personal anecdotes or reflections on how the mini dress has influenced their perception of style, identity, or liberation.\n   - **SEO Keywords:** \"Mini dress evolution,\" \"influence of mini dresses in society,\" \"fashion icons and the mini dress.\"\n   - **Tone and Style:** Richly narrative and thought-provoking, blending rigorous historical insights with captivating tales of fashion revolutions.

Additional notes:
- The patient had prior degenerative changes in the cervical and lumbar spine, but they were non-labor disabling.
- The patient was able to work at CVS as a Cashier/Customer Service/Stocker from November 2021 until the date of injury on January 24, 2023.
- The doctor provided an add-on of 3% for chronic pain related to the cervical, thoracic, and lumbar spine.
- The doctor did not believe the Combined Values Chart accurately reflected the patient's disability and instead added the impairments for the bilateral knees.

Questions:
1. In the doctor's report, does it mention which specific edition of the AMA Guides was used for evaluating the patient's impairments?
2. Regarding the mechanism of injury, does the report provide any additional details about how the head-on collision led to the diagnosed conditions and impairments?
3. What are the specific ranges of motion measured for the shoulders, and are they within normal limits or did they reveal any restrictions?"""
ama_sample_3 = """It sounds like you opened Chrome's Developer Tools in a mode that persists across sessions, possibly related to service worker inspection or another advanced debugging feature. Here's how you can try to resolve this:

### Closing Chrome Developer Tools

Normally, Chrome Developer Tools can be toggled on and off by pressing `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Opt+I` (Mac). However, if they keep reappearing every time you close them, it might be due to an accidental configuration or a glitch.

### Clearing Service Workers

Since the issue started after inspecting service workers, let's try clearing them:

1. **Open Chrome Developer Tools**: Press `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Opt+I` (Mac).
2. **Go to the Application Tab**: Click on the "Application" tab at the top of the Developer Tools window.
3. **Find Service Workers**: In the left sidebar of the Application tab, look for the "Service Workers" section.
4. **Unregister Service Workers**: You should see a list of service workers registered by various websites, including `localhost`. Click on "Unregister" for any service workers related to `localhost` or your project.

### Resetting Chrome Developer Tools

If the Developer Tools window keeps popping up due to a configuration issue, resetting the Developer Tools settings might help:

1. **Open Chrome Developer Tools**: Press `Ctrl+Shift+I` (Windows/Linux) or `Cmd+Opt+I` (Mac).
2. **Open Settings in Developer Tools**: Click on the gear icon in the upper-right corner of the Developer Tools window, or press `F1` while the Developer Tools window is focused.
3. **Reset to Defaults**: In the Settings pane, find the "Restore defaults and reload" button (usually at the bottom) and click it. This will reset all Developer Tools settings to their default values.

### Checking Chrome Settings

If the issue persists and seems to affect more than just the Developer Tools (e.g., Chrome behaves oddly in other ways), you may want to review Chrome's settings or consider resetting Chrome entirely:

1. **Open Chrome Settings**: Click on the three dots in the upper right corner of Chrome, then select "Settings".
2. **Reset Settings**: Scroll down to "Advanced", find the "Reset and clean up" section, and choose "Restore settings to their original defaults". Note that this will reset your startup page, new tab page, search engine, and pinned tabs, as well as disable all extensions and clear temporary data. It won't delete your bookmarks, history, or saved passwords.

Before you reset Chrome settings, ensure you've backed up or are ready to reconfigure anything as needed. Resetting Chrome can help resolve persistent issues but can also remove customizations you've made to the browser.

If these steps do not resolve the problem, or if there was a specific command or action taken that you believe caused this issue, please let me know for further assistance."""
ama_sample_4 = """**Headline:** \"Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance\"\n   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.\n   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.\n   - **Unique Angle:** This piece offers a fresh perspective by examining not just the aesthetic appeal of the mini dress but its socio-cultural impact, weaving in untold stories of resistance, liberation, and identity that surround this iconic garment.\n   - **Value Proposition:** Readers will gain insightful narratives that paint the mini dress not merely as a fashion item but as a beacon of societal change, challenging conventional narratives and enriching their understanding of fashion history.\n   - **Key Takeaways:** An exploration of the mini dress's roots, the visionaries who propelled it to fame, and its enduring symbol as an instrument of empowerment and fashion innovation.\n   - **Engagement Triggers:** Invitations for readers to contribute personal anecdotes or reflections on how the mini dress has influenced their perception of style, identity, or liberation.\n   - **SEO Keywords:** \"Mini dress evolution,\" \"influence of mini dresses in society,\" \"fashion icons and the mini dress.\"\n   - **Tone and Style:** Richly narrative and thought-provoking, blending rigorous historical insights with captivating tales of fashion revolutions."""
ama_sample_5 = """Date of Birth or Age: 07/26/1972
Date of Injury: 01/24/2023

**Headline:** "Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance"
   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.
   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.

---

In crafting this article, imagine yourself not just as a writer but as an architectural historian of fabric and societal norms.

| Body Part | DRE Category | Percentage Range | Whole Person Impairment | Method Used | Industrial % | Non-Industrial % | Evidence Used |
|-----------|--------------|------------------|-------------------------|-------------|--------------|------------------|---------------|
| Cervical Spine | - | - | 16% | Range of Motion | 69% | 31% | Multilevel disc bulges at C3-C4, C4-C5, and C5-C6 |
| Lumbar Spine | DRE III | 10-13% | 13% | DRE Method | 62% | 38% | Radiculopathy verified by EMG/NCV |
---
Injury & Job Information:
- Occupational Code: Cashier/Customer Service/Stocker
- Injury Circumstances: Cumulative trauma from repetitive work duties, including manually opening and closing heavy automatic doors.

```python
table_as_list_of_tuples = [('text 1', 'text 2', 'text 3'), ('text 4', 'text 5', 'text 6'), ('text 7', 'text 8', 'text 9')]
lengths = [len(text) for text in table_as_list_of_tuples[0]]
```
***section break***
Additional notes:
- The patient had prior degenerative changes in the cervical and lumbar spine, but they were non-labor disabling.
- The patient was able to work at CVS as a Cashier/Customer Service/Stocker from November 2021 until the date of injury on January 24, 2023.

Questions:
1. In the doctor's report, does it mention which specific edition of the AMA Guides was used for evaluating the patient's impairments?
2. Regarding the mechanism of injury, does the report provide any additional details about how the head-on collision led to the diagnosed conditions and impairments?"""
ama_sample_6 = '''#Medical Report Review
Date of Birth or Age: 07/26/1972
Date of Injury: 01/24/2023

**Headline:** "Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance"
   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.
   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.
   - **Unique Angle:** This piece offers a fresh perspective by examining not just the aesthetic appeal of the mini dress but its socio-cultural impact, weaving in untold stories of resistance, liberation, and identity that surround this iconic garment.
   - **Value Proposition:** Readers will gain insightful narratives that paint the mini dress not merely as a fashion item but as a beacon of societal change, challenging conventional narratives and enriching their understanding of fashion history.
   - **Key Takeaways:** An exploration of the mini dress's roots, the visionaries who propelled it to fame, and its enduring symbol as an instrument of empowerment and fashion innovation.
   - **Engagement Triggers:** Invitations for readers to contribute personal anecdotes or reflections on how the mini dress has influenced their perception of style, identity, or liberation.
   - **SEO Keywords:** "Mini dress evolution," "influence of mini dresses in society," "fashion icons and the mini dress."
   - **Tone and Style:** Richly narrative and thought-provoking, blending rigorous historical insights with captivating tales of fashion revolutions.

---

| Body Part      | Impairment Code  | DRE Category | Whole Person Impairment | Industrial % |
|----------------|------------------|--------------|------------------------|--------------|
| Cervical Spine   | 15.01.01.00      | II           | 6%                    | 85%          |
| Right Shoulder   | 16.02.01.00      | N/A          | 5%                    | 100%         |
| Left Shoulder    | 16.02.01.00      | N/A          | 6%                    | 100%         |
| Lumbar Spine     | 15.03.01.00      | II           | 7%                    | 85%          |


In crafting this article, imagine yourself not just as a writer but as an architectural historian of fabric and societal norms, reconstructing the story of a garment that did more than just shorten the hemline—it redefined the silhouette of freedom, rebellion, and identity. You have the power to weave a tapestry that combines the threads of fashion, culture, and history into a compelling narrative that challenges the reader to view the mini dress not only as a piece of clothing but as a cultural milestone that marked a significant shift in societal attitudes towards women's autonomy and expression. Through rich storytelling, detailed historical context, and engaging personal reflections, inspire your readers to see the mini dress as a herald of change, a symbol of progress, and an emblem of modern elegance that continues to influence fashion trends today. This isn't just a blog post; it's a tribute to a revolutionary garment that captured the spirit of an era and continues to inspire those who dare to defy convention.

| Body Part | DRE Category | Percentage Range | Whole Person Impairment | Method Used | Industrial % | Non-Industrial % | Evidence Used |
|-----------|--------------|------------------|-------------------------|-------------|--------------|------------------|---------------|
| Cervical Spine | - | - | 16% | Range of Motion | 69% | 31% | Multilevel disc bulges at C3-C4, C4-C5, and C5-C6 |
| Lumbar Spine | DRE III | 10-13% | 13% | DRE Method | 62% | 38% | Radiculopathy verified by EMG/NCV |
| Thoracic Spine | DRE II | 5-8% | 8% | DRE Method | 100% | 0% | Paraspinal muscle spasms and asymmetric loss of motion |
| Left Knee | - | - | 3% | Arthritis Impairment | 70% | 30% | Mild degenerative changes on MRI |
| Right Knee | - | - | 3% | Arthritis Impairment | 70% | 30% | Mild degenerative changes on MRI |
---
Injury & Job Information:
- Occupational Code: Cashier/Customer Service/Stocker
- Injury Circumstances: Cumulative trauma from repetitive work duties, including manually opening and closing heavy automatic doors.

```python
table_as_list_of_tuples = [('text 1', 'text 2', 'text 3'), ('text 4', 'text 5', 'text 6'), ('text 7', 'text 8', 'text 9')]
lengths = [len(text) for text in table_as_list_of_tuples[0]]
```
**Headline:** \"Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance\"\n   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.\n   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.\n   - **Unique Angle:** This piece offers a fresh perspective by examining not just the aesthetic appeal of the mini dress but its socio-cultural impact, weaving in untold stories of resistance, liberation, and identity that surround this iconic garment.\n   - **Value Proposition:** Readers will gain insightful narratives that paint the mini dress not merely as a fashion item but as a beacon of societal change, challenging conventional narratives and enriching their understanding of fashion history.\n   - **Key Takeaways:** An exploration of the mini dress's roots, the visionaries who propelled it to fame, and its enduring symbol as an instrument of empowerment and fashion innovation.\n   - **Engagement Triggers:** Invitations for readers to contribute personal anecdotes or reflections on how the mini dress has influenced their perception of style, identity, or liberation.\n   - **SEO Keywords:** \"Mini dress evolution,\" \"influence of mini dresses in society,\" \"fashion icons and the mini dress.\"\n   - **Tone and Style:** Richly narrative and thought-provoking, blending rigorous historical insights with captivating tales of fashion revolutions.

Additional notes:
- The patient had prior degenerative changes in the cervical and lumbar spine, but they were non-labor disabling.
- The patient was able to work at CVS as a Cashier/Customer Service/Stocker from November 2021 until the date of injury on January 24, 2023.
- The doctor provided an add-on of 3% for chronic pain related to the cervical, thoracic, and lumbar spine.
- The doctor did not believe the Combined Values Chart accurately reflected the patient's disability and instead added the impairments for the bilateral knees.

Questions:
1. In the doctor's report, does it mention which specific edition of the AMA Guides was used for evaluating the patient's impairments?
2. Regarding the mechanism of injury, does the report provide any additional details about how the head-on collision led to the diagnosed conditions and impairments?
3. What are the specific ranges of motion measured for the shoulders, and are they within normal limits or did they reveal any restrictions?'''
sample_code = '''
table_as_list_of_tuples = [('text 1', 'text 2', 'text 3'), ('text 4', 'text 5', 'text 6'), ('text 7', 'text 8', 'text 9')]
lengths = [len(text) for text in table_as_list_of_tuples[0]]
'''
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
ama_sample_7 = """# Full AMA Medical Report Analysis

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

| Body Part       | DRE Category | Whole Person Impairment | Industrial % | Non-Industrial % | Specific Evidence Used by Physician                                                  |
|-----------------|--------------|-------------------------|--------------|------------------|--------------------------------------------------------------------------------------|
| Cervical Spine  | II           | 6%                      | 85%          | 15%              | Tenderness, non-verifiable radicular complaints, MRI findings, difficulties with ADL |
| Right Shoulder  | N/A          | 5%                      | 100%         | 0%               | Tenderness, loss of range of motion, MRI findings                                    |
| Left Shoulder   | N/A          | 6%                      | 100%         | 0%               | Tenderness, loss of range of motion, MRI findings                                    |
| Lumbar Spine    | II           | 7%                      | 85%          | 15%              | Tenderness, non-verifiable radicular complaints, MRI findings, difficulties with ADL |

**Apportionment and Methodology:**
- **Cervical and Lumbar Spine Impairments:** 85% attributed to the industrial injury, with the remaining 15% considered non-industrial based on unspecified factors.
- **Shoulder Impairments:** 100% attributed to the industrial injury.

+-----+--------+-------+
| ID  | Name   | Age   |
+-----+--------+-------+
| 1   | Alice  | 30    |
| 2   | Bob    | 25    |
+-----+--------+-------+

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


#Added by jatin
ama_medical_report_sample = """ 

[STRUCTURED_MEDICAL_REPORT]

Date of Birth or Age: 07/26/1972
Date of Injury: 01/24/2023

Reporting Doctor: Michael D. Zeger, D.C., Q.M.E.

Doctor Role: QME

| Body Part | DRE Category | Percentage Range | Whole Person Impairment Percentage | Method of Rating Used | Industrial % | Non-Industrial % | Specific Evidence Used by Doctor | Other Specific Relevant Information |
|-----------|--------------|------------------|-------------------------------------|----------------------|--------------|------------------|-----------------------------------|-------------------------------------|
| Cervical Spine | DRE Category II | - | 8% | AMA Guides, 5th edition | 20% | 80% | Observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy | No evidence of alteration in structural integrity of the vertebra |
| Thoracic Spine | DRE Category II | - | 8% | AMA Guides, 5th edition | 20% | 80% | Observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy | No documented alteration in structural integrity of the vertebra |
| Lumbar Spine | DRE Category II | - | 8% | AMA Guides, 5th edition | 20% | 80% | Observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy | No evidence of alteration in structural integrity of the vertebra |

Injury & Job Information:
- Occupational Code: Cashier/Customer Service/Stocker
- Injury Circumstances: The applicant was required to open and close a heavy automatic entrance door manually for several weeks, which caused pain in her neck, mid back, and low back.

Additional notes:
- The applicant has a history of prior neck and back injuries, including a slip-and-fall in the shower in 2021 and a work injury in 2022 while moving heavy items at CVS.
- The applicant has a history of diabetes, hypertension, and hyperlipidemia.
- The applicant had concurrent employment at a gas station from April 2021 to December 2022.

[INITIAL_FEEDBACK]

Based on the provided medical facts and the AMA Guides, let's evaluate the accuracy of the doctor's findings, method of rating, and overall record submission, specifically focusing on the relevance of the patient's history to the current impairment determination.
First, let's address some cases that indicate a history of prior injuries followed by a complete analysis for the final question:


[SPECIFIC_QUESTIONS]

1. What is the doctor's role in the context of this evaluation? Is the role of QME explicitly stated in the report?
2. Has the date of birth or the patient's age, as well as the date of the injury, been reported accurately? What are these dates?
3. Are all the body parts involved in the claimed injuries stated? Can you please provide a list of the body parts mentioned and their corresponding impairment ratings?
4. For each body part, is the impairment rating derived from: a) the range of motion (ROM) and; b) stability and clinical assessment stated in line with the AMA Guides' formulas and criteria? Please provide a brief explanation for each body part.
5. Have you identified any discrepancies between the doctor's findings and the formulaic calculations in the AMA Guides?
6. Does the report provide a clear breakdown of the proportion of the whole person impairment that the doctor believes arises from industrial versus non-industrial factors?
7. Is there detailed documentation of the specific evidence used by the doctor to support the impairment ratings for each body part? Please provide a summary of this evidence.
8. Does the report include any relevant information about the mechanism of injury or the circumstances surrounding it, including any exacerbating or mitigating events? Please provide a summary of these details.
9. Has the doctor taken into account the patient's medical and occupational history, including any pre-existing conditions and prior work-related injuries? If so, how have these factors influenced the impairment rating and apportionment?
10. Lastly, does the report conclude with an assessment of the patient's overall impairment, clarifying the methodology used for the evaluation and the extent to which the doctor believes the patient's current condition is due to the industrial injury?

[COMPLETE_ANSWERS]

1. The doctor's role is explicitly stated as a Qualified Medical Evaluator (QME) in the report: "Reporting Doctor: Michael D. Zeger, D.C., Q.M.E." and "Doctor Role: QME".

2. Yes, the date of birth (07/26/1972) and the date of injury (01/24/2023) are accurately reported in the medical report.

3. Yes, the body parts involved in the claimed injuries are stated:
   - Cervical Spine: 8% whole person impairment
   - Thoracic Spine: 8% whole person impairment
   - Lumbar Spine: 8% whole person impairment

4. For each body part, the impairment rating is derived from the AMA Guides, 5th edition, using the DRE (Diagnosis-Related Estimate) method:
   - Cervical Spine: DRE Category II, based on "observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy."
   - Thoracic Spine: DRE Category II, based on "observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy."
   - Lumbar Spine: DRE Category II, based on "observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy."

5. No discrepancies between the doctor's findings and the AMA Guides' formulaic calculations have been identified in the report.

6. Yes, the report provides a clear breakdown of the industrial versus non-industrial factors contributing to the whole person impairment for each body part:
   - Cervical Spine: 20% industrial, 80% non-industrial
   - Thoracic Spine: 20% industrial, 80% non-industrial
   - Lumbar Spine: 20% industrial, 80% non-industrial

7. Yes, the report provides detailed documentation of the specific evidence used by the doctor to support the impairment ratings:
   - Cervical Spine: "Observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy" and "No evidence of alteration in structural integrity of the vertebra."
   - Thoracic Spine: "Observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy" and "No documented alteration in structural integrity of the vertebra."
   - Lumbar Spine: "Observed residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy" and "No evidence of alteration in structural integrity of the vertebra."

8. Yes, the report includes relevant information about the mechanism of injury and the circumstances surrounding it: "The applicant was required to open and close a heavy automatic entrance door manually for several weeks, which caused pain in her neck, mid back, and low back."

9. Yes, the doctor has taken into account the patient's medical and occupational history:
   - Prior neck and back injuries: "The applicant has a history of prior neck and back injuries, including a slip-and-fall in the shower in 2021 and a work injury in 2022 while moving heavy items at CVS."
   - Pre-existing conditions: "The applicant has a history of diabetes, hypertension, and hyperlipidemia."
   - Concurrent employment: "The applicant had concurrent employment at a gas station from April 2021 to December 2022."
   These factors have influenced the apportionment of the impairment ratings, with 80% attributed to non-industrial factors for each body part.

10. The report concludes with an assessment of the patient's overall impairment, clarifying the methodology used (AMA Guides, 5th edition, DRE method) and the extent to which the doctor believes the patient's current condition is due to the industrial injury (20% for each body part).

[AME_SPECIALIST_FINAL_ANALYSIS]

Based on the provided information, your doctor performed a thorough evaluation and adhered to the principles outlined in the AMA Guides, 5th edition. 
- The reporting doctor accurately identified the body parts involved (cervical, thoracic, and lumbar spine) and assigned a 8% whole person impairment to each category based on the findings of residual muscle spasm, asymmetrical range of motion loss, and non-verifiable radiculopathy, all indicative of DRE Category II.

[JOB_CODE_DATA]

Based on the information provided, the official job code that is closest to this case is:

Cashier/Customer Service/Stocker

The key details that support this:

- The occupational code listed is "Cashier/Customer Service/Stocker"
- The injury circumstances indicate the applicant was required to open and close a heavy automatic entrance door manually for several weeks, which caused pain in her neck, mid back, and low back. This type of job duty is consistent with a Cashier/Customer Service/Stocker role.

The other details provided, such as the reporting doctor being a QME, the impairment ratings for the cervical, thoracic, and lumbar spine, and the applicant's prior injury history, are all relevant information but do not directly impact the determination of the closest official job code.

[IMPAIRMENT_CODE_TABLE]

| Body Part     | Impairment Code | DRE Category | Whole Person Impairment | Industrial % |
|---------------|------------------|--------------|-------------------------|--------------|
| Cervical Spine | 15.01.01.00      | II           | 8%                      | 20%          |
| Thoracic Spine | 15.02.01.00      | II           | 8%                      | 20%          |
| Lumbar Spine   | 15.03.01.00      | II           | 8%                      | 20%          |

[FULL_AME_EVALUATION_REPORT]

**Final Report: Summary of Medical Evaluation and Findings**

**Introduction:**

This report consolidates key findings from a series of evaluations involving an applicant suffering from injuries sustained in the workplace. The significant details encompass dates, medical assessments, job information, and the applicant's medical history, carefully organized to provide a comprehensive overview.

**Applicant Details:**

- **Date of Birth:** 07/26/1972
- **Date of Injury:** 01/24/2023

**Medical Evaluation Overview:**

- **Reporting Doctor:** Michael D. Zeger, D.C., Q.M.E.
- **Doctor's Role:** Qualified Medical Evaluator (QME)
- **Evaluation Methodology:** AMA Guides, 5th edition

**Injury Circumstances and Job Information:**

- **Occupational Role:** Cashier/Customer Service/Stocker
- **Mechanism of Injury:** Required manual operation of a heavy automatic entrance door over several weeks, causing pain in the neck, mid back, and low back.

**Medical Findings and Ratings:**

| Body Part     | DRE Category | Whole Person Impairment (%) | Industrial % | Non-Industrial % | Specific Evidence                       |
|---------------|--------------|-----------------------------|--------------|------------------|-----------------------------------------|
| Cervical Spine| II           | 8                           | 20           | 80               | Residual muscle spasm, asymmetrical ROM loss, non-verifiable radiculopathy |
| Thoracic Spine| II           | 8                           | 20           | 80               | Residual muscle spasm, asymmetrical ROM loss, non-verifiable radiculopathy |
| Lumbar Spine  | II           | 8                           | 20           | 80               | Residual muscle spasm, asymmetrical ROM loss, non-verifiable radiculopathy |

- **No evidence or documented alteration in the structural integrity of the vertebra was found for any of the assessed body parts.**

**Applicant's Medical and Occupational History:**

- **Prior Injuries:** Slip-and-fall in 2021, work injury in 2022 at CVS while moving heavy items.
- **Pre-existing Conditions:** Diabetes, hypertension, hyperlipidemia.
- **Concurrent Employment:** Gas station work from April 2021 to December 2022.

**Assessment Summary:**

The comprehensive evaluation conducted by Dr. Michael D. Zeger, leveraging the AMA Guides, 5th edition, focuses on the cervical, thoracic, and lumbar spine, each receiving an 8% whole person impairment rating. The assessment attributes 20% of the impairment to industrial causes and 80% to non-industrial factors, grounded in a detailed examination of specific evidence like residual muscle spasm and asymmetrical range of motion loss. Crucially, the medical history, particularly previous injuries and conditions, played a significant role in the apportionment process.

The circumstances surrounding the injury, notably the manual operation of a heavy entrance door, align with the occupational duties of a Cashier/Customer Service/Stocker, reinforcing the job relevance of the injury. The evaluation meticulously considers the applicant's medical and occupational background to ensure a fair and accurate understanding of the injury's nature and its lasting impact.

**Conclusion:**

This report effectively encapsulates the essential findings from the medical evaluations conducted by Dr. Zeger. It provides a structured and clear understanding of the applicant's work-related injuries, taking into account previous health conditions and employment history to arrive at an informed evaluation. The impairment ratings, grounded in robust evidence and following established guidelines, offer critical insights into the extent of the injuries and their implications for the applicant's functional capabilities."""
