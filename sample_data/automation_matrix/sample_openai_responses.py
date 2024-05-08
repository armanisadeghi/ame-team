# sample_openai_responses.py

sample_1 = """
If the `extract_code_snippets` method is still returning an empty dictionary `{}`, even after adjusting the regular expression pattern, there might be a few areas to investigate:

1. **Regular Expression Pattern**: Ensure the regular expression correctly matches the expected format of the code blocks in your sample data. If your code blocks are surrounded by triple single quotes (`'''`) followed by a language identifier, the pattern needs to capture this structure accurately. 

2. **Sample Data Format**: Verify the format of your sample data. Ensure the code blocks are formatted as expected and match the pattern you're trying to capture. For instance, if there's an inconsistency in newline characters or spaces around the code blocks, that might prevent matches.

3. **Regular Expression Flags**: The use of `re.MULTILINE` is appropriate for matching patterns that span multiple lines, but if your code snippets are within a larger body of text, you might also need `re.DOTALL` to allow the dot (`.`) to match newline characters.

4. **Function Call**: Confirm that the `extract_code_snippets` method is being called with the correct `text` argument and that the `text` contains the code snippets as expected.

Given these considerations, let's refine the method and pattern:

```python
import re
from typing import Dict, List

async def extract_code_snippets(self, text: str, *args) -> Dict[str, List[str]]:
    print(f"Original text for extract_code_snippets:\n{text}\n\n")
    code_snippets = {}

    # Updated pattern to match both triple backticks and triple single quotes, with optional language identifier
    pattern = '(?P<delim>`{3}|\'{3})(?P<lang>[\\w#+-]*)\\n(?P<code>.*?)\\n(?P=delim)'

    matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE)
    for match in matches:
        # Extract named groups
        language = match.group('lang') or 'no_language'
        code = match.group('code').strip()

        # Append code snippet to the appropriate language key
        code_snippets.setdefault(language, []).append(code)

    print(f"Code Snippets:\n{code_snippets}\n")
    return code_snippets
```

In this version:
- The regular expression pattern uses named groups (`?P<name>`) for clarity and uses a backreference (`(?P=delim)`) to match the same type of delimiter that started the code block.
- `re.finditer` is used for iteration, which provides match objects from which named groups are extracted.
- `re.DOTALL` allows the dot (`.`) to match newline characters, ensuring that multi-line code snippets are captured.
- The `setdefault` method is used to simplify appending code snippets to their respective language keys.

Please adjust the sample data or method call accordingly if the issue persists, and ensure that the `text` passed into the method indeed contains the expected code blocks.
"""

sample_2 = """
To facilitate local testing of post-processing without making actual calls to the OpenAI API each time, you can create a function called `local_sample_data_processing` that utilizes predefined sample data from a module named `sample_openai_responses`. This function will simulate the processing of an OpenAI response by using this sample data and applying the specified post-processing defined in `return_params`.

First, ensure you have a module `sample_openai_responses.py` with a function `get_sample_data()` that returns a sample response identical in structure to what you'd expect from an actual OpenAI call:

```python
# sample_openai_responses.py

def get_sample_data():
    # Return a sample structure mimicking the OpenAI response content
    return "Your sample response content here."
```

Then, in your main module where `AsyncOpenAIWrapper` is defined, import `get_sample_data` and create the `local_sample_data_processing` function:

```python
from sample_openai_responses import get_sample_data
import asyncio

class AsyncOpenAIWrapper:
    # Your class definition continues as before...

    async def local_sample_data_processing(self, return_params=None):
        # Use get_sample_data to simulate the "content" part of an OpenAI response
        sample_content = get_sample_data()

        # Process the sample content as if it were the content of a real OpenAI response
        processed_content = {
            'initial_content': sample_content,
            'steps': {}
        }

        if return_params:
            for step in return_params:
                processor_name = step.get('processor')
                args = step.get('args', {})
                depends_on = step.get('depends_on', 'content')

                if depends_on == 'content':
                    input_data = sample_content
                else:
                    input_data = processed_content['steps'].get(depends_on, {}).get('output')

                processor = getattr(self, processor_name, None)

                if asyncio.iscoroutinefunction(processor):
                    output_data = await processor(input_data, **args)
                else:
                    output_data = await asyncio.to_thread(processor, input_data, **args)

                processed_content['steps'][processor_name] = {
                    'output': output_data,
                    'depends_on': depends_on
                }

        return processed_content
```

To use this for testing:

```python
async def test_post_processing():
    wrapper = AsyncOpenAIWrapper()
    return_params = [
        # Define your post-processing steps here
        {'processor': 'your_processor_function_name', 'depends_on': 'content', 'args': {}},
        # Add more steps as needed
    ]

    processed_content = await wrapper.local_sample_data_processing(return_params)
    print(processed_content)

if __name__ == "__main__":
    asyncio.run(test_post_processing())
```

This setup allows you to test post-processing logic locally using predefined sample data, enabling debugging and development without the need for actual API calls. Ensure that your post-processing functions (`processor`) are defined within the `AsyncOpenAIWrapper` class and are compatible with the expected input from `get_sample_data()`.
"""

sample_3 = """
1. **Headline:** "Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance"
   - **Overview:** This post dives into the history of the mini dress and its impact on contemporary fashion trends.
   - **Target Audience:** Fashion enthusiasts, history buffs, and anyone interested in the evolution of women's clothing.
   - **Unique Angle:** Exploring the cultural and fashion industry shifts that popularized the mini dress, highlighting iconic moments and figures.
   - **Value Proposition:** Offers readers a deep understanding of the mini dress beyond its aesthetic, enriching their appreciation for this fashion item.
   - **Key Takeaways:** Insight into the origins of the mini dress, key designers who championed its rise, and its symbolic representation of freedom and rebellion.
   - **Engagement Triggers:** Encourage readers to share their first mini dress experience or how this iconic piece has empowered them.
   - **SEO Keywords:** "History of mini dress," "impact of mini dresses on fashion," "iconic mini dresses."
   - **Tone and Style:** Informative yet captivating, weaving historical facts with engaging storytelling.

2. **Headline:** "From Day to Night: Styling Your Mini Dress for Any Occasion"
   - **Overview:** A comprehensive guide to styling different types of mini dresses for various occasions.
   - **Target Audience:** Fashion-forward readers looking for styling tips, from casual outings to elegant evening events.
   - **Unique Angle:** Offers practical styling tips, accessory pairings, and footwear suggestions for making the most out of one mini dress.
   - **Value Proposition:** Readers will learn how to creatively style their mini dresses, maximizing their wardrobe versatility.
   - **Key Takeaways:** Tips on accessorizing, layering, and selecting appropriate footwear to transition the mini dress across different settings.
   - **Engagement Triggers:** Invite readers to share their favorite mini dress styling tips or post before-and-after styling transformations.
   - **SEO Keywords:** "Styling mini dresses," "mini dress outfits," "versatile mini dress looks."
   - **Tone and Style:** Friendly and inspirational, packed with actionable advice and vibrant descriptions.

3. **Headline:** "Embracing Body Positivity: Mini Dresses for Every Shape and Size"
   - **Overview:** This post aims to celebrate diverse body types and offer styling advice for mini dresses that flatter all figures.
   - **Target Audience:** Women of all shapes and sizes looking to embrace their bodies and step out confidently in mini dresses.
   - **Unique Angle:** Promotes body positivity and inclusivity, debunking myths about who can wear mini dresses.
   - **Value Proposition:** Readers gain confidence to wear mini dresses regardless of societal beauty standards, with tips tailored to various body types.
   - **Key Takeaways:** Understanding body shapes, selecting mini dresses that enhance features, and fostering a positive body image.
   - **Engagement Triggers:** Encourage reader stories about their journey to body acceptance and confidence in wearing mini dresses.
   - **SEO Keywords:** "Body positivity mini dresses," "mini dress for all sizes," "inclusive fashion tips."
   - **Tone and Style:** Empowering and supportive, with an emphasis on self-love and fashion freedom.

4. **Headline:** "The Sustainable Edit: Eco-Friendly Mini Dresses You'll Love"
   - **Overview:** Showcases brands and designers creating sustainable mini dresses, highlighting eco-friendly practices in fashion.
   - **Target Audience:** Eco-conscious consumers, fashionistas interested in sustainability, and supporters of ethical fashion.
   - **Unique Angle:** Focuses on the intersection of trendy fashion and environmental responsibility.
   - **Value Proposition:** Readers discover stylish and sustainable mini dress options, aligning their fashion choices with their values.
   - **Key Takeaways:** Brands leading in sustainable fashion, materials to look for in eco-friendly dresses, and the importance of sustainability in the industry.
   - **Engagement Triggers:** Ask readers to recommend their favorite sustainable fashion brands or share experiences with eco-friendly mini dresses.
   - **SEO Keywords:** "Sustainable mini dresses," "eco-friendly fashion wear," "ethical mini dress brands."
   - **Tone and Style:** Inspirational yet informative, sparking interest in sustainable fashion choices.

5. **Headline:** "Celebrity Influence: Iconic Mini Dress Moments in Pop Culture"
   - **Overview:** Chronicles memorable mini dress moments in film, music videos, and celebrity fashion, and their influence on popular culture.
   - **Target Audience:** Pop culture enthusiasts, movie buffs, and anyone intrigued by celebrity fashion trends.
   - **Unique Angle:** Analyzes how certain mini dresses worn by celebrities have shaped fashion trends and perceptions.
   - **Value Proposition:** Offers entertaining insights into the impact of celebrity fashion choices on the popularity of mini dresses.
   - **Key Takeaways:** Understanding of how media and celebrities can dictate fashion trends, with a focus on impactful mini dress moments.
   - **Engagement Triggers:** Invite readers to share their favorite celebrity mini dress moments and discuss their impact on personal style choices.
   - **SEO Keywords:** "Celebrity mini dress moments," "pop culture fashion," "iconic mini dresses in film."
   - **Tone and Style:** Engaging and nostalgic, filled with fascinating anecdotes and pop culture deep dives.
"""
sample_4 = """
1. **Headline:** "Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance"
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
"""

sample_5 = """
**Question 1:** A patient presents with a complaint of numbness and a tingling sensation in their right hand, primarily affecting the thumb, index, and middle fingers. How would you go about determining the peripheral nerve involved using the AMA Guides, and what steps would you take to evaluate the impairment?

**Answer 1:** To determine the involvement of the peripheral nerve based on the patient's symptoms of numbness and tingling affecting the thumb, index, and middle fingers of the right hand, refer to Figure 16-48 for cutaneous innervation and related nerves, and Table 16-12 for the origins and functions of peripheral nerves serving the upper extremities. These symptoms are suggestive of median nerve involvement. To evaluate the impairment:
1. Correlate the sensory deficits, namely numbness and tingling, with the median nerve distribution using Figures 16-48 and 16-49.
2. Identify any motor deficits or loss of power in key muscles of the are and relate this to the median nerve structure using Table 16-12 and Figure 16-47.
3. Grade the severity of sensory deficits according to Table 16-10a.
4. Determine the maximum impairment of the upper extremity due to sensory deficits of the median nerve from Table 16-15.
5. Multiply the grade of severity of the sensory deficit by the maximum upper extremity impairment value for the median nerve to determine the impairment percent.
6. If motor deficits are present, repeat steps 3 to 5 for motor impairment, then combine sensory and motor impairment percents using the Combined Values Chart to obtain the total upper extremity impairment due to median nerve involvement.

**Question 2:** A patient exhibits loss of motor function and diminished sensation in their left upper arm, with identifiable brachial plexus involvement. Walk through the process of determining the impairment rating according to the AMA Guides.

**Answer 2:** For a patient with brachial plexus involvement affecting the left upper arm:
1. Confirm the sensory deficits and diminished sensation and associate these with the regions served by the brachial plexus using Table 16-12 and Figures 16-48 through 16-50.
2. Determine the key muscles exhibiting loss of motor function related to the brachial plexus using Table 16-12 and Figure 16-47.
3. Grade the severity of both sensory deficits and motor loss using Tables 16-10a and 16-11a respectively.
4. Consult Table 16-14 to find the values for maximum impairment of the upper extremity due to deficits of the brachial plexus.
5. Calculate the upper extremity impairment percent for each function (sensory and motor) by multiplying the severity grade by the maximum impairment value.
6. Since the brachial plexus has mixed motor and sensory fibers, determine the upper extremity impairment for each function and then combine these using the Combined Values Chart to obtain the total upper extremity impairment value.
7. If applicable, convert this upper extremity impairment to whole person impairment using Table 16-3.

**Question 3:** How would you rate the upper extremity impairment in a patient with peripheral nerve damage affecting both the median and ulnar nerves?

**Answer 3:** To rate the upper extremity impairment for a patient with both median and ulnar nerves damage:
1. Localize the distribution of sensory deficits or pain, and motor loss related to both the median and ulnar nerves using Table 16-12 and Figures 16-47 through 16-50.
2. Grade the severity of both the sensory and motor deficits for each nerve individually using Tables 16-10a (for sensory) and 16-11a (for motor).
3. Find the maximum impairment values for individual nerves from Table 16-15 for both the median and ulnar nerves.
4. For each nerve, multiply the severity grade by the max impairment value to determine the impairment percent for each function (sensory and motor).
5. Combine the sensory and motor impairments for each nerve separately using the Combined Values Chart.
6. Combine the total impairment values for both the median and ulnar nerves using the Combined Values Chart to obtain the total upper extremity impairment.
7. If necessary, convert this to whole person impairment using Table 16-3.

**Question 4:** A patient reports sustained, burning pain in the forearm suggestive of CRPS I (RSD). How is this condition evaluated and rated for impairment based on the AMA Guides?

**Answer 4:** For a patient presenting with CRPS I (RSD) characterized by sustained, burning pain in the forearm, the evaluation and rating process would involve:
1. First, confirming the diagnosis of CRPS I through clinical assessment and correlating the symptoms to peripheral nerve damage that could contribute to CRPS I.
2. Since CRPS I involves complex pathophysiology, including both sensory disturbances and potentially motor function impairment, thorough documentation of all symptoms is crucial.
3. Grading the severity of sensory deficits or pain should accord with the specific criteria outlined for CRPS I, likely relying on patient descriptions and clinical findings.
4. Consult the AMA Guides to identify any specific guidelines for CRPS I, as treatment and impairment rating might differ from other peripheral nerve injuries. Typically, this requires a comprehensive approach that considers the multifaceted nature of CRPS I, including pain, sensitivity, and potential motor impairment.
5. Calculate the impairment based on the guidance for pain conditions and sensory deficits provided, often requiring a more subjective evaluation compared to direct nerve damage.
6. If CRPS I leads to motor deficits or other specific impairments, each should be evaluated and rated accordingly, then combined to reflect the total impairment accurately.
7. The assessment should conclude with determining the upper extremity impairment, which then may be converted into whole person impairment via Table 16-3, reflecting the broader impact of CRPS I on the individual's functions.

**Question 5:** When assessing a patient for workers' compensation due to bilateral upper extremity impairment resulting from peripheral nerve injuries, what steps should be followed according to the AMA Guides to calculate the total whole person impairment?

**Answer 5:** In assessing bilateral upper extremity impairment resulting from peripheral nerve injuries for workers' compensation purposes:
1. Start by individually assessing each upper extremity following the outlined steps for impairment determination due to peripheral nerve injuries, which involve localizing sensory and motor deficits, grading these deficits, and determining the impairment percent for each affected nerve structure.
2. For each extremity, calculate the impairment values for the involved nerve structures, then combine these values using the Combined Values Chart to obtain the total upper extremity impairment for each side.
3. Convert the total upper extremity impairment for each side into whole person impairment using Table 16-3, ensuring separate calculations for the left and right extremities.
4. Use the Combined Values Chart to combine the whole person impairment values for each side. Pay special attention to consult page 435 for comments and guidance on how to appropriately account for bilateral impressions, as bilateral conditions can have a more significant impact on a person's overall function and may not be simply additive.
5. This comprehensive approach ensures an accurate representation of the patient's total impairment, taking into account the combined effect of bilateral upper extremity impairments, which is crucial for fair compensation in workers' compensation cases.

**Question 1:** How is passive motion differentiated from assisted active motion in joint evaluation, and why is this distinction important when using the AMA Guides for impairment ratings?

**Answer 1:** Passive motion is generated by an external force to evaluate the range of motion of a joint when all muscles are relaxed, which helps determine the joint's freedom and range uninfluenced by patient effort. Assisted active motion, on the other hand, results from both active muscle contraction and an external force, stabilizing a segment to enhance muscle effectiveness in moving the joint. This distinction is crucial in the AMA Guides for accurately assessing impairment, as measurements of active motion are given precedence. Understanding the difference ensures that impairment evaluation is based on the joint's actual functional capacity rather than its potential movement, providing more relevant information for determining work-related impairments.

**Question 2:** Explain how the limitation of active and passive motion of upper extremity joints can occur, and discuss the implications for impairment ratings.

**Answer 2:** Limitations in active motion of upper extremity joints can result from failure in nerve, muscle, or tendon functionality required for motion execution. Passive motion limitations may arise from the joint itself, fixed contracture, or adherence/shortening of the antagonistic muscle or tendon. These nuances significantly impact impairment ratings, as they require sound clinical knowledge and precise measurement techniques to identify the underlying cause of motion restriction accurately. Recognizing whether the limitation is due to active or passive issues enables a more tailored, accurate impairment evaluation, directly affecting compensation and rehabilitation approaches.

**Question 3:** In the context of the AMA Guides, describe the correct technique for measuring the range of motion (ROM) of digital joints and explain its relevance to impairment evaluation.

**Answer 3:** Digital joints are measured with the wrist in a neutral position and the forearm pronated, stabilizing proximal joints in extension while flexing only the joint being measured, to ensure accuracy. This technique is relevant for impairment evaluation as it helps isolate individual joint function, necessary for a precise impairment rating. It acknowledges that simultaneous flexion of all three joints, as in making a fist, decreases active metacarpophalangeal joint flexion, potentially misleading the evaluation. Accurate joint measurement techniques, distinguishing individual joint movement from total active motion, enable a more refined assessment of digital impairment.

**Question 4:** Discuss the methodology behind calculating motion impairment for a digit using individual joint measurements and the total active range of motion, as per the AMA Guides.

**Answer 4:** The AMA Guides outline a methodology that differentiates between using individual joint measurements and the total active range of motion (TAM) for assessing digit impairment. Individual joint measurements focus on the ROM for each joint independently, critical for identifying specific limitations. Conversely, TAM measures the digit's full motion capacity as a unit, providing an overview of its functional impairment. The Guides specify different approaches for combining abnormal motion across multiple joints (as detailed on page 465), guiding the selection of the most accurate technique reflecting the existing impairment. This bifurcated approach allows for a comprehensive evaluation, ensuring impairments are accurately characterized and rated.

**Question 5:** Explain the principle for motion impairment calculation (A = E + F) in the AMA Guides and its importance in evaluating joint impairments.

**Answer 5:** The principle for motion impairment calculation, articulated as A = E + F, where A represents the total degrees of motion lost, E the degrees lost in extension, and F the degrees lost in flexion, is pivotal for quantifying joint impairments. This calculation method is used to determine the severity of motion restriction, particularly in cases of complete loss of motion or ankylosis, by tallying the lost degrees in both extension and flexion against the joint's normal arc of motion. This principle allows for a straightforward, quantifiable measurement of impairment, essential for assigning accurate impairment ratings. By representing the total impairment as a sum of losses in both directions, it ensures comprehensive assessment of the joint's functional limitations, crucial for determining appropriate compensation in Workers' Compensation cases.

**Question 1:** How do you approach the rating of a cervical spine impairment that leads to isolated bowel and/or bladder dysfunction but is not caused by corticospinal damage according to the AMA Guides?

**Answer 1:** In instances where a cervical spine impairment results in isolated bowel and/or bladder dysfunction not due to corticospinal damage, it's crucial to first obtain the impairment estimates for bowel and bladder dysfunction from the gastrointestinal (Chapter 6) and urology chapters (Chapter 7) of the AMA Guides. These estimates are then combined with the impairment rating for the cervical spine, selected from the DRE categories I to V as detailed in Table 15-5. This process ensures that both the primary impairment (cervical spine disorder) and the secondary impairments (bowel and bladder dysfunction) are appropriately evaluated and incorporated into the overall impairment rating, providing a comprehensive and accurate representation of the individual's condition.

**Question 2:** Describe the methodology for determining the neurologic level of involvement in spinal impairments as per the AMA Guides.

**Answer 2:** The neurologic level of involvement in spinal impairments is established by identifying the lowest normally functioning nerve root. This determination is made in relation to the vertebral level at which nerve roots exit the spine, as illustrated in Figure 15-5 of the AMA Guides. This assessment is critical for accurately diagnosing the extent and severity of spinal impairments, particularly those that could affect the nervous system. The methodology focuses on preserving accuracy in the correlation between nerve root function and vertebral anatomy, as it's pivotal for developing targeted and effective management plans for spinal impairments.

**Question 3:** How are spinal cord injuries evaluated in the current edition of the AMA Guides, especially concerning bilateral neurologic or corticospinal tract damage?

**Answer 3:** In the current edition of the AMA Guides, spinal cord injuries, especially those involving bilateral neurologic or corticospinal tract damage, are evaluated based on the criteria detailed in the nervous system chapter (Chapter 13). This marks a shift from prior editions where spinal cord injury ratings could employ a combination of DRE categories or assessments outlined in the nervous system chapter. For assessing bilateral or corticospinal tract impairments, consultation with a spinal cord injury specialist is recommended. Additionally, referencing Chapter 13, "The Central and Peripheral Nervous System," provides essential criteria for comprehensively evaluating such injuries. For impairments specifically affecting the upper extremities, use Table 15-6 and the corresponding impairment rating for one or both upper extremities is determined. For lower extremity loss of use, the section in Table 15-6 pertaining to station and gait impairment is used. Supplemental bowel or bladder dysfunctions are combined with the extremity impairments as warranted.

**Question 4:** Explain the process for converting a whole person spine impairment into a regional spine impairment as per Section 15.13 of the AMA Guides.

**Answer 4:** Section 15.13 of the AMA Guides details the process for translating a whole person spine impairment (WPI) into a regional spine impairment rating. This involves carefully combining and adjusting the values obtained from the specific impairment assessments (e.g., cervical, thoracic, lumbar) to accurately reflect the regional impact of the spine impairment. The process requires a comprehensive understanding of the various tables and categories detailed in the spine impairment sections (Tables 15-4 to 15-18), as well as proficiency in applying these criteria to calculate a region-specific impairment score. This translation from whole person to regional is essential for capturing the unique ways in which spinal impairments might affect an individual's functional capabilities within a specific spinal region, thus enabling more precise and tailored compensation or treatment strategies.

**Question 5:** How are impairment estimates for ankylosis and range of motion versus neurologic impairments differentiated within the AMA Guides?

**Answer 5:** Within the AMA Guides, the approach to rating impairments due to ankylosis and range of motion is distinct from the methodology applied to neurologic impairments. Ankylosis and range of motion impairment estimates are provided in Tables 15-8 through 15-14, focusing on the physical limitations and functional reductions in movement associated with spinal impairments. Conversely, neurologic impairments are assessed based on Tables 15-15 through 15-18, which concentrate on the degree of nervous system disruption and the associated sensory, motor, and reflex changes. This distinction recognizes the differential impacts of structural versus neurologic spinal impairments, facilitating a more nuanced and appropriate evaluation of the individual's overall impairment level. In both cases, the impairment estimates are expressed as whole person impairments, and the data employed in setting these standards reflect a combination of medical studies and consensus judgments, ensuring a robust and evidence-based framework for impairment rating.
"""
sample_6 = """
**Question 1:** A patient presents with a complaint of numbness and a tingling sensation in their right hand, primarily affecting the thumb, index, and middle fingers. How would you go about determining the peripheral nerve involved using the AMA Guides, and what steps would you take to evaluate the impairment?

**Answer 1:** To determine the involvement of the peripheral nerve based on the patient's symptoms of numbness and tingling affecting the thumb, index, and middle fingers of the right hand, refer to Figure 16-48 for cutaneous innervation and related nerves, and Table 16-12 for the origins and functions of peripheral nerves serving the upper extremities. These symptoms are suggestive of median nerve involvement. To evaluate the impairment:
1. Correlate the sensory deficits, namely numbness and tingling, with the median nerve distribution using Figures 16-48 and 16-49.
2. Identify any motor deficits or loss of power in key muscles of the are and relate this to the median nerve structure using Table 16-12 and Figure 16-47.
3. Grade the severity of sensory deficits according to Table 16-10a.
4. Determine the maximum impairment of the upper extremity due to sensory deficits of the median nerve from Table 16-15.
5. Multiply the grade of severity of the sensory deficit by the maximum upper extremity impairment value for the median nerve to determine the impairment percent.
6. If motor deficits are present, repeat steps 3 to 5 for motor impairment, then combine sensory and motor impairment percents using the Combined Values Chart to obtain the total upper extremity impairment due to median nerve involvement.

**Question 2:** A patient exhibits loss of motor function and diminished sensation in their left upper arm, with identifiable brachial plexus involvement. Walk through the process of determining the impairment rating according to the AMA Guides.

**Answer 2:** For a patient with brachial plexus involvement affecting the left upper arm:
1. Confirm the sensory deficits and diminished sensation and associate these with the regions served by the brachial plexus using Table 16-12 and Figures 16-48 through 16-50.
2. Determine the key muscles exhibiting loss of motor function related to the brachial plexus using Table 16-12 and Figure 16-47.
3. Grade the severity of both sensory deficits and motor loss using Tables 16-10a and 16-11a respectively.
4. Consult Table 16-14 to find the values for maximum impairment of the upper extremity due to deficits of the brachial plexus.
5. Calculate the upper extremity impairment percent for each function (sensory and motor) by multiplying the severity grade by the maximum impairment value.
6. Since the brachial plexus has mixed motor and sensory fibers, determine the upper extremity impairment for each function and then combine these using the Combined Values Chart to obtain the total upper extremity impairment value.
7. If applicable, convert this upper extremity impairment to whole person impairment using Table 16-3.

**Question 3:** How would you rate the upper extremity impairment in a patient with peripheral nerve damage affecting both the median and ulnar nerves?

**Answer 3:** To rate the upper extremity impairment for a patient with both median and ulnar nerves damage:
1. Localize the distribution of sensory deficits or pain, and motor loss related to both the median and ulnar nerves using Table 16-12 and Figures 16-47 through 16-50.
2. Grade the severity of both the sensory and motor deficits for each nerve individually using Tables 16-10a (for sensory) and 16-11a (for motor).
3. Find the maximum impairment values for individual nerves from Table 16-15 for both the median and ulnar nerves.
4. For each nerve, multiply the severity grade by the max impairment value to determine the impairment percent for each function (sensory and motor).
5. Combine the sensory and motor impairments for each nerve separately using the Combined Values Chart.
6. Combine the total impairment values for both the median and ulnar nerves using the Combined Values Chart to obtain the total upper extremity impairment.
7. If necessary, convert this to whole person impairment using Table 16-3.

**Question 4:** A patient reports sustained, burning pain in the forearm suggestive of CRPS I (RSD). How is this condition evaluated and rated for impairment based on the AMA Guides?

**Answer 4:** For a patient presenting with CRPS I (RSD) characterized by sustained, burning pain in the forearm, the evaluation and rating process would involve:
1. First, confirming the diagnosis of CRPS I through clinical assessment and correlating the symptoms to peripheral nerve damage that could contribute to CRPS I.
2. Since CRPS I involves complex pathophysiology, including both sensory disturbances and potentially motor function impairment, thorough documentation of all symptoms is crucial.
3. Grading the severity of sensory deficits or pain should accord with the specific criteria outlined for CRPS I, likely relying on patient descriptions and clinical findings.
4. Consult the AMA Guides to identify any specific guidelines for CRPS I, as treatment and impairment rating might differ from other peripheral nerve injuries. Typically, this requires a comprehensive approach that considers the multifaceted nature of CRPS I, including pain, sensitivity, and potential motor impairment.
5. Calculate the impairment based on the guidance for pain conditions and sensory deficits provided, often requiring a more subjective evaluation compared to direct nerve damage.
6. If CRPS I leads to motor deficits or other specific impairments, each should be evaluated and rated accordingly, then combined to reflect the total impairment accurately.
7. The assessment should conclude with determining the upper extremity impairment, which then may be converted into whole person impairment via Table 16-3, reflecting the broader impact of CRPS I on the individual's functions.

**Question 5:** When assessing a patient for workers' compensation due to bilateral upper extremity impairment resulting from peripheral nerve injuries, what steps should be followed according to the AMA Guides to calculate the total whole person impairment?

**Answer 5:** In assessing bilateral upper extremity impairment resulting from peripheral nerve injuries for workers' compensation purposes:
1. Start by individually assessing each upper extremity following the outlined steps for impairment determination due to peripheral nerve injuries, which involve localizing sensory and motor deficits, grading these deficits, and determining the impairment percent for each affected nerve structure.
2. For each extremity, calculate the impairment values for the involved nerve structures, then combine these values using the Combined Values Chart to obtain the total upper extremity impairment for each side.
3. Convert the total upper extremity impairment for each side into whole person impairment using Table 16-3, ensuring separate calculations for the left and right extremities.
4. Use the Combined Values Chart to combine the whole person impairment values for each side. Pay special attention to consult page 435 for comments and guidance on how to appropriately account for bilateral impressions, as bilateral conditions can have a more significant impact on a person's overall function and may not be simply additive.
5. This comprehensive approach ensures an accurate representation of the patient's total impairment, taking into account the combined effect of bilateral upper extremity impairments, which is crucial for fair compensation in workers' compensation cases.

**Question 6:** Is this a sample question?
1. Yes. It certainly is.
2. We have to do some testing.
- But we also need some bullet points.
- So, we can test the formatting.
"""
sample_7 = """
1. **Headline:** "Revolutionizing Chic: How The Mini Dress Redefined Modern Elegance"
   - **Overview:** This post delves into the intriguing evolution of the mini dress, unpacking its pivotal role in shaping contemporary fashion landscapes.
   - **Target Audience:** Fashion aficionados, cultural historians, and anyone captivated by the transformative journey of women's attire.

**Question 5:** Sample question?
1. This is Number One.
- This is a dash.
"""


def get_sample_fine_tuning_response():
    data = {
        "value": "**Question 1:** A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?\n\n**Answer 1:** For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities.\n\n**Question 2:** How should a physician determine the neurologic level of involvement for a cervical spine problem?\n\n**Answer 2:** The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating.\n\n**Question 3:** In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?\n\n**Answer 3:** For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions.\n\n**Question 4:** What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?\n\n**Answer 4:** Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments.\n\n**Question 5:** When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?\n\n**Answer 5:** The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings.",
        "signature": "OpenAIWrapperResponse",
        "processing": True,
        "source_info": {
            "source": "Manual Generic AMA Guide Source - Not Implemented Yet",
            "datapoint_id": 99999
        },
        "variable_name": "FINE_TUNE_DATA_SETS_1001",
        "processed_values": {
            "add_parts": {
                "args": {
                    "additional_parts": [
                        "You are an AMA Guide Expert."
                    ]
                },
                "value": [
                    {
                        "parts": [
                            "A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?",
                            "For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "How should a physician determine the neurologic level of involvement for a cervical spine problem?",
                            "The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?",
                            "For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?",
                            "Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?",
                            "The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings.",
                            "You are an AMA Guide Expert."
                        ]
                    }
                ],
                "depends_on": "clean_groups"
            },
            "data_groups": {
                "args": {
                    "parts_count": 2
                },
                "value": [
                    {
                        "parts": [
                            "Question 1: A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?",
                            "Answer 1: For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities."
                        ]
                    },
                    {
                        "parts": [
                            "Question 2: How should a physician determine the neurologic level of involvement for a cervical spine problem?",
                            "Answer 2: The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating."
                        ]
                    },
                    {
                        "parts": [
                            "Question 3: In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?",
                            "Answer 3: For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions."
                        ]
                    },
                    {
                        "parts": [
                            "Question 4: What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?",
                            "Answer 4: Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments."
                        ]
                    },
                    {
                        "parts": [
                            "Question 5: When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?",
                            "Answer 5: The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings."
                        ]
                    }
                ],
                "depends_on": "get_markdown_asterisk_structure"
            },
            "clean_groups": {
                "args": {},
                "value": [
                    {
                        "parts": [
                            "A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?",
                            "For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "How should a physician determine the neurologic level of involvement for a cervical spine problem?",
                            "The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?",
                            "For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?",
                            "Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments.",
                            "You are an AMA Guide Expert."
                        ]
                    },
                    {
                        "parts": [
                            "When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?",
                            "The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings.",
                            "You are an AMA Guide Expert."
                        ]
                    }
                ],
                "depends_on": "data_groups"
            },
            "oai_fine_tuning_data": {
                "args": {},
                "value": [
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AMA Guide Expert."
                            },
                            {
                                "role": "user",
                                "content": "A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?"
                            },
                            {
                                "role": "assistant",
                                "content": "For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities."
                            }
                        ]
                    },
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AMA Guide Expert."
                            },
                            {
                                "role": "user",
                                "content": "How should a physician determine the neurologic level of involvement for a cervical spine problem?"
                            },
                            {
                                "role": "assistant",
                                "content": "The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating."
                            }
                        ]
                    },
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AMA Guide Expert."
                            },
                            {
                                "role": "user",
                                "content": "In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?"
                            },
                            {
                                "role": "assistant",
                                "content": "For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions."
                            }
                        ]
                    },
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AMA Guide Expert."
                            },
                            {
                                "role": "user",
                                "content": "What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?"
                            },
                            {
                                "role": "assistant",
                                "content": "Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments."
                            }
                        ]
                    },
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an AMA Guide Expert."
                            },
                            {
                                "role": "user",
                                "content": "When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?"
                            },
                            {
                                "role": "assistant",
                                "content": "The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings."
                            }
                        ]
                    }
                ],
                "depends_on": "define_key_value_pairs"
            },
            "define_key_value_pairs": {
                "args": {
                    "keys": [
                        "user_content",
                        "assistant_content",
                        "system_content"
                    ]
                },
                "value": [
                    {
                        "user_content": "A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?",
                        "system_content": "You are an AMA Guide Expert.",
                        "assistant_content": "For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities."
                    },
                    {
                        "user_content": "How should a physician determine the neurologic level of involvement for a cervical spine problem?",
                        "system_content": "You are an AMA Guide Expert.",
                        "assistant_content": "The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating."
                    },
                    {
                        "user_content": "In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?",
                        "system_content": "You are an AMA Guide Expert.",
                        "assistant_content": "For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions."
                    },
                    {
                        "user_content": "What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?",
                        "system_content": "You are an AMA Guide Expert.",
                        "assistant_content": "Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments."
                    },
                    {
                        "user_content": "When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?",
                        "system_content": "You are an AMA Guide Expert.",
                        "assistant_content": "The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings."
                    }
                ],
                "depends_on": "add_parts"
            },
            "get_markdown_asterisk_structure": {
                "args": {},
                "value": {
                    "plain_text": [
                        "Question 1: A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?",
                        "Answer 1: For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities.",
                        "Question 2: How should a physician determine the neurologic level of involvement for a cervical spine problem?",
                        "Answer 2: The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating.",
                        "Question 3: In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?",
                        "Answer 3: For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions.",
                        "Question 4: What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?",
                        "Answer 4: Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments.",
                        "Question 5: When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?",
                        "Answer 5: The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings."
                    ],
                    "nested_structure": {
                        "Question 2: How should a physician determine the neurologic level of involvement for a cervical spine problem?": [],
                        "Question 4: What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?": [],
                        "Question 5: When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?": [],
                        "Question 3: In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?": [],
                        "Question 1: A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?": [],
                        "Answer 2: The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating.": [],
                        "Answer 3: For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions.": [],
                        "Answer 1: For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities.": [],
                        "Answer 4: Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments.": [],
                        "Answer 5: The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings.": []
                    }
                },
                "depends_on": "content"
            }
        }
    }

    return data


def get_sample_fine_tuning_response_2():
    data = {
        "value": "Summary or content here",
        "signature": "Identifier",
        "processing": True,
        "source_info": {
            "source": "Information source",
            "datapoint_id": 99999
        },
        "variable_name": "DATA_SET_IDENTIFIER",
        "processed_values": {
            "define_key_value_pairs": {
                "args": {
                    "keys": ["user_content", "assistant_content", "system_content"]
                },
                "value": [
                    {
                        "user_content": "User's question or statement",
                        "assistant_content": "Assistant's response or content",
                        "system_content": "System's directive or content"
                    },
                    {
                        "user_content": "User's send question or statement",
                        "assistant_content": "Assistant's response or content",
                        "system_content": "System's directive or content"
                    }
                ],
                "depends_on": "Dependency identifier or condition"
            }
        }
    }
    return data


def get_sample_training_data():
    data = [
        {
            "user_content": "A patient sustains a cervical spine injury resulting in isolated bladder dysfunction but no direct corticospinal damage. Which chapters should be consulted for impairment ratings, and how should these ratings be combined?",
            "system_content": "You are an AMA Guide Expert.",
            "assistant_content": "For a patient with a cervical spine injury that leads to isolated bladder dysfunction without corticospinal damage, the physician should refer to Chapters 6 (Gastrointestinal System) and 7 (Urology) for obtaining appropriate estimates for bladder dysfunction. These estimates should then be combined with the applicable Diagnostic Related Estimate (DRE) cervical spine category, ranging from DRE I to V as listed in Table 15-5. This combination approach ensures a comprehensive impairment rating that accurately reflects the patient's condition and the impacts on their daily living activities."
        },
        {
            "user_content": "How should a physician determine the neurologic level of involvement for a cervical spine problem?",
            "system_content": "You are an AMA Guide Expert.",
            "assistant_content": "The neurologic level of involvement for cervical spine issues is determined by identifying the lowest normally functioning nerve root. This can be done by referring to Figure 15-5, which illustrates the relationship of nerve roots to the vertebral level. Understanding the level at which nerve roots exit the spine relative to the vertebrae is crucial in accurately determining the neurologic level of involvement and, therefore, in providing an accurate impairment rating."
        },
        {
            "user_content": "In the case of spinal cord injuries involving bilateral upper extremity impairments, which resources or chapters should be reviewed for an accurate impairment rating according to the AMA Guides?",
            "system_content": "You are an AMA Guide Expert.",
            "assistant_content": "For bilateral upper extremity impairments resulting from spinal cord injury, it is recommended to consult Chapter 13, The Central and Peripheral Nervous System, for guidance. Additionally, Table 15-6 should be used to determine the appropriate impairment rating for one or both upper extremities. This case may also require a consultation with a spinal cord injury specialist to ensure a comprehensive and accurate assessment, especially considering the complexity of spinal cord injuries and their impact on bodily functions."
        },
        {
            "user_content": "What is the process for expressing a whole person spine impairment as a regional spine impairment in the AMA Guides?",
            "system_content": "You are an AMA Guide Expert.",
            "assistant_content": "Expressing a whole person spine impairment as a regional spine impairment involves a specific process detailed in Section 15.13 of the AMA Guides. This section outlines how to translate whole person impairment estimates, which are the global assessments of impairment, into focused, region-specific impairment values. This translation is crucial in cases like workers' compensation, where specific regional impairments may need to be precisely quantified for legal or insurance purposes. The section provides clear guidelines for accurately interpreting and applying the whole person impairments to the regional context, ensuring practitioners can provide detailed, legally, and medically relevant assessments."
        },
        {
            "user_content": "When evaluating spinal cord injuries, what changes were made in the current edition of the AMA Guides regarding the rating approach?",
            "system_content": "You are an AMA Guide Expert.",
            "assistant_content": "The current edition of the AMA Guides introduced a significant change in the approach to rating spinal cord injuries. Instead of relying on a combination of DRE categories or solely the nervous system chapter, as in prior editions, the current approach mandates evaluating spinal cord injuries based on criteria specified in the nervous system chapter (Chapter 13). This change underscores the complex nature of spinal cord injuries and the necessity for detailed, nuanced assessments. By focusing the evaluation criteria within the nervous system chapter and emphasizing the use of specific tables like Table 15-6 for upper and lower extremities impairment and additional dysfunctions, the Guides enhance the accuracy and comprehensiveness of spinal cord injury impairment ratings."
        }
    ]

    fine_tuning_data = data
    return fine_tuning_data


def get_sample_data(sample_options=sample_7):
    sample_data = sample_options

    return sample_data


async def get_sample_data_async(sample_options=sample_1):
    sample_data = sample_options

    return sample_data


if __name__ == "__main__":
    print(get_sample_data())
