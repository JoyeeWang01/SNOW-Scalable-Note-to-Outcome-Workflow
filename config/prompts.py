"""
Prompt templates for multi-agent oncology feature extraction and validation.
"""

FEATURE_PROPOSAL_TEMPLATE = (
"""You are a clinical machine learning scientist developing a prognosis model.
The clinical notes you will receive are: {notes_description}.
Your prediction target is: {outcome_description}.

Identify the most clinically relevant structured features—either categorical or numerical—commonly found in such reports.
You have access to each clinical note (indexed 0 to {MAX_NOTE_INDEX}) via the get_note tool. You are expected to identify patterns, structures, and field repetitions (e.g., per-region or per-sample reporting) by reading and analyzing these notes directly.
Please read at least 6 notes.

Do not include the following features, as they are already available:
{structured_feature_names}

Important context:
Clinical notes may contain multiple anatomical regions, repeated measurements, or temporal sequences of observations. It is your responsibility to identify such structures from the notes and propose features that are specific to these subgroups or aggregated across them.

Feature requirements:
	1.	Each feature must be directly usable in a regression model: clearly defined and numerically or categorically encoded.
	2.	Features can be:
	•	Specific_subgroups: If the notes report values per anatomical region, time point, or other subgroup, identify all distinct subgroups and propose one feature per subgroup.
	•	Aggregated: If multiple features are summarized using functions like mean, max, or count, clearly indicate this.

If a feature varies across subgroups (e.g., different values per region or timepoint), then:
1. Return only one feature object, with the specific_subgroups field listing all relevant subgroups.
2. We would like to extract the feature for each subgroup separately.

If a feature is aggregated from multiple fields, for example, *max*, *mean*, *count*, *sum*, you must:
1. Set "is_aggregated": True
2. In "aggregated_from", include a list of JSON objects, each containing:
   - "feature_name": the base feature name
   - (if applicable) "specific_subgroups": a list of the associated subgroup names
3. **Make sure that each of those base features is also included as a feature object in the JSON output**

If the feature is not aggregated, set "is_aggregated": False and leave out "aggregated_from".

For each feature, include:
1. Feature Name
2. Description and clinical rationale for inclusion
3. A sample extraction rule as instructions (if the feature is categorical, explicitly list all possible categories and assign a numeric code for each category)
4. Whether the feature is aggregated ('True' or 'False')
5. If aggregated, which features it was aggregated from

Return your output as a JSON array of objects, where each object has the following keys:
- "feature_name"
- *(optional)* "specific_subgroups" (a list of subgroup names if values vary across subgroups)
- "description" (include rationale for why this feature relates to the prediction target)
- "instructions" (if the feature is categorical, explicitly list all possible categories and their numeric codes)
- "is_aggregated" (boolean)
- *(optional)* "aggregated_from" (JSON object array, required only if is_aggregated is True)

Return the JSON array between <JSON> and </JSON> tags."""
)

FEATURE_EXTRACTION_TEMPLATE = (
"""You are tasked with extracting specific feature from a patient's clinical note. The clinical notes you will receive are {notes_description}.
The note is provided below:

<clinical_note>
{note}
</clinical_note>

Your task is to extract the following features from the clinical note:

<features_detail>
{features_detail}
</features_detail>

1. Consider the different ways each feature might be described or implied in clinical language.
2. Apply different strategies to identify, infer, or calculate each feature based on the available information in the note.
3. If a feature is truly not found or not extractable, set "value" to null.
4. Make sure that all extracted values are returned strictly as numerical values (integers or floats) without any text, units, or symbols.

After your analysis, provide the extracted data in JSON format. The JSON object should contain the following keys:
{feature_list}

Return the JSON object between <JSON> and </JSON> tags."""
)

FEATURE_VALIDATION_TEMPLATE = (
"""You are a senior clinical data scientist. Your job is to review a feature that was extracted from clinical notes and decide whether to:
1. proceed - if the feature is consistently and correctly extracted.
2. remove - if the feature cannot be reliably extracted due to too many absences from the notes.
3. reextract - if the feature could be consistently extracted but extraction logic needs revision or the values require post-processing (e.g., formatting, grouping, normalization).

The clinical notes you will receive are: {notes_description}.
Your prediction target is: {outcome_description}.

The current feature is:
<feature_detail>
{feature_detail}
</feature_detail>

The **extracted values**, keyed by clinical note index (0–{MAX_NOTE_INDEX}):
<extracted_values>
{extracted_values}
</extracted_values>

The extracted values will be used in a regression model. Please analyze the consistency, accuracy, and usefulness of this feature extraction. You have access to each of the clinical notes (indexed from 0 to {MAX_NOTE_INDEX}) via the 'get_note' tool. Use it to validate extracted values against the original notes as needed.

For your reference, this feature has been extracted {extraction_count} times and validated {validation_count} times.
Features that are already included in the model are: {all_feature_names}

For missing values:
1. {missing_percent} percent of the values are currently missing.
2. Validate against at least 4 of the original notes to see whether these are true absences or extraction mistakes.
3. Apply different strategies to identify, infer, or calculate the feature based on the available information in the notes.
4. If a new strategy works, update the instructions and re-extract the feature.
5. Try reduce the missing values as much as possible through better extraction.
6. If the missing values are due to true absences and the missing rate remains too high to support meaningful analysis, remove the feature.

Special Instructions for Categorical Features:

If the feature is **categorical**, additionally evaluate:
- Are the **categories consistent** in terminology, format, and meaning?
- Do the **categories make clinical and modeling sense**?
- Are there semantically redundant or overlapping categories?
- Is the **number of categories appropriate** (e.g., not overly granular)?

If the current categories are unclear or inconsistent:
- Define a clear set of **numerical categories**
- Recommend reextract if the extraction logic must be updated to match these categories or the extracted values are correct, but can be transformed or mapped into numerical categories

### Additional Features/Columns/Indicators:
If you believe new features (e.g. binary indicator or count) should be derived from or replace the current feature, propose them in a JSON array. Each object should include:
- "feature_name"
- *(optional)* "specific_subgroups" (a list of subgroup names if values vary across subgroups)
- "description" (include rationale for why this feature relates to the prediction target)
- "instructions" (if the feature is categorical, explicitly list all possible categories and their numeric codes)

New features will be reextracted from the notes. If the newly suggested features can effectively replace the current feature, your final decision must be **remove**, meaning removing the current feature and keeping the new features.

### Formatting Requirement:
Make sure that all extracted values are returned strictly as **numerical values (integers or floats) without any text, units, or symbols**. If a value cannot be determined, return `null`.

### Final Output:
Return your decision as a JSON object with this structure:
{{
    "decision": proceed|remove|reextract,
    "add_additional_feature": null or JSON array of the suggested additional features
    "reasoning": Brief explanation of your decision,
    "current_feature_instructions": If decision is "reextract", provide concrete guidance for improving category mapping or extraction logic. Reference specific input examples where possible. **IMPORTANT**: DON'T mention additional features/columns/indicators here.
}}

Return the JSON object between <JSON> and </JSON> tags."""
)

FEATURE_ALIGNMENT_TEMPLATE = (
"""You are a clinical machine learning scientist performing **feature validation**.  
You previously generated a set of candidate features from clinical notes.  
The clinical notes you will receive are: {notes_description}.
Your prediction target is: {outcome_description}.
Now, you must carefully review each feature recommendation against 10 of the actual notes given.

Each feature has the following fields:
- "feature_name" 
- *(optional)* "specific_subgroups" (a list of subgroup names if values vary across subgroups)
- "description" (include rationale for why this feature relates to the prediction target)
- "instructions" (if the feature is categorical, explicitly list all possible categories and their numeric codes)
- "is_aggregated" (boolean)
- *(optional)* "aggregated_from" (JSON object array, required only if is_aggregated is True)

Your tasks:  
1. **Confirm relevance**: For each feature, check whether the concept is consistently present and clinically meaningful in these notes. Mark features that are too rare, ambiguous, redundant with existing structured features, or unlikely to be extractable.  
2. **Suggest edits**: If a feature is useful but poorly defined, suggest better subgroup definitions, sharper extraction rules, or (for categorical features) a better list of categories with explicit numeric codes.  
3. **Identify gaps**: While reviewing the notes, look for clinically important elements that were not included in the current candidate feature list. Propose additional features following the same schema.  
4. **Aggregation consistency**: For aggregated features, verify that the underlying base features exist, are valid, and are not double-counted.  

Feature requirements:
	1.	Each feature must be directly usable in a regression model: clearly defined and numerically or categorically encoded.
	2.	Features can have:
	•	Specific_subgroups: If the notes report values per anatomical region, time point, or other subgroup, identify all distinct subgroups and propose one feature per subgroup.
	•	Aggregated: If multiple features are summarized using functions like mean, max, or count, clearly indicate this.

If a feature varies across subgroups (e.g., different values per region or timepoint), then:
1. Return only one feature object, with the specific_subgroups field listing all relevant subgroups.
2. We would like to extract the feature for each subgroup separately.

If a feature is aggregated from multiple fields, for example, *max*, *mean*, *count*, *sum*, you must:
1. Set "is_aggregated": True
2. In "aggregated_from", include a list of JSON objects, each containing:
   - "feature_name": the base feature name
   - (if applicable) "specific_subgroups": a list of the associated subgroup names
3. **Make sure that each of those base features is also included as a feature object in the JSON output**

If the feature is not aggregated, set "is_aggregated": False and leave out "aggregated_from".

### Final output rules:  
Return an updated JSON array of feature objects between `<JSON>` and `</JSON>` tags, where:  
- **New features (`"status": "new"`)** → Return the **full feature object** with all required fields:  
  - `"feature_name"`, optional `"specific_subgroups"`, `"description"`, `"instructions"`, `"is_aggregated"`, optional `"aggregated_from"`, and `"status"`.  
- **Edited features (`"status": "edited"`)** → Return only the **updated fields** (not the full object) along with `"feature_name"` and `"status"`.  
- **Aligned features (`"status": "aligned"`)** → Return only `"feature_name"` and `"status"`.  
- **Dropped features (`"status": "drop"`)** → Return only `"feature_name"`, `"status"`, and a short `"rationale"`.

Features that you recommend for **dropping must still be included in the output with "status": "drop" and a rationale. These drops are not final—they may be reversed later if evidence from other notes supports keeping the feature.

### Constraints:  
- Do not include the following features, as they are already available:  
{structured_feature_names}  
- Ensure each feature is well-defined for direct use in a regression model.  
- All feature values must be strictly numerical (integers or floats) with no text, units, or symbols.

Note #1:
{note_1}

Note #2:
{note_2}

Note #3:
{note_3}

Note #4:
{note_4}

Note #5:
{note_5}

Note #6:
{note_6}

Note #7:
{note_7}

Note #8:
{note_8}

Note #9:
{note_9}

Note #10:
{note_10}

Candidate features:
{features}"""
)

MERGE_FEATURE_TEMPLATE = (
"""You are a clinical machine learning scientist performing **feature validation**.
You previously generated {num_chunks} sets of candidate features from reviewing {num_chunks} subsets of clinical notes.
The clinical notes you received were: {notes_description}.
Your prediction target is: {outcome_description}.

### Your task:
1. Take the **union** of the {num_chunks} sets of candidate features and return a **single consolidated set** of features.
2. If multiple features are very similar or redundant, keep only one of them (the clearest and most clinically relevant).

### Final output:
Return an updated JSON array of feature objects between `<JSON>` and `</JSON>` tags, where:
Each feature must include:
- "feature_name"
- *(optional)* "specific_subgroups" (a list of subgroup names if values vary across subgroups)
- "description" (include rationale for why this feature relates to the prediction target)
- "instructions" (if the feature is categorical, explicitly list all possible categories and their numeric codes)
- "is_aggregated" (boolean)
- *(optional)* "aggregated_from" (JSON object array, required only if is_aggregated is True)

### Inputs:
{feature_sets}

"""
)

AGGREGATION_CODE_TEMPLATE = (
"""You are an expert Python data scientist.
Below is ONE aggregated clinical feature:

<aggregated_feature>
{aggregated_feature}
</aggregated_feature>

Write a pure-Python function:

def aggregate_<feature_name>(features: dict[str, Any]) -> Any:
    \"\"\"Return the aggregated value or None.\"\"\"

Rules
------
* "features" contains the base feature keys listed in "aggregated_from".
* Treat None as missing.
* If the aggregation is an average/mean/min/max/sum, ignore None values in the mean/min/max/sum. Return None if *all* constituents are None.
* Use only the Python stdlib and numpy as np.
* If unable to write the function given the provided base features, return <ERROR> followed by a string explaining why.

Below are descriptions of the base feature keys listed in "aggregated_from":
{aggregated_from_features}

ONLY output valid Python code — no markdown or extra text.
"""
)
