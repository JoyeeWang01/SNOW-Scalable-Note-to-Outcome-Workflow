"""
Utility functions for multi-agent oncology feature extraction.
"""

import json
import re
from typing import Any, Dict, List
from copy import deepcopy


def extract_json_from_tags(text: str) -> str:
    """Extract JSON content between <JSON> and </JSON> tags or markdown code blocks."""

    # Handle None input
    if text is None:
        raise ValueError("Input text is None - no response received from LLM")

    # First try to find <JSON> tags
    start_tag = "<JSON>"
    end_tag = "</JSON>"

    start_idx = text.find(start_tag)
    if start_idx != -1:
        end_idx = text.find(end_tag, start_idx)
        if end_idx != -1:
            # Extract content between tags
            json_content = text[start_idx + len(start_tag):end_idx].strip()
            return json_content

    # If no <JSON> tags found, try to find markdown code blocks with ```json
    if "```json" in text:
        start_idx = text.find("```json")
        if start_idx != -1:
            start_idx += 7  # Move past ```json
            # Handle potential newline after ```json
            if start_idx < len(text) and text[start_idx] == '\n':
                start_idx += 1
            end_idx = text.find("```", start_idx)
            if end_idx != -1:
                return text[start_idx:end_idx].strip()

    # Try generic markdown code blocks ```
    if "```" in text:
        start_idx = text.find("```")
        if start_idx != -1:
            # Move past the opening ```
            start_idx += 3
            # Skip any language identifier on the same line
            newline_idx = text.find("\n", start_idx)
            first_char_idx = start_idx
            # Check if there's content on the same line (language identifier)
            if newline_idx != -1 and newline_idx > start_idx:
                # Check if the content between ``` and newline is just a language identifier
                potential_lang = text[start_idx:newline_idx].strip()
                if len(potential_lang) < 20 and not potential_lang.startswith("{") and not potential_lang.startswith("["):
                    # It's likely a language identifier, skip to next line
                    first_char_idx = newline_idx + 1

            end_idx = text.find("```", first_char_idx)
            if end_idx != -1:
                content = text[first_char_idx:end_idx].strip()
                # Basic check if it looks like JSON
                if content.startswith("{") or content.startswith("["):
                    return content

    raise ValueError("No JSON content found in <JSON> tags or markdown code blocks")


def flatten_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand every feature that contains a ``specific_regions`` list into one
    feature per region, then rewrite any ``aggregated_from`` lists so they
    reference the newly-created feature names.

    Parameters
    ----------
    features : list of dict
        The original feature specification list.

    Returns
    -------
    list of dict
        A new list with all region-specific features flattened and all
        ``aggregated_from`` relationships updated.
    """
    flattened: List[Dict[str, Any]] = []
    # maps *original* feature_name ➜ list of flattened names (or itself)
    name_map: Dict[str, List[str]] = {}

    # ---------- 1. Expand region-specific features ----------
    for feat in features:
        regions = feat.get("specific_subgroups")
        if regions:                                               # needs flattening
            generated_names = []
            for region in regions:
                new_feat = deepcopy(feat)
                new_feat.pop("specific_subgroups", None)            # no longer needed
                new_name = f"{feat['feature_name']}_{region}"
                new_feat["feature_name"] = new_name
                flattened.append(new_feat)
                generated_names.append(new_name)
            name_map[feat["feature_name"]] = generated_names
        else:                                                     # keep as-is
            new_feat = deepcopy(feat)
            flattened.append(new_feat)
            name_map[feat["feature_name"]] = [feat["feature_name"]]

    # ---------- 2. Fix up every aggregated_from list ----------
    for feat in flattened:
        if "aggregated_from" in feat:
            aggregated_from = feat["aggregated_from"]

            # Ensure aggregated_from is a list
            if not isinstance(aggregated_from, list):
                # Convert to list if it's not already
                aggregated_from = [aggregated_from]
                feat["aggregated_from"] = aggregated_from

            new_sources: List[str] = []
            for base_item in aggregated_from:
                # Handle both string and dict formats
                if isinstance(base_item, str):
                    # If it's already a string, find the feature in the full list to check for subgroups
                    base_feature = next((f for f in features if f["feature_name"] == base_item), None)
                    if base_feature and "specific_subgroups" in base_feature:
                        # Expand the subgroups
                        for region in base_feature["specific_subgroups"]:
                            expanded_name = f"{base_item}_{region}"
                            new_sources.append(expanded_name)
                    else:
                        # No subgroups, use the string directly
                        new_sources.append(base_item)
                else:
                    # If it's a dict, extract feature_name and handle subgroups
                    base_name = base_item.get("feature_name", "")
                    # If this dict item has specific_subgroups, we need to expand those too
                    if "specific_subgroups" in base_item:
                        for region in base_item["specific_subgroups"]:
                            expanded_name = f"{base_name}_{region}"
                            new_sources.append(expanded_name)
                    else:
                        new_sources.append(base_name)
            feat["aggregated_from"] = new_sources

            # After expansion, if aggregated_from has 1 or fewer sources, mark as non-aggregated
            if len(new_sources) <= 1:
                feat.pop("aggregated_from", None)
                feat["is_aggregated"] = False

    return flattened
