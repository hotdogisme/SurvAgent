"""Bladder pathology feature extraction from get_feature.py."""

import json
import re
from typing import Dict, List

from .prompt_templates import fill_prompt, prompt_text


def extract_bladder_cancer_features(report: str, features_to_extract: List[str] = None, *, backend) -> Dict[str, str]:

    if features_to_extract is None:
        features_to_extract = [
            "Tumor Grade", "Pathological Stage", "Depth of Invasion", "Lymphovascular Invasion", "Perineural Invasion",
            "Lymph Node Metastasis", "Margin Status", "Tumor Number", "Tumor Size", "Carcinoma in Situ",
            "Variant Histology", "Squamous Differentiation", "Glandular Differentiation", "Micropapillary Component", "Plasmacytoid Component",
            "Sarcomatoid Differentiation", "Lymphocytic Infiltration", "Necrosis Percentage", "Ki67 Index", "PD-L1 Expression"
        ]
    
    feature_template = _build_bladder_cancer_template(features_to_extract)
    
    prompt = fill_prompt(
        prompt_text("wsi", "bladder_checklist.txt"),
        report=report,
        features_to_extract=", ".join(features_to_extract),
        feature_template=feature_template,
    )

    # response = img_api(img_path= img_path, user_input= prompt)

    response = backend.chat(prompt)

    features_dict = _parse_llm_response(response, features_to_extract)

    validated_features = _validate_bladder_cancer_features(features_dict, features_to_extract)
    
    return validated_features


def _parse_llm_response(response: str, expected_features: List[str]) -> Dict[str, str]:
    """Parse LLM response and extract feature dictionary"""
    
    # Method 1: Try direct JSON parsing
    try:
        # Find JSON object
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            json_str = json_match.group()
            features_dict = json.loads(json_str)
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in features_dict:
                    features_dict[feature] = "Unknown"
                    
            return features_dict
    except:
        pass
    
    # Method 2: If JSON parsing fails, try line-by-line parsing
    features_dict = {}
    lines = response.strip().split('\n')
    
    for line in lines:
        # Look for "feature: value" pattern
        match = re.search(r'["\']?([^"\':]+)["\']?\s*[：:]\s*["\']?([^"\'\n]+)["\']?', line)
        if match:
            feature_name = match.group(1).strip()
            feature_value = match.group(2).strip()
            
            # Check if it's an expected feature
            for expected_feature in expected_features:
                if expected_feature in feature_name or feature_name in expected_feature:
                    features_dict[expected_feature] = feature_value
                    break
    
    # Fill missing features
    for feature in expected_features:
        if feature not in features_dict:
            features_dict[feature] = "Unknown"
    
    return features_dict


def _build_bladder_cancer_template(features: List[str]) -> str:
    """Build JSON template for bladder cancer feature extraction"""
    
    # Bladder cancer feature value options mapping
    bladder_cancer_options = {
        # Basic features
        "Tumor Grade": ["Low Grade", "High Grade", "Unknown"],
        "Pathological Stage": ["Ta", "T1", "T2", "T3", "T4", "Tis", "Unknown"],
        "Depth of Invasion": ["Non-invasive", "Lamina propria invasion", "Muscularis propria invasion", "Deep muscle invasion", "Unknown"],
        "Lymphovascular Invasion": ["Present", "Absent", "Unknown"],
        "Perineural Invasion": ["Present", "Absent", "Unknown"],
        "Lymph Node Metastasis": ["None", "1-2 nodes", "3-10 nodes", ">10 nodes", "Unknown"],
        "Margin Status": ["Negative", "Positive", "Unknown"],
        
        # Tumor morphological features
        "Tumor Number": ["Single", "Multiple", "Unknown"],
        "Tumor Size": ["<3cm", "3-5cm", ">5cm", "Specific measurement", "Unknown"],
        "Tumor Morphology": ["Papillary", "Nodular", "Ulcerative", "Diffuse", "Unknown"],
        
        # Histological features
        "Carcinoma in Situ": ["Present", "Absent", "Extensive", "Unknown"],
        "Variant Histology": ["Urothelial carcinoma", "Squamous differentiation", "Glandular differentiation", "Micropapillary", "Plasmacytoid", 
                            "Sarcomatoid", "Nested", "Lymphoepithelioma-like", "Unknown"],
        "Squamous Differentiation": ["Present", "Absent", "Focal", "Extensive", "Unknown"],
        "Glandular Differentiation": ["Present", "Absent", "Focal", "Extensive", "Unknown"],
        "Micropapillary Component": ["Present", "Absent", "Focal", "Extensive", "Unknown"],
        "Plasmacytoid Component": ["Present", "Absent", "Focal", "Extensive", "Unknown"],
        "Sarcomatoid Differentiation": ["Present", "Absent", "Focal", "Extensive", "Unknown"],
        
        # Microenvironment features
        "Lymphocytic Infiltration": ["Minimal", "Mild", "Moderate", "Abundant", "Unknown"],
        "Necrosis Percentage": ["None", "<10%", "10-30%", "30-50%", ">50%", "Unknown"],
        
        # Molecular features (if mentioned in report)
        "Ki67 Index": ["<10%", "10-30%", "30-50%", ">50%", "Unknown"],
        "PD-L1 Expression": ["Negative", "Low expression", "High expression", "Unknown"],
        "FGFR3 Mutation": ["Present", "Absent", "Unknown"],
        "P53 Expression": ["Normal", "Overexpression", "Loss", "Unknown"]
    }
    
    template_lines = []
    for feature in features:
        if feature in bladder_cancer_options:
            options = "/".join(bladder_cancer_options[feature])
            template_lines.append(f'  "{feature}": "{options}"')
        else:
            # For unknown features, provide generic template
            template_lines.append(f'  "{feature}": "..."')
    
    return '{\n' + ',\n'.join(template_lines) + '\n}'


def _validate_bladder_cancer_features(features_dict: Dict[str, str], expected_features: List[str]) -> Dict[str, str]:
    """Validate bladder cancer feature extraction results"""
    
    validated_dict = {}
    
    for feature in expected_features:
        value = features_dict.get(feature, "Unknown")
        
        # Bladder cancer-specific feature value cleaning and standardization
        cleaned_value = _clean_bladder_cancer_value(feature, value)
        validated_dict[feature] = cleaned_value
    
    return validated_dict


def _clean_bladder_cancer_value(feature: str, value: str) -> str:
    """Clean and standardize bladder cancer feature values"""
    
    if not value or value.strip() == "":
        return "Unknown"
    
    value = value.strip().lower()
    
    # Bladder cancer-specific feature value standardization
    if feature == "Tumor Grade":
        if any(term in value for term in ["low grade", "low-grade"]):
            return "Low Grade"
        elif any(term in value for term in ["high grade", "high-grade"]):
            return "High Grade"
        else:
            return "Unknown"
    
    elif feature == "Pathological Stage":
        if "ta" in value or "non-invasive" in value:
            return "Ta"
        elif "t1" in value or "lamina propria" in value:
            return "T1"
        elif "t2" in value or "superficial muscle" in value:
            return "T2"
        elif "t3" in value or "deep muscle" in value:
            return "T3"
        elif "t4" in value or "adjacent organs" in value:
            return "T4"
        elif "tis" in value or "in situ" in value:
            return "Tis"
        else:
            return "Unknown"
    
    elif feature == "Depth of Invasion":
        if any(term in value for term in ["non-invasive", "non invasive", "ta"]):
            return "Non-invasive"
        elif any(term in value for term in ["lamina propria", "t1"]):
            return "Lamina propria invasion"
        elif any(term in value for term in ["muscularis propria", "superficial muscle", "t2"]):
            return "Muscularis propria invasion"
        elif any(term in value for term in ["deep muscle", "t3"]):
            return "Deep muscle invasion"
        else:
            return "Unknown"
    
    elif feature == "Tumor Number":
        if "single" in value or "solitary" in value:
            return "Single"
        elif "multiple" in value or "multifocal" in value:
            return "Multiple"
        else:
            return "Unknown"
    
    elif feature == "Carcinoma in Situ":
        if any(term in value for term in ["present", "identified", "seen", "tis"]):
            if "extensive" in value:
                return "Extensive"
            else:
                return "Present"
        elif any(term in value for term in ["absent", "not identified", "not seen"]):
            return "Absent"
        else:
            return "Unknown"
    
    elif feature == "Variant Histology":
        if "urothelial" in value and not any(other in value for other in ["squamous", "glandular", "micropapillary"]):
            return "Urothelial carcinoma"
        elif "squamous" in value:
            return "Squamous differentiation"
        elif "glandular" in value or "adenocarcinoma" in value:
            return "Glandular differentiation"
        elif "micropapillary" in value:
            return "Micropapillary"
        elif "plasmacytoid" in value:
            return "Plasmacytoid"
        elif "sarcomatoid" in value:
            return "Sarcomatoid"
        elif "nested" in value:
            return "Nested"
        elif "lymphoepithelioma" in value:
            return "Lymphoepithelioma-like"
        else:
            return "Unknown"
    
    elif feature in ["Squamous Differentiation", "Glandular Differentiation", "Micropapillary Component", "Plasmacytoid Component", "Sarcomatoid Differentiation"]:
        if any(term in value for term in ["present", "identified", "seen"]):
            if "extensive" in value or "prominent" in value:
                return "Extensive"
            elif "focal" in value or "limited" in value:
                return "Focal"
            else:
                return "Present"
        elif any(term in value for term in ["absent", "not identified", "not seen"]):
            return "Absent"
        else:
            return "Unknown"
    
    elif feature == "Tumor Size":
        if any(term in value for term in ["<3", "less than 3", "small"]):
            return "<3cm"
        elif any(term in value for term in ["3-5", "3 to 5", "medium"]):
            return "3-5cm"
        elif any(term in value for term in [">5", "greater than 5", "large"]):
            return ">5cm"
        elif re.search(r'\d+\.?\d*\s*[×x]\s*\d+\.?\d*\s*cm', value) or re.search(r'\d+\.?\d*\s*cm', value):
            # Extract specific measurements
            size_match = re.search(r'(\d+\.?\d*)\s*[×x]?\s*(\d+\.?\d*)?\s*cm', value)
            if size_match:
                return f"{size_match.group(1)}cm" if not size_match.group(2) else f"{size_match.group(1)}×{size_match.group(2)}cm"
            return "Specific measurement"
        else:
            return "Unknown"
    
    elif feature == "Ki67 Index":
        if any(term in value for term in ["<10", "less than 10", "low"]):
            return "<10%"
        elif any(term in value for term in ["10-30", "10 to 30", "moderate"]):
            return "10-30%"
        elif any(term in value for term in ["30-50", "30 to 50", "high"]):
            return "30-50%"
        elif any(term in value for term in [">50", "greater than 50", "very high"]):
            return ">50%"
        else:
            return "Unknown"
    
    elif feature == "PD-L1 Expression":
        if any(term in value for term in ["negative", "no expression", "absent"]):
            return "Negative"
        elif any(term in value for term in ["low expression", "weak positive", "1+"]):
            return "Low expression"
        elif any(term in value for term in ["high expression", "strong positive", "2+", "3+"]):
            return "High expression"
        else:
            return "Unknown"
    
    # For other features, use general cleaning
    return _clean_general_feature_value(feature, value)


def _clean_general_feature_value(feature: str, value: str) -> str:
    """General feature value cleaning"""
    
    value_lower = value.lower()
    
    if feature in ["Lymphovascular Invasion", "Perineural Invasion"]:
        if any(term in value_lower for term in ["present", "positive", "identified", "seen"]):
            return "Present"
        elif any(term in value_lower for term in ["absent", "negative", "not identified", "not seen"]):
            return "Absent"
        else:
            return "Unknown"
    
    elif feature == "Lymph Node Metastasis":
        if "none" in value_lower or "no" in value_lower or "negative" in value_lower:
            return "None"
        elif any(term in value_lower for term in ["1-2", "1 node", "2 nodes"]):
            return "1-2 nodes"
        elif any(term in value_lower for term in ["3-10", "3 nodes", "4 nodes", "5 nodes", "6 nodes", "7 nodes", "8 nodes", "9 nodes", "10 nodes"]):
            return "3-10 nodes"
        elif any(term in value_lower for term in [">10", "more than 10", "multiple", "extensive"]):
            return ">10 nodes"
        else:
            return "Unknown"
    
    elif feature == "Margin Status":
        if "negative" in value_lower:
            return "Negative"
        elif "positive" in value_lower:
            return "Positive"
        else:
            return "Unknown"
    
    elif feature == "Lymphocytic Infiltration":
        if "minimal" in value_lower or "rare" in value_lower:
            return "Minimal"
        elif "mild" in value_lower:
            return "Mild"
        elif "moderate" in value_lower:
            return "Moderate"
        elif "abundant" in value_lower or "dense" in value_lower:
            return "Abundant"
        else:
            return "Unknown"
    
    elif feature == "Necrosis Percentage":
        if "none" in value_lower or "absent" in value_lower:
            return "None"
        elif "<10" in value_lower or "less than 10" in value_lower or "minimal" in value_lower:
            return "<10%"
        elif "10-30" in value_lower or "10%" in value_lower:
            return "10-30%"
        elif "30-50" in value_lower or "30%" in value_lower or "moderate" in value_lower:
            return "30-50%"
        elif ">50" in value_lower or "greater than 50" in value_lower or "extensive" in value_lower or "confluent" in value_lower:
            return ">50%"
        else:
            return "Unknown"
    
    # For other features, return original value (cleaned)
    return value
