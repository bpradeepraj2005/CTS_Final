import json
import sys
import os
# Ensure workspace root is on sys.path so `prior_auth` can be imported when
# running this script directly.
sys.path.insert(0, os.getcwd())
from prior_auth.decision_engine import Corpus, adjudicate
from prior_auth import summary

def make_request():
    return {
        "request_id": "CLM-2026-8421",
        "patient_name": "Anil Krishnan",
        "patient": {"age": 71, "sex": "M"},
        "diagnosis": {"icd10": ["I48.91"], "text": "Persistent Atrial Fibrillation"},
        "requested_service": {
            "code": "PA-INT-CARD001",
            "text": "Pre-Authorization for Direct Current (DC) Cardioversion & Pre-Cardioversion Heparinization",
            "facility_level": "tertiary",
        },
        "clinical_findings": [
            {"parameter": "heart_rate", "value": 124, "confidence": "high", "provenance": "pasted_application"},
            {"parameter": "metoprolol", "value": True, "confidence": "high", "provenance": "pasted_application"},
            {"parameter": "met_trial_duration_weeks", "value": 4, "confidence": "high", "provenance": "pasted_application"},
            {"parameter": "inr", "value": 2.5, "confidence": "high", "provenance": "lab_report"},
            {"parameter": "tte_left_atrial_thrombus_absent", "value": True, "confidence": "high", "provenance": "echocardiography"},
        ],
        "prior_therapies": [
            {"therapy": "Metoprolol Succinate", "duration_weeks": 4, "outcome": "inadequate_rate_control"},
            {"therapy": "Warfarin", "duration_weeks": 3, "outcome": "therapeutic", "inr": 2.5},
        ],
        "documentation_present": [],
        "documentation_absent": [],
        "documents": [
            {"doc_type": "12 lead ECG", "present": True, "legible_or_parseable": True, "within_validity_window": True},
            {"doc_type": "Hemogram", "present": True, "legible_or_parseable": True, "within_validity_window": True},
            {"doc_type": "Blood sugar", "present": True, "legible_or_parseable": True, "within_validity_window": True},
            {"doc_type": "Creatinine", "present": True, "legible_or_parseable": True, "within_validity_window": True},
            {"doc_type": "Electrolytes", "present": True, "legible_or_parseable": True, "within_validity_window": True},
            {"doc_type": "Prothrombin time / INR (coagulation profile)", "present": True, "legible_or_parseable": True, "within_validity_window": True},
            {"doc_type": "Echocardiography", "present": True, "legible_or_parseable": True, "within_validity_window": True},
        ],
        "clinical_notes": "71-year-old male with persistent atrial fibrillation for 7 weeks. Ventricular rate remains 124 bpm despite 4 weeks of Metoprolol Succinate 50 mg BID. Completed 3 weeks of Warfarin with therapeutic INR 2.5. TTE shows no left atrial thrombus. Provider Justification & Requested: Pre-authorization for DC cardioversion.",
        "eligibility": {"enrollment_active_on_service_date": True},
    }

def main():
    corpus = Corpus()
    req = make_request()
    packet = adjudicate(req, corpus)
    report = summary.render_json_report(packet, req)
    print(report)

if __name__ == "__main__":
    main()
