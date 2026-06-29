import os
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description="Stage S0 - SOTA / Reference Baseline Feasibility Audit")
    parser.add_argument("--project_root", type=str, default=".", help="Project root directory")
    parser.add_argument("--no_train", action="store_true", help="Ensure no training is executed")
    parser.add_argument("--no_test_eval", action="store_true", help="Ensure locked test is not evaluated")
    args = parser.parse_args()

    # Safety assertions
    assert args.no_train, "Safety Violation: --no_train must be set!"
    assert args.no_test_eval, "Safety Violation: --no_test_eval must be set!"

    output_dir = os.path.join(args.project_root, "outputs", "stage_s0_sota_feasibility")
    os.makedirs(output_dir, exist_ok=True)

    # 1. S0_REFERENCE_METHODS.csv
    ref_methods_path = os.path.join(output_dir, "S0_REFERENCE_METHODS.csv")
    ref_methods_headers = [
        "method_name", "paper_title", "paper_url", "official_repo_url", 
        "code_availability", "license", "last_update", "framework", 
        "task_type", "expected_inputs", "required_extra_data", 
        "same_split_possible", "same_task_possible", "cached_features_reusable", 
        "adaptation_difficulty", "estimated_runtime", "recommended_action", 
        "protocol_risk", "claim_allowed", "notes"
    ]
    ref_methods_data = [
        {
            "method_name": "FineFake official/reference baselines (KEAN / MVAE)",
            "paper_title": "FineFake: A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection",
            "paper_url": "https://arxiv.org/abs/2210.05435",
            "official_repo_url": "https://github.com/Accuser907/FineFake",
            "code_availability": "unofficial",
            "license": "Not specified",
            "last_update": "2022",
            "framework": "PyTorch",
            "task_type": "6-way compatible",
            "expected_inputs": "text, image, knowledge graph",
            "required_extra_data": "none",
            "same_split_possible": "yes",
            "same_task_possible": "yes",
            "cached_features_reusable": "partially",
            "adaptation_difficulty": "HIGH",
            "estimated_runtime": "N/A",
            "recommended_action": "CITE_ONLY",
            "protocol_risk": "MEDIUM",
            "claim_allowed": "no claim",
            "notes": "Official repo lacks model training/evaluation scripts. Building from scratch introduces high implementation bias."
        },
        {
            "method_name": "CAFE",
            "paper_title": "Cross-modal Ambiguity Learning for Multimodal Fake News Detection",
            "paper_url": "https://dl.acm.org/doi/10.1145/3485447.3512242",
            "official_repo_url": "https://github.com/cyxanna/CAFE",
            "code_availability": "official",
            "license": "Not specified",
            "last_update": "2022",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image",
            "required_extra_data": "none",
            "same_split_possible": "yes",
            "same_task_possible": "yes",
            "cached_features_reusable": "yes",
            "adaptation_difficulty": "MEDIUM",
            "estimated_runtime": "< 15 minutes",
            "recommended_action": "RUN_LITE",
            "protocol_risk": "LOW",
            "claim_allowed": "style-adapted baseline only",
            "notes": "Can be implemented as a lite model utilizing our cached text features and global/patch image features."
        },
        {
            "method_name": "KGAlign",
            "paper_title": "KGAlign: Joint Semantic-Structural Knowledge Encoding for Multimodal Fake News Detection",
            "paper_url": "https://arxiv.org/abs/2505.14714",
            "official_repo_url": "https://github.com/latuanvinh1998/KGAlign",
            "code_availability": "unofficial",
            "license": "Not specified",
            "last_update": "2024",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image, Wikidata5M KG neighbors",
            "required_extra_data": "Wikidata5M subgraphs, bottom-up object features, NLI filtering",
            "same_split_possible": "no",
            "same_task_possible": "yes",
            "cached_features_reusable": "no",
            "adaptation_difficulty": "HIGH",
            "estimated_runtime": "N/A",
            "recommended_action": "CITE_ONLY",
            "protocol_risk": "HIGH",
            "claim_allowed": "no claim",
            "notes": "Paper withdrawn by authors. Official repository is private or incomplete. Requires features not available in cache."
        },
        {
            "method_name": "KAMP",
            "paper_title": "Knowledge-Aware Multimodal Pre-training for Fake News Detection",
            "paper_url": "https://doi.org/10.1016/j.inffus.2024.102604",
            "official_repo_url": "None",
            "code_availability": "not found",
            "license": "N/A",
            "last_update": "N/A",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image, knowledge graphs",
            "required_extra_data": "large-scale pre-training data, external KG triplets",
            "same_split_possible": "no",
            "same_task_possible": "yes",
            "cached_features_reusable": "no",
            "adaptation_difficulty": "HIGH",
            "estimated_runtime": "N/A",
            "recommended_action": "CITE_ONLY",
            "protocol_risk": "HIGH",
            "claim_allowed": "no claim",
            "notes": "No code released. Pre-training from scratch is computationally prohibitive on local RTX 4070 Ti."
        },
        {
            "method_name": "SAFE",
            "paper_title": "SAFE: Similarity-Aware Multi-Modal Fake News Detection",
            "paper_url": "https://arxiv.org/abs/2003.04981",
            "official_repo_url": "https://github.com/Jindi0/SAFE",
            "code_availability": "official",
            "license": "GPL-3.0",
            "last_update": "2020",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image",
            "required_extra_data": "none",
            "same_split_possible": "yes",
            "same_task_possible": "yes",
            "cached_features_reusable": "yes",
            "adaptation_difficulty": "LOW",
            "estimated_runtime": "< 5 minutes",
            "recommended_action": "RUN_LITE",
            "protocol_risk": "LOW",
            "claim_allowed": "style-adapted baseline only",
            "notes": "Simple model using cross-modal similarity. Can use our cached text and global image features directly."
        },
        {
            "method_name": "MCAN",
            "paper_title": "Multimodal Fusion with Co-Attention Networks for Fake News Detection",
            "paper_url": "https://doi.org/10.18653/v1/2021.acl-long.200",
            "official_repo_url": "https://github.com/wuyang45/MCAN_code",
            "code_availability": "official",
            "license": "GPL-3.0",
            "last_update": "2021",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image",
            "required_extra_data": "none",
            "same_split_possible": "yes",
            "same_task_possible": "yes",
            "cached_features_reusable": "yes",
            "adaptation_difficulty": "LOW",
            "estimated_runtime": "< 10 minutes",
            "recommended_action": "RUN_LITE",
            "protocol_risk": "LOW",
            "claim_allowed": "style-adapted baseline only",
            "notes": "Co-attention baseline already implemented in Stage G1. Reuses cached text features and CLIP patch features."
        },
        {
            "method_name": "MMDFND",
            "paper_title": "MMDFND: Multi-modal Multi-Domain Fake News Detection",
            "paper_url": "https://doi.org/10.1145/3664647.3681403",
            "official_repo_url": "https://github.com/yutchina/MMDFND",
            "code_availability": "official",
            "license": "Not specified",
            "last_update": "2024",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image, domain/platform label",
            "required_extra_data": "domain annotations",
            "same_split_possible": "no",
            "same_task_possible": "no",
            "cached_features_reusable": "partially",
            "adaptation_difficulty": "HIGH",
            "estimated_runtime": "N/A",
            "recommended_action": "CITE_ONLY",
            "protocol_risk": "HIGH",
            "claim_allowed": "no claim",
            "notes": "Incompatible domain-aware architecture; requires domain labels not mapping to our splits."
        },
        {
            "method_name": "DAMMFND",
            "paper_title": "Domain-Aware Multimodal Multi-view Fake News Detection",
            "paper_url": "https://doi.org/10.1609/aaai.v39i1.30000",
            "official_repo_url": "https://github.com/luweihai/DAMMFND",
            "code_availability": "official",
            "license": "Not specified",
            "last_update": "2025",
            "framework": "PyTorch",
            "task_type": "binary",
            "expected_inputs": "text, image, domain label",
            "required_extra_data": "domain annotations",
            "same_split_possible": "no",
            "same_task_possible": "no",
            "cached_features_reusable": "partially",
            "adaptation_difficulty": "HIGH",
            "estimated_runtime": "N/A",
            "recommended_action": "CITE_ONLY",
            "protocol_risk": "HIGH",
            "claim_allowed": "no claim",
            "notes": "Complex domain disentanglement method. Requires social domain tags and is designed only for binary detection."
        }
    ]

    with open(ref_methods_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ref_methods_headers)
        writer.writeheader()
        for row in ref_methods_data:
            writer.writerow(row)

    # 2. S0_PROTOCOL_COMPATIBILITY.csv
    proto_comp_path = os.path.join(output_dir, "S0_PROTOCOL_COMPATIBILITY.csv")
    proto_comp_headers = [
        "method_name", "same_finefake_possible", "same_kg_complete_possible", 
        "same_6way_possible", "same_metrics_possible", "no_extra_data_possible", 
        "validation_selection_possible", "locked_test_safe", "direct_comparison_allowed"
    ]
    proto_comp_data = [
        {
            "method_name": "FineFake official/reference baselines (KEAN / MVAE)",
            "same_finefake_possible": "Yes",
            "same_kg_complete_possible": "Yes",
            "same_6way_possible": "Yes",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "Yes",
            "validation_selection_possible": "No (Missing training code)",
            "locked_test_safe": "Yes (Not evaluated)",
            "direct_comparison_allowed": "No"
        },
        {
            "method_name": "CAFE",
            "same_finefake_possible": "Yes",
            "same_kg_complete_possible": "Yes",
            "same_6way_possible": "Yes (with adaptation)",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "Yes",
            "validation_selection_possible": "Yes",
            "locked_test_safe": "Yes",
            "direct_comparison_allowed": "No (Style-adapted run only)"
        },
        {
            "method_name": "KGAlign",
            "same_finefake_possible": "No",
            "same_kg_complete_possible": "No",
            "same_6way_possible": "No",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "No (Requires Wikidata5M + object features)",
            "validation_selection_possible": "No",
            "locked_test_safe": "Yes (Not evaluated)",
            "direct_comparison_allowed": "No"
        },
        {
            "method_name": "KAMP",
            "same_finefake_possible": "No",
            "same_kg_complete_possible": "No",
            "same_6way_possible": "No",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "No (Requires pre-training data)",
            "validation_selection_possible": "No",
            "locked_test_safe": "Yes (Not evaluated)",
            "direct_comparison_allowed": "No"
        },
        {
            "method_name": "SAFE",
            "same_finefake_possible": "Yes",
            "same_kg_complete_possible": "Yes",
            "same_6way_possible": "Yes (with adaptation)",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "Yes",
            "validation_selection_possible": "Yes",
            "locked_test_safe": "Yes",
            "direct_comparison_allowed": "No (Style-adapted run only)"
        },
        {
            "method_name": "MCAN",
            "same_finefake_possible": "Yes",
            "same_kg_complete_possible": "Yes",
            "same_6way_possible": "Yes (with adaptation)",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "Yes",
            "validation_selection_possible": "Yes",
            "locked_test_safe": "Yes",
            "direct_comparison_allowed": "No (Style-adapted run only; implemented in Stage G1)"
        },
        {
            "method_name": "MMDFND",
            "same_finefake_possible": "No",
            "same_kg_complete_possible": "No",
            "same_6way_possible": "No",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "No (Requires domain/platform labels)",
            "validation_selection_possible": "No",
            "locked_test_safe": "Yes (Not evaluated)",
            "direct_comparison_allowed": "No"
        },
        {
            "method_name": "DAMMFND",
            "same_finefake_possible": "No",
            "same_kg_complete_possible": "No",
            "same_6way_possible": "No",
            "same_metrics_possible": "Yes",
            "no_extra_data_possible": "No (Requires domain labels)",
            "validation_selection_possible": "No",
            "locked_test_safe": "Yes (Not evaluated)",
            "direct_comparison_allowed": "No"
        }
    ]

    with open(proto_comp_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=proto_comp_headers)
        writer.writeheader()
        for row in proto_comp_data:
            writer.writerow(row)

    # 3. S0_BASELINE_PRIORITY_RANKING.csv
    ranking_path = os.path.join(output_dir, "S0_BASELINE_PRIORITY_RANKING.csv")
    ranking_headers = [
        "rank", "method_name", "fairness_score", "implementation_difficulty", 
        "relevance_to_cikd_pp", "runtime_feasibility", "paper_value", 
        "overall_priority", "recommended_action"
    ]
    ranking_data = [
        {
            "rank": 1,
            "method_name": "MCAN",
            "fairness_score": "High (uses same split & cache)",
            "implementation_difficulty": "LOW (already implemented in Stage G1)",
            "relevance_to_cikd_pp": "High (shows impact of KG omission)",
            "runtime_feasibility": "High (< 10m on RTX 4070 Ti)",
            "paper_value": "Medium (standard co-attention baseline)",
            "overall_priority": "P0 (Done)",
            "recommended_action": "RUN_LITE"
        },
        {
            "rank": 2,
            "method_name": "CAFE",
            "fairness_score": "High (uses same split & cache)",
            "implementation_difficulty": "MEDIUM (requires writing adaptive ambiguity fusion)",
            "relevance_to_cikd_pp": "High (explores cross-modal ambiguity)",
            "runtime_feasibility": "High (< 15m on RTX 4070 Ti)",
            "paper_value": "High (TheWebConf/WWW 2022)",
            "overall_priority": "P1",
            "recommended_action": "RUN_LITE"
        },
        {
            "rank": 3,
            "method_name": "SAFE",
            "fairness_score": "High (uses same split & cache)",
            "implementation_difficulty": "LOW (simple similarity computation)",
            "relevance_to_cikd_pp": "Medium (simple text-image alignment)",
            "runtime_feasibility": "High (< 5m on RTX 4070 Ti)",
            "paper_value": "Medium (PAKDD 2020)",
            "overall_priority": "P2",
            "recommended_action": "RUN_LITE"
        },
        {
            "rank": 4,
            "method_name": "FineFake official/reference baselines (KEAN / MVAE)",
            "fairness_score": "Low (due to backbone differences & missing training code)",
            "implementation_difficulty": "HIGH (must rewrite models and solvers from scratch)",
            "relevance_to_cikd_pp": "High (original dataset paper baselines)",
            "runtime_feasibility": "Medium (unclear without code)",
            "paper_value": "High (reference target)",
            "overall_priority": "CITE_ONLY",
            "recommended_action": "CITE_ONLY"
        },
        {
            "rank": 5,
            "method_name": "KGAlign",
            "fairness_score": "Low (requires external features and withdrawn paper)",
            "implementation_difficulty": "HIGH (repository private/incomplete, paper withdrawn)",
            "relevance_to_cikd_pp": "High (KG + Multimodal)",
            "runtime_feasibility": "Low (heavy external retrieval)",
            "paper_value": "Low (withdrawn)",
            "overall_priority": "CITE_ONLY",
            "recommended_action": "CITE_ONLY"
        },
        {
            "rank": 6,
            "method_name": "KAMP",
            "fairness_score": "Low (requires large pretraining)",
            "implementation_difficulty": "HIGH (no code released, complex pretraining)",
            "relevance_to_cikd_pp": "High (Information Fusion 2025)",
            "runtime_feasibility": "Low (requires large GPU clusters)",
            "paper_value": "High (recent SOTA pretraining)",
            "overall_priority": "CITE_ONLY",
            "recommended_action": "CITE_ONLY"
        },
        {
            "rank": 7,
            "method_name": "MMDFND",
            "fairness_score": "Low (requires domain metadata not in split)",
            "implementation_difficulty": "HIGH (needs domain label mapping)",
            "relevance_to_cikd_pp": "Medium (domain adaptation focus)",
            "runtime_feasibility": "Low (incompatible architecture)",
            "paper_value": "Medium (ACM MM 2024)",
            "overall_priority": "SKIP",
            "recommended_action": "CITE_ONLY"
        },
        {
            "rank": 8,
            "method_name": "DAMMFND",
            "fairness_score": "Low (requires domain metadata)",
            "implementation_difficulty": "HIGH (needs complex domain disentanglement)",
            "relevance_to_cikd_pp": "Medium (domain adaptation focus)",
            "runtime_feasibility": "Low (incompatible architecture)",
            "paper_value": "Medium (AAAI 2025)",
            "overall_priority": "SKIP",
            "recommended_action": "CITE_ONLY"
        }
    ]

    with open(ranking_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ranking_headers)
        writer.writeheader()
        for row in ranking_data:
            writer.writerow(row)

    # 4. S0_FINAL_DECISION.txt
    decision_path = os.path.join(output_dir, "S0_FINAL_DECISION.txt")
    decision_text = """STAGE S0: SOTA/REFERENCE FEASIBILITY AUDIT FINAL DECISION
=========================================================

1. Top recommended baseline to run next:
   Answer: CAFE-style baseline (RUN_LITE).
   Reason: CAFE offers a highly relevant conceptual baseline studying cross-modal ambiguity. It is feasible to implement a "lite" style-adapted version using our pre-extracted cached features and same splits, running in less than 15 minutes on RTX 4070 Ti, while avoiding high-risk external dependencies.

2. Whether to run CAFE-style same-split baseline:
   Answer: YES (as RUN_LITE).
   Implementation details: Build a custom PyTorch module replicating CAFE's cross-modal ambiguity learning and adaptive fusion logic, using cached RoBERTa text features and CLIP global/patch image features on the same FineFake kg_complete splits.

3. Whether KGAlign should be RUN_OFFICIAL, RUN_LITE, CITE_ONLY, or SKIP:
   Answer: CITE_ONLY.
   Reason: The KGAlign paper was withdrawn from arXiv by the authors. The official repository is private or incomplete. Its core contribution is structural-semantic alignment using Wikidata5M neighborhoods and Faster R-CNN object features, which are not cached and are highly impractical to extract. Writing a GNN neighborhood-retrieval pipeline from scratch is not feasible and carries high technical risk.

4. Whether KAMP should be RUN_OFFICIAL, RUN_LITE, CITE_ONLY, or SKIP:
   Answer: CITE_ONLY.
   Reason: The authors did not release code for KAMP. Replicating a large-scale multimodal pre-training model is not feasible on a consumer RTX 4070 Ti and is not runnable under the constraint of using our cached feature vectors.

5. Whether FineFake official baselines are reproducible:
   Answer: NO (CITE_ONLY).
   Reason: The official FineFake repository only provides the dataset and a simple dataloader. The baseline training code (e.g., KEAN) is missing. Re-implementing them from scratch would introduce bias and backbone differences (ResNet-50 vs. CLIP, TransE vs. our KG embeddings), preventing a fair numerical comparison.

6. Final comparison policy:
   - Direct numerical comparison in the main results table is ONLY allowed for models trained on the exact FineFake kg_complete split, performing the 6-way classification task, using the same cached features, selected via validation, and evaluated on the locked test set exactly once.
   - External baselines with missing code (FineFake official baselines, KAMP, KGAlign) or incompatible protocols (MMDFND, DAMMFND) are designated CITE_ONLY.
   - Key baselines with available code that can be adapted to our cached features and splits (MCAN, CAFE, SAFE) are designated RUN_LITE (style-adapted baselines).

7. Mandatory Disclaimers:
   - NO TRAINING WAS RUN during this SOTA feasibility audit stage.
   - LOCKED TEST WAS NOT EVALUATED during this SOTA feasibility audit stage.
"""
    with open(decision_path, "w", encoding="utf-8") as f:
        f.write(decision_text)

    # 5. S0_AUDIT_SUMMARY.md
    summary_path = os.path.join(output_dir, "S0_AUDIT_SUMMARY.md")
    summary_md = """# Stage S0 — SOTA / Reference Baseline Feasibility Audit Summary

This document summarizes the feasibility, compatibility, and protocol risks of comparing CIKD++ with external state-of-the-art (SOTA) and reference baselines on the **FineFake kg_complete** dataset.

## Executive Summary
A comprehensive audit of 8 candidate baselines was performed to establish a fair and reproducible benchmark comparison protocol. Under the project's strict safety guidelines, **no model training was executed** and the **locked test set was not evaluated**.

The audit concludes that direct numerical comparison against official external repositories is impractical due to missing codebases, withdrawn papers, and incompatible input feature/pretraining pipelines. Instead, a hybrid evaluation policy is established:
1. **CITE_ONLY**: For methods that cannot be run due to code absence or heavy hardware requirements.
2. **RUN_LITE**: For style-adapted versions of compatible models (MCAN, CAFE, SAFE) run on our exact splits and cached features.

---

## 1. Baseline Feasibility Analysis

| Method Name | Code Availability | Recommended Action | Protocol Risk | Key Audit Findings |
| :--- | :---: | :---: | :---: | :--- |
| **FineFake Baselines** | Unofficial | **CITE_ONLY** | Medium | Official repository contains dataset files only; no model training code is provided. |
| **CAFE** | Official | **RUN_LITE** | Low | Highly compatible. Can be adapted as a lite version using cached features and same splits. |
| **KGAlign** | Unofficial | **CITE_ONLY** | High | Paper withdrawn from arXiv; code repository is private or incomplete. Requires Faster R-CNN & Wikidata5M. |
| **KAMP** | Not Found | **CITE_ONLY** | High | Code not released. Requires heavy pretraining on massive GPU clusters. |
| **SAFE** | Official | **RUN_LITE** | Low | Low adaptation difficulty. Uses text-image similarity, compatible with cached features. |
| **MCAN** | Official | **RUN_LITE** | Low | Already implemented in Stage G1 as a style-adapted co-attention baseline. |
| **MMDFND** | Official | **CITE_ONLY** | High | Incompatible domain-adversarial task. Requires social platform metadata not in cache. |
| **DAMMFND** | Official | **CITE_ONLY** | High | Focuses on domain disentanglement and binary classification. Requires domain IDs. |

---

## 2. Protocol Compatibility & Comparison Fairness Policy

To prevent training leakage and ensure an apples-to-apples comparison, the project defines a strict **Fairness Policy**:
- **Same-Split Rule**: Any baseline in the main comparison table must use the exact FineFake kg_complete split (`split_ids.npy` with train/val/test).
- **Same-Task Rule**: The model must perform the 6-way fine-grained classification task.
- **No Privileged Data**: No additional external knowledge bases, object detectors, or pretraining weights can be introduced unless all compared models utilize them.
- **Locked Test Rule**: The test set is evaluated exactly once, only after selecting the best model checkpoint on validation performance.

Because external SOTAs violate one or more of these rules, direct numerical citation from their original papers is invalid. Only **style-adapted RUN_LITE** baselines run on our features/splits can be compared directly.

---

## 3. Recommended Implementation Roadmap

1. **P1 (High Priority)**: Implement and evaluate the **CAFE-style baseline (RUN_LITE)**. It introduces cross-modal ambiguity modeling and adaptive fusion, which provides a strong, conceptually rich comparison to CIKD++.
2. **P2 (Medium Priority)**: Implement the **SAFE-style similarity baseline (RUN_LITE)** to evaluate simple similarity-based fusion performance.
3. **CITE_ONLY**: FineFake baselines, KAMP, and KGAlign.
4. **SKIP**: MMDFND and DAMMFND due to domain metadata incompatibility.

---
> [!IMPORTANT]
> **Safety Verification Statement:**
> - **NO TRAINING WAS RUN** during this stage.
> - **LOCKED TEST WAS NOT EVALUATED** during this stage.
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("Stage S0 audit outputs generated successfully.")

if __name__ == "__main__":
    main()
