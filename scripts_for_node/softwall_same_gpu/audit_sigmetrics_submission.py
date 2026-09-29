#!/usr/bin/env python3
"""Audit the anonymous SIGMETRICS SoftWall submission bundle.

This audit is intentionally claim-scoped.  It checks that the LaTeX paper
faithfully carries the already-audited evidence and submission requirements;
it does not promote an unqualified lifecycle mode or create a new experiment
gate.
"""

from __future__ import print_function

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "softwall_sigmetrics27"
TEX = PAPER / "main.tex"
BIB = PAPER / "references.bib"
BBL = PAPER / "main.bbl"
LOG = PAPER / "main.log"
PDF = PAPER / "main.pdf"
UPSTREAM = ROOT / "results" / "softwall_multigpu" / "softwall_manuscript_claim_audit_v1.json"
LIFECYCLE = ROOT / "results" / "softwall_multigpu" / "c164_lifecycle_qualification_summary_v1.json"
NECESSITY = ROOT / "results" / "softwall_multigpu" / "softwall_necessity_witness_v3.json"
CFP = ROOT / "results" / "softwall_multigpu" / "sigmetrics27_cfp_recheck_v1.json"
PRODUCTION = ROOT / "results" / "softwall_multigpu" / "softwall_production_exit_gate_v3.json"
CAPACITY = ROOT / "results" / "softwall_multigpu" / "c167_reclaimable_capacity_model_v2.json"
HEADROOM = ROOT / "results" / "softwall_multigpu" / "c174_trace_oracle_screen_v1.json"
LEVERS = ROOT / "results" / "softwall_multigpu" / "c175_lever_summary_v1.json"
OUT = ROOT / "results" / "softwall_multigpu" / "softwall_sigmetrics_submission_audit_v1.json"
PROVENANCE_MARKERS = (
    "Q1_MPS_DIAGNOSTICS",
    "Q2_WARM_PATH",
    "C160_FAULT",
    "C162_ENVELOPE",
    "C162_SCHEDULER",
    "C162_NECESSITY",
    "C159_BASELINE",
    "C167_CAPACITY",
    "C174_HEADROOM",
    "C175_LEVERS",
    "C164_LIFECYCLE",
    "P2_FAST_PATH_AND_STAGE",
    "P3_CHANNEL_HOLDOUT",
    "C158_LONG_TAIL",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def page_count(log):
    matches = re.findall(r"Output written on main\.pdf \((\d+) pages?", log)
    return int(matches[-1]) if matches else None


def extract_pdf_text():
    try:
        return subprocess.check_output(
            ["pdftotext", str(PDF), "-"], stderr=subprocess.STDOUT
        ).decode("utf-8", "replace")
    except Exception:
        return ""


def main():
    required = [TEX, BIB, BBL, LOG, PDF, UPSTREAM, LIFECYCLE, NECESSITY, CFP,
                PRODUCTION, CAPACITY, HEADROOM, LEVERS]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("missing required artifacts: " + ", ".join(missing))

    tex = read(TEX)
    normalized_tex = " ".join(tex.split())
    bib = read(BIB)
    log = read(LOG)
    pdf_text = extract_pdf_text()
    upstream = json.loads(read(UPSTREAM))
    lifecycle = json.loads(read(LIFECYCLE))
    necessity = json.loads(read(NECESSITY))
    cfp = json.loads(read(CFP))
    production = json.loads(read(PRODUCTION))
    capacity = json.loads(read(CAPACITY))
    headroom = json.loads(read(HEADROOM))
    levers = json.loads(read(LEVERS))

    cite_groups = re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex)
    cited = sorted({k.strip() for group in cite_groups for k in group.split(",")})
    bib_keys = set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)", bib))
    unresolved_keys = sorted(set(cited) - bib_keys)
    pages = page_count(log)

    identity_patterns = [
        r"/pscratch/",
        r"\bsgkim\b",
        r"\bairan_cloudlab\b",
        r"\bnid\d{6}\b",
        r"\bjob\s*#?\s*\d{6,}\b",
        r"\bj\d{8,}\b",
    ]
    identity_hits = []
    for pattern in identity_patterns:
        if re.search(pattern, tex, re.IGNORECASE) or re.search(pattern, pdf_text, re.IGNORECASE):
            identity_hits.append(pattern)

    figures = [
        ROOT / "docs" / "current" / "figures" / "softwall_system_contract.pdf",
        ROOT / "docs" / "current" / "figures" / "softwall_c162_envelope_scalability.pdf",
        PAPER / "figures" / "softwall_capacity_headroom.pdf",
    ]

    core_numbers = {
        "q2_nrx": "4,800",
        "q2_recovery": "1,312",
        "q2_qwen": "1,077",
        "fault_nrx": "2,800",
        "fault_recovery": "942",
        "model_states": "16,023",
        "boundary_rounds": "180",
        "decision_p99": "1.494",
        "oracle_tokens": "349,387",
        "capacity_points": "21,384",
        "capacity_static_gain": "3,600",
        "headroom_points": "362",
        "greedy_closed_points": "361",
        "placement_positive_points": "203",
    }

    checks = {
        "required_artifacts_present": not missing,
        "acmart_review_anonymous": r"\documentclass[acmsmall,screen,review,anonymous]{acmart}" in tex,
        "no_author_block": not re.search(r"\\author\s*\{", tex),
        "no_identity_or_local_path_leak": not identity_hits,
        "title_and_abstract_present": (re.search(r"\\title(\[[^\]]*\])?\{Backstop:", tex) is not None
                                       and r"\begin{abstract}" in tex),
        "conference_metadata_present": "SIGMETRICS 2027" in tex and "June 7--11, 2027" in tex,
        "official_cfp_rechecked": (
            cfp.get("all_pass") is True
            and cfp["winter_deadline"]["abstract_registration"] == "January 4, 2027 23:59 AoE"
            and cfp["winter_deadline"]["paper_submission"] == "January 11, 2027 23:59 AoE"
            and cfp["winter_deadline"]["notification"] == "March 10, 2027"
            and cfp["format"]["technical_content_pages_max"] == 20
            and cfp["format"]["review"] == "double anonymous"
        ),
        "ccs_and_keywords_present": r"\begin{CCSXML}" in tex and r"\keywords{" in tex,
        "formal_contract_present": (
            "All-fail dominance" in tex
            and "Certificate-to-execution refinement" in tex
            and r"T_{\mathrm{dec},h}+b_{\mathrm{eff}}" in tex
        ),
        "four_way_envelope_present": all(x in tex for x in [r"\QSU", r"\QSN", r"\MI", r"\UQ"]),
        "core_numbers_match_audited_claims": all(v in tex for v in core_numbers.values()),
        "negative_optimizer_result_preserved": (
            "no executed optimizer claim" in normalized_tex and "1.14\\%" in tex
        ),
        "capacity_headroom_and_levers_preserved": (
            capacity.get("all_pass") is True
            and capacity["summary"]["grid_points"] == 21384
            and capacity["summary"]["softwall_better_than_static_points"] == 3600
            and capacity["summary"]["debt_blind_positive_violation_probability_points"] == 4338
            and capacity["exact_checker"]["mismatches"] == 0
            and headroom.get("all_pass") is True
            and headroom["summary"]["gap_at_least_mde_and_positive_all_windows_points"] == 362
            and levers.get("all_pass") is True
            and levers["greedy_within_5pct_of_fixed_oracle_points"] == 361
            and levers["flexible_recovery_positive_gain_points"] == 203
            and levers["by_gpu"]["1"] == {
                "candidate_points": 102,
                "flexible_recovery_positive_gain_points": 0,
            }
            and levers["sionna_recovery_debt_reduction"] == [0.0, 0.0]
        ),
        "necessity_witness_preserved": (
            necessity.get("all_pass") is True
            and necessity["summary"]["physical_debt_blind_E4_selected_deadline_violations"] == 40
            and necessity["summary"]["physical_E6b_shadow_selected_guard_violations"] == 40
            and necessity["summary"]["physical_softwall_selected_guard_violations"] == 0
            and necessity["summary"]["minimum_contract_excess_ms"] == 1
            and necessity["summary"]["maximum_contract_excess_ms"] == 12
            and necessity["contract_sensitivity"]["thresholds"]["19_ms_lt_B_conv_le_24_ms"]
                == "E6b disappears but E4 remains false-safe"
            and "What the certificate prevents" in tex
            and "certificate-preserving recovery-first" in tex
            and "bound-padded diagnostic" in tex
            and "40/40" in tex
            and "E6b disappears at or below 24" in tex
            and "E4 remains until 19" in tex
        ),
        "lifecycle_scope_preserved": (
            re.search(r"five qualified or partial subsets and five\s+\\UQ", tex) is not None
            and re.search(r"does not require or imply a 10/10\s+lifecycle", tex) is not None
            and lifecycle.get("claim_scope_complete") is True
            and lifecycle.get("counts", {}).get("qualified_or_partial_modes") == 5
            and lifecycle.get("counts", {}).get("unqualified_modes") == 5
        ),
        "process_replacement_not_promoted": (
            "Development canary only" in tex
            and "Durable journal recovery would be a new mechanism" in tex
        ),
        "production_and_wcet_limits_present": (
            "make no production" in normalized_tex
            and "mechanism evidence rather than qualification or WCET" in normalized_tex
            and "WCET" in tex
            and production.get("all_pass") is False
            and production.get("status") == "FAIL_CURRENT_DESIGN_NOT_PRODUCTION_QUALIFIED"
            and production["current_timing_diagnosis"]["gate_pass"] is False
            and production["external_channel_diagnosis"]["gate_pass"] is True
            and production["external_channel_diagnosis"]["aerial_tdl_a"]["qualified"] is False
            and production["current_timing_diagnosis"]["cross_gpu_candidates"]
                ["raw_iq_full_remote_persistent_ordered"]["deadline_counts"]
                ["parallel_pair_wall_le_deadline"] == 995
            and production["current_timing_diagnosis"]["cross_gpu_candidates"]
                ["raw_iq_full_remote_persistent_ordered"]["holdout_opened"] is False
            and "995/1,000" in tex
            and "External Aerial TDL-A" in tex
            and "31 neural-only versus 12 conventional-only" in tex
            and "optional \\NRx has conditional radio value" in tex
        ),
        "quantitative_provenance_preserved": (
            upstream.get("checks", {}).get("quantitative_provenance_complete") is True
            and len(upstream.get("evidence", {}).get("quantitative_provenance", [])) >= 70
            and all(f"% provenance: {marker}" in tex for marker in PROVENANCE_MARKERS)
            and "350.948\\,ms" in tex
            and "8.111\\,ms" in tex
            and "4/400 deadline misses" in tex
        ),
        "reproducibility_statement_present": r"\paragraph{Reproducibility.}" in tex,
        "generative_ai_disclosure_present": r"\paragraph{Use of generative AI.}" in tex,
        "all_citations_resolve": not unresolved_keys and "Citation `" not in log,
        "all_references_resolve": "There were undefined references" not in log,
        "no_overfull_boxes": "Overfull \\hbox" not in log and "Overfull \\vbox" not in log,
        "figures_present": all(p.is_file() and p.stat().st_size > 0 for p in figures),
        "pdf_compiled": PDF.stat().st_size > 0,
        "page_count_within_20": pages is not None and pages <= 20,
        "upstream_claim_audit_pass": (
            upstream.get("all_pass") is True
            and upstream.get("status") == "SOFTWALL_MANUSCRIPT_CLAIMS_AUDITED_PASS"
        ),
    }

    all_pass = all(checks.values())
    artifacts = [TEX, BIB, BBL, PDF, UPSTREAM, LIFECYCLE, NECESSITY, CFP, PRODUCTION,
                 CAPACITY, HEADROOM, LEVERS] + figures
    out = {
        "schema": "softwall-sigmetrics-submission-audit-v1",
        "status": "SOFTWALL_SIGMETRICS_SUBMISSION_AUDIT_PASS" if all_pass else "SOFTWALL_SIGMETRICS_SUBMISSION_AUDIT_FAIL",
        "all_pass": all_pass,
        "checks": checks,
        "evidence": {
            "pages_including_references": pages,
            "cited_entries": len(cited),
            "unresolved_citation_keys": unresolved_keys,
            "identity_hits": identity_hits,
            "lifecycle_qualified_or_partial": lifecycle["counts"]["qualified_or_partial_modes"],
            "lifecycle_unqualified": lifecycle["counts"]["unqualified_modes"],
            "necessity": necessity["summary"],
            "production_exit": {
                "status": production["status"],
                "timing_gate_pass": production["current_timing_diagnosis"]["gate_pass"],
                "channel_gate_pass": production["external_channel_diagnosis"]["gate_pass"],
                "best_raw_iq_development_timely": production["current_timing_diagnosis"]
                    ["cross_gpu_candidates"]["raw_iq_full_remote_persistent_ordered"]
                    ["deadline_counts"]["parallel_pair_wall_le_deadline"],
            },
            "core_numbers_required": core_numbers,
            "capacity": capacity["summary"],
            "headroom": headroom["summary"],
            "levers": levers,
        },
        "artifact_sha256": {
            str(p.relative_to(ROOT)): sha256(p) for p in artifacts
        },
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
