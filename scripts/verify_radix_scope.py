"""Check the dated scope record against the released radix evidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from zipfile import ZipFile

RECORD = "receipts/reproduction-scope-v1.json"
CLEAN_REPLAY_RECORD = "receipts/2026-11-27-clean-replay.json"
RELEASED_PACKAGE_SHA256 = "25612d81a480dd5f8148a0dc247db58cb2fd719fe619e75ef0cd4945b3c1364b"
ARCHIVE_EVIDENCE = {
    "protocol": "experiment/preregistration.json",
    "producer-environment": "evidence/reference/key-environment.json",
    "fresh-summary": "evidence/reference/summary.json",
    "retained-traces": "experiment/traces/manifest.json",
    "comparator": "experiment/compare_reproduction.py",
    "decision-comparison": "evidence/reference/reproduction-comparison.json",
}
RETAINED_RECEIPTS = {
    "archive-replay": "receipts/offline-replay-receipt.json",
    "archive-run": "receipts/public-replay-action.json",
    "compiler-verification": "receipts/experiment-compiler-verify-receipt.json",
}
STAGE_EVIDENCE = {
    "screening-selection": (
        "unavailable", "source-study", ("screening-aggregate-csv", "screening-product-csv"),
        ("protocol",), "frozen-selection-rule",
    ),
    "fresh-measurement": (
        "source-reported", "producer-workflow", ("protocol", "producer-environment"),
        ("fresh-summary",), "pinned-capture-and-protocol",
    ),
    "retained-trace-analysis": (
        "independently-replayed", "archive-replay-workflow", ("protocol", "retained-traces"),
        ("archive-run", "archive-replay", "compiler-verification"), "replay-retained-traces",
    ),
    "decision-agreement": (
        "matched", "producer-and-archive-workflows", ("comparator", "fresh-summary"),
        ("decision-comparison", "archive-replay"), "frozen-decision-and-margin",
    ),
    "numerical-agreement": (
        "unassessed", "producer-workflow", ("comparator", "fresh-summary"),
        ("decision-comparison",), "report-differences-without-numeric-criterion",
    ),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_clean_replay(root: Path, scope: dict, compiler: dict, replay: dict, comparison: dict) -> None:
    record = json.loads((root / CLEAN_REPLAY_RECORD).read_text(encoding="utf-8"))
    source = record["source"]
    if (
        type(record["schemaVersion"]) is not int
        or record["schemaVersion"] != 1
        or record["recordedOn"] != "2026-11-27"
        or record["experimentId"] != root.name
        or source["packageSha256"] != scope["canonicalPackageSha256"]
        or source["protocolSha256"] != scope["evidenceSha256"]["protocol"]
        or source["compilerCommit"] != scope["archiveReplayWorkflow"]["compilerCommit"]
        or source["archiveCommit"] != "9815b62ba7fddb28caf4fc1d33abe885d5b686cd"
        or source["releaseCommit"] != "a406be61e711dda31679fc716bd9cf480f531166"
    ):
        raise ValueError("clean replay does not identify the released package, protocol and pinned source")

    if record["compilerVerification"] != {
        "receipt": dict(compiler, matchedExpectedPackage=True),
        "wrongExpectedDigestRejected": True,
    }:
        raise ValueError("clean replay Compiler receipt differs from the exact package verification")

    if record["scope"] != {
        "independence": ["workflow", "fresh-runner-environment"],
        "implementation": "producer-frozen",
        "data": "producer-retained-traces",
        "historicalScreening": "unavailable",
        "freshMeasurement": "not-run",
        "numericalAgreement": "unassessed",
    }:
        raise ValueError("clean replay scope overstates independence or numerical agreement")
    environment = record["environment"]
    if (
        environment["network"] != "disabled"
        or not environment["python"].startswith("3.12.")
        or environment["numpy"] != scope["archiveReplayWorkflow"]["environment"]["numpy"]
        or re.fullmatch(r"sha256:[0-9a-f]{64}", environment["image"]) is None
        or re.fullmatch(r"[0-9a-f]{64}", environment["numpyWheelSha256"]) is None
        or record["argv"] != ["/tmp/environment/bin/python", "reproduce.py", "--output-dir", "/outputs/reproduction"]
    ):
        raise ValueError("clean replay command or environment differs from the recorded procedure")

    observed = record["replayReceipt"]
    observed_summary = observed.get("candidateSummarySha256")
    if (
        not isinstance(observed_summary, str)
        or re.fullmatch(r"[0-9a-f]{64}", observed_summary) is None
        or observed != dict(replay, candidateSummarySha256=observed_summary)
        or record["priorReplaySummaryByteMatch"] is not (observed_summary == replay["candidateSummarySha256"])
    ):
        raise ValueError("clean replay result differs from the retained trace or frozen decision")
    result = record["comparison"]
    if (
        result["status"] != "matched"
        or type(result["schemaVersion"]) is not int
        or result["schemaVersion"] != comparison["schemaVersion"]
        or result["scientificDecisionMatched"] is not True
        or result["referenceDecision"] != observed["decision"]
        or result["candidateDecision"] != observed["decision"]
        or result["decisionStabilityPolicy"] != comparison["decisionStabilityPolicy"]
        or result["failures"]
        or not isinstance(result["toleratedDifferences"], list)
        or not all(isinstance(difference, str) and difference for difference in result["toleratedDifferences"])
    ):
        raise ValueError("clean replay comparison does not establish the frozen decision")


def verify_scope(root: Path, artifact: Path, package_digest: str) -> None:
    scope = json.loads((root / RECORD).read_text(encoding="utf-8"))
    if (
        type(scope["schemaVersion"]) is not int
        or scope["schemaVersion"] != 1
        or scope["experimentId"] != root.name
        or not (scope["canonicalPackageSha256"] == package_digest == RELEASED_PACKAGE_SHA256)
    ):
        raise ValueError("reproduction scope does not identify this archive")
    if date.fromisoformat(scope["recordedOn"]) < date(2026, 10, 5):
        raise ValueError("reproduction scope predates the retained runs")

    # Only read named metadata from the already verified ZIP. Never extract or run it.
    with ZipFile(artifact) as package:
        evidence = {key: package.read(member) for key, member in ARCHIVE_EVIDENCE.items()}

    for key, filename in RETAINED_RECEIPTS.items():
        # Git checkouts may convert the recorded UTF-8 text to CRLF.
        evidence[key] = (root / filename).read_text(encoding="utf-8").encode("utf-8")
    recorded_digests = scope["evidenceSha256"]
    if recorded_digests != {key: digest(data) for key, data in evidence.items()}:
        raise ValueError("reproduction scope evidence digest mismatch")

    for key, filename in (
        ("protocol", "source/preregistration.json"),
        ("comparator", "source/compare_reproduction.py"),
        ("fresh-summary", "receipts/fresh-reproduction-summary.json"),
        ("decision-comparison", "receipts/decision-stable-comparison.json"),
    ):
        if (root / filename).read_text(encoding="utf-8").encode("utf-8") != evidence[key]:
            raise ValueError(f"archival mirror differs from the canonical ZIP: {filename}")

    protocol = json.loads(evidence["protocol"])
    sweep = protocol["source_sweep"]
    expected_inputs = {
        "screening-aggregate-csv": {
            "availability": "checksum-only", "sha256": sweep["aggregate_csv_sha256"],
        },
        "screening-product-csv": {
            "availability": "checksum-only", "sha256": sweep["product_csv_sha256"],
        },
        "offline-candidate-summary": {
            "availability": "checksum-only",
            "sha256": json.loads(evidence["archive-replay"])["candidateSummarySha256"],
        },
    }
    if scope["unavailableInputs"] != expected_inputs or any(
        re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
        for value in expected_inputs.values()
    ):
        raise ValueError("checksum-only inputs must remain explicitly unavailable")

    compiler = json.loads(evidence["compiler-verification"])
    run = json.loads(evidence["archive-run"])
    replay = json.loads(evidence["archive-replay"])
    summary = json.loads(evidence["fresh-summary"])
    comparison = json.loads(evidence["decision-comparison"])
    producer = {
        "source": compiler["source"],
        "environment": json.loads(evidence["producer-environment"]),
        "runAvailability": "restricted",
    }
    archive_run = {
        "repository": run["repository"],
        "runId": run["workflowRunId"],
        "jobId": run["jobId"],
        "workflowConclusion": "failure",
        "replayStepConclusion": "success",
        "independentOfProducer": ["workflow", "fresh-runner-environment"],
        "workflowCommit": "a15828e2f50d56c85763cefec66650076bad43c6",
        "compilerCommit": "693e4a253893158ccd635bcedea6516dd52d46e8",
        "environment": {"runner": "ubuntu-24.04", "python": "3.12", "numpy": "2.5.2"},
        "workingDirectory": "experiment",
        "argv": ["python", "reproduce.py", "--output-dir", "$RUNNER_TEMP/replay-result"],
    }
    if (
        scope["producerWorkflow"] != producer
        or scope["archiveReplayWorkflow"] != archive_run
        or run["status"] != "success-before-protected-branch-write"
        or "offline-replay" not in run["verifiedSteps"]
    ):
        raise ValueError("workflow identities, environment or replay status differ from retained evidence")

    stages = scope["stages"]
    if set(stages) != set(STAGE_EVIDENCE):
        raise ValueError("all five reproduction stages must be present")
    for stage_id, (result, performer, inputs, receipts, rule) in STAGE_EVIDENCE.items():
        stage = stages[stage_id]
        if (
            stage["result"] != result
            or stage["performedBy"] != performer
            or stage["inputs"] != list(inputs)
            or stage["receipts"] != list(receipts)
            or stage["comparisonRule"] != rule
        ):
            raise ValueError(f"unsupported result, performer or evidence for {stage_id}")

    decision = stages["decision-agreement"]
    if (
        any(type(decision[key]) is not int for key in ("validatedPairs", "requiredPairs", "frozenPairs"))
        or compiler["packageSha256"] != package_digest
        or replay["referenceSummarySha256"] != recorded_digests["fresh-summary"]
        or replay["traceManifestSha256"] != recorded_digests["retained-traces"]
        or replay["status"] != "matched"
        or comparison["status"] != "matched"
        or comparison["scientificDecisionMatched"] is not True
        or comparison["failures"]
        or not (
            decision["decision"] == replay["decision"] == summary["decision"]
            == comparison["candidateDecision"] == comparison["referenceDecision"]
        )
        or decision["validatedPairs"] != replay["validatedPairCount"]
        or decision["validatedPairs"] != summary["validated_pair_count"]
        or decision["requiredPairs"] != replay["requiredPairCount"]
        or decision["requiredPairs"] != summary["required_pair_count"]
        or decision["frozenPairs"] != len(summary["pair_decisions"])
        or decision["frozenPairs"] != len(protocol["frozen_pairs"])
    ):
        raise ValueError("retained analysis or decision does not match the replay and producer records")

    numeric = stages["numerical-agreement"]
    if not comparison["toleratedDifferences"] or (
        numeric["reportedDifferences"] != comparison["toleratedDifferences"]
    ):
        raise ValueError("numerical differences must remain visible independently of the decision")

    verify_clean_replay(root, scope, compiler, replay, comparison)
