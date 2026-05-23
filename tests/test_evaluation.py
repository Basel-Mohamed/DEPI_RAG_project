import csv
import json

from scripts.evaluate_rag import evaluate_items, load_dataset, write_reports


class FakePipeline:
    def run(self, question: str):
        return {
            "answer": f"Answer about {question}",
            "sources": [{"id": "policy.pdf", "title": "policy.pdf", "metadata": {"source": "policy.pdf"}}],
            "retrieval": {"documents": 1, "mode": "hybrid"},
        }


def test_load_dataset_supports_json_and_csv(tmp_path):
    json_path = tmp_path / "items.json"
    csv_path = tmp_path / "items.csv"
    json_path.write_text(json.dumps([{"question": "Q1"}]), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["question"])
        writer.writeheader()
        writer.writerow({"question": "Q2"})

    assert load_dataset(json_path)[0]["question"] == "Q1"
    assert load_dataset(csv_path)[0]["question"] == "Q2"


def test_evaluation_report_generation(tmp_path):
    report = evaluate_items(
        [
            {
                "question": "refunds",
                "expected_answer": "Answer about refunds",
                "expected_source": "policy.pdf",
            }
        ],
        FakePipeline(),
    )

    write_reports(report, tmp_path)

    assert report["summary"]["count"] == 1
    assert report["summary"]["source_hit_rate"] == 1
    assert (tmp_path / "evaluation_results.json").exists()
    assert (tmp_path / "evaluation_report.md").exists()
