"""Streamlit demo for ClauseLens contract clause retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
EVAL_RESULTS_PATH = Path("data/processed/eval_results.json")


def api_get_json(api_base_url: str, path: str) -> dict[str, object]:
    with urlopen(f"{api_base_url.rstrip('/')}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def api_post_json(
    api_base_url: str,
    path: str,
    payload: dict[str, object],
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{api_base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def load_saved_eval(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def render_eval_panel() -> None:
    st.subheader("Evaluation")
    rows = load_saved_eval(EVAL_RESULTS_PATH)
    if not rows:
        st.caption(
            "No saved evaluation report yet. Run: "
            "`python evaluation\\eval.py --top-k 5 --output data\\processed\\eval_results.json`"
        )
        return

    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    avg_mrr = sum(float(row.get("clause_type_mrr", 0)) for row in rows) / total
    avg_ndcg = sum(float(row.get("ndcg", 0)) for row in rows) / total

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Passed", f"{passed}/{total}")
    col_b.metric("Avg MRR", f"{avg_mrr:.3f}")
    col_c.metric("Avg nDCG", f"{avg_ndcg:.3f}")


def main() -> None:
    st.set_page_config(page_title="ClauseLens", page_icon="CL", layout="wide")
    st.title("ClauseLens")
    st.caption("Contract clause evidence search over CUAD with Qdrant citations.")

    with st.sidebar:
        st.header("Search controls")
        api_base_url = st.text_input("API URL", value=API_BASE_URL)
        try:
            clause_type_response = api_get_json(api_base_url, "/clause-types")
            clause_types = [
                str(item)
                for item in clause_type_response.get("clause_types", [])
            ]
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            st.error(f"API is not reachable: {exc}")
            st.stop()

        clause_options = ["All clause types", *clause_types]
        selected_clause_type = st.selectbox("Clause type", clause_options)
        limit = st.slider("Top results", min_value=1, max_value=20, value=5)
        render_eval_panel()

    query = st.text_input(
        "Ask a contract review question",
        value="Does the contract restrict assignment?",
    )
    search_clicked = st.button("Search", type="primary")

    if not search_clicked:
        st.info("Enter a question and search to retrieve cited clause evidence.")
        return

    if not query.strip():
        st.error("Enter a non-empty search query.")
        return

    clause_type = (
        None if selected_clause_type == "All clause types" else selected_clause_type
    )

    try:
        response = api_post_json(
            api_base_url,
            "/search",
            {
                "query": query,
                "clause_type": clause_type,
                "limit": limit,
            },
        )
        results = list(response.get("results", []))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        st.error(
            "Search failed. Confirm the API is running and the starter data is "
            f"indexed. Details: {exc}"
        )
        return

    st.subheader(f"Results ({len(results)})")
    if not results:
        st.warning("No clause evidence matched this query.")
        return

    for index, item in enumerate(results, start=1):
        with st.container(border=True):
            left, right = st.columns([0.72, 0.28])
            left.markdown(
                f"**{index}. {item['clause_type'] or 'Unknown clause'}** "
                f"`score {float(item['score']):.3f}`"
            )
            right.caption(item["source_pdf"] or "Unknown source")
            if item.get("answer"):
                st.caption(f"CUAD answer label: {item['answer']}")
            st.write(item["text"])
            st.caption(f"Document: {item['document_id']} | TXT: {item['source_txt']}")


if __name__ == "__main__":
    main()
