"""Inspect the Fashion RAG pipeline one stage at a time.

Examples:
    python scripts/debug_rag_pipeline.py --query "tìm áo sơ mi trắng đi làm"
    python scripts/debug_rag_pipeline.py --query "phối đồ đi tiệc cho dáng quả lê" --profile-json "{\"dang_nguoi\":\"Dáng quả lê\"}"
    python scripts/debug_rag_pipeline.py --cases tests/retrieval_debug_cases.example.jsonl --json-out eval_outputs/debug_run.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PRODUCT_SEARCH_CANDIDATE_K = 30
DEFAULT_PRODUCT_SEARCH_PAGE_SIZE = 5
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VIFASHIONCLIP_SERVICE_URL = "http://localhost:18080"
DEFAULT_PRODUCT_COLLECTION = "fashion_products_vifashionclip_vi_65k_structured_vi"


def check_services(timeout: float) -> dict:
    import requests
    from qdrant_client import QdrantClient

    timeout = min(timeout, 5.0)
    report = {}

    print("\n=== Service check ===")
    try:
        qdrant = QdrantClient(url=DEFAULT_QDRANT_URL, timeout=timeout, check_compatibility=False)
        exists = qdrant.collection_exists(DEFAULT_PRODUCT_COLLECTION)
        report["qdrant"] = {"ok": True, "product_collection_exists": bool(exists)}
        print(f"Qdrant  : OK | {DEFAULT_PRODUCT_COLLECTION} exists={exists}")
    except Exception as exc:
        report["qdrant"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"Qdrant  : FAIL | {type(exc).__name__}: {exc}")

    try:
        response = requests.get(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        report["ollama"] = {"ok": True, "models": models[:12]}
        print(f"Ollama  : OK | models={models[:5]}")
    except Exception as exc:
        report["ollama"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"Ollama  : FAIL | {type(exc).__name__}: {exc}")

    try:
        response = requests.get(f"{DEFAULT_VIFASHIONCLIP_SERVICE_URL}/health", timeout=timeout)
        response.raise_for_status()
        report["vifashionclip_service"] = {"ok": True, "health": response.json()}
        print(f"Embedder: OK | {response.json()}")
    except Exception as exc:
        report["vifashionclip_service"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"Embedder: FAIL | {type(exc).__name__}: {exc}")

    return report


def compact(value: Any, width: int = 80) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def score_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def doc_row(doc, rank: int | None = None) -> dict:
    from app.core.vector_store import normalize_product_metadata

    doc = normalize_product_metadata(doc)
    metadata = doc.metadata
    return {
        "rank": rank,
        "product_id": metadata.get("product_id", ""),
        "title": metadata.get("title", ""),
        "category": metadata.get("category", ""),
        "brand": metadata.get("brand", ""),
        "price": metadata.get("price", ""),
        "dense_score": metadata.get("dense_score"),
        "rerank_score": metadata.get("rerank_score"),
        "rerank_rank": metadata.get("rerank_rank"),
        "image_score": metadata.get("image_search_score"),
        "preview": doc.page_content[:240],
    }


def print_rows(title: str, rows: list[dict], limit: int | None = None) -> None:
    rows = rows[:limit] if limit else rows
    print(f"\n=== {title} ({len(rows)}) ===")
    if not rows:
        print("(empty)")
        return
    for row in rows:
        rank = row.get("rank") or row.get("rerank_rank") or "-"
        parts = [
            f"#{rank}",
            f"id={compact(row.get('product_id'), 24)}",
            f"score={score_text(row.get('dense_score'))}",
        ]
        if row.get("rerank_score") is not None:
            parts.append(f"rerank={score_text(row.get('rerank_score'))}")
        if row.get("image_score") is not None:
            parts.append(f"image={score_text(row.get('image_score'))}")
        print(" | ".join(parts))
        print(f"  title   : {compact(row.get('title'), 120)}")
        print(f"  category: {compact(row.get('category'), 60)} | brand: {compact(row.get('brand'), 40)} | price: {row.get('price')}")
        print(f"  preview : {compact(row.get('preview'), 180)}")


def parse_profile(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --profile-json: {exc}") from exc


def decide_route(query: str, router_mode: str, state: dict | None = None) -> RouteDecision:
    from app.core.intent import ROUTE_PRODUCT_SEARCH, RouteDecision, route_from_keywords, route_user_request

    if router_mode == "keyword":
        decision = route_from_keywords(query, state=state)
        if decision:
            return decision
        return RouteDecision(
            route=ROUTE_PRODUCT_SEARCH,
            action="search",
            confidence=0.0,
            rewrite_query=query,
            reason="No keyword route matched; debug defaulted to product_search.",
            source="debug_keyword_default",
        )
    return route_user_request(query, state=state)


def debug_product_retrieval(query: str, k: int, page_size: int, threshold: float | None) -> dict:
    print("\n[Stage] Layer A product retrieval")
    print("Loading retrieval modules...")
    from app.config import ENABLE_PRODUCT_RERANKER, RERANKER_TOP_N
    from app.core.vector_store import diversity_filter_documents, get_product_vector_db, normalize_product_metadata, rerank_documents

    print(f"query={query!r} | k={k} | threshold={threshold} | reranker_env={ENABLE_PRODUCT_RERANKER}")

    vector_db = get_product_vector_db()
    pairs = vector_db.similarity_search_with_score(query=query, k=k)
    raw_docs = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        doc.metadata["dense_rank"] = rank
        doc.metadata["dense_score"] = float(score)
        raw_docs.append(normalize_product_metadata(doc))

    raw_rows = [doc_row(doc, i) for i, doc in enumerate(raw_docs, start=1)]
    print_rows("Qdrant raw candidates", raw_rows, limit=min(k, 30))

    candidate_docs = raw_docs
    if threshold is not None:
        candidate_docs = [doc for doc in raw_docs if float(doc.metadata.get("dense_score", 0.0)) >= threshold]
        print_rows("After dense threshold", [doc_row(doc, i) for i, doc in enumerate(candidate_docs, start=1)])

    ranked_docs = rerank_documents(query, candidate_docs, top_n=RERANKER_TOP_N)
    print_rows("After optional rerank", [doc_row(doc, i) for i, doc in enumerate(ranked_docs, start=1)])

    final_docs = diversity_filter_documents(ranked_docs, max_docs=page_size)
    final_rows = [doc_row(doc, i) for i, doc in enumerate(final_docs, start=1)]
    print_rows("Final docs sent to LLM", final_rows)

    return {
        "raw": raw_rows,
        "after_threshold_count": len(candidate_docs),
        "after_rerank": [doc_row(doc, i) for i, doc in enumerate(ranked_docs, start=1)],
        "final": final_rows,
    }


def _layer_b_query(collection: str, query_vector: list[float], search_filter, stage: str) -> tuple[str, Any | None]:
    from app.config import LAYER_B_SCORE_THRESHOLD
    from app.core.vector_store import client

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=search_filter,
        limit=1,
        score_threshold=LAYER_B_SCORE_THRESHOLD,
    )
    return stage, response.points[0] if response.points else None


def debug_layer_b_rule(query: str, gender: str, profile: dict) -> dict:
    print("\n[Stage] Layer B outfit rule retrieval")
    print("Loading Layer B retrieval modules...")
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny

    from app.config import LAYER_B_WILDCARD_DANG, LAYER_B_WILDCARD_TONE
    from app.core.vector_store import get_rule_embeddings

    collection = f"layer_b_{gender}"
    print(f"query={query!r} | collection={collection} | profile={profile}")
    query_vector = get_rule_embeddings().embed_query(query)

    conditions = []
    if profile.get("dang_nguoi"):
        conditions.append(
            FieldCondition(
                key="dang_nguoi",
                match=MatchAny(any=[profile["dang_nguoi"], LAYER_B_WILDCARD_DANG]),
            )
        )
    if profile.get("tone_da"):
        conditions.append(
            FieldCondition(
                key="tone_da",
                match=MatchAny(any=[profile["tone_da"], LAYER_B_WILDCARD_TONE]),
            )
        )

    attempts = []
    search_filter = Filter(must=conditions) if conditions else None
    attempts.append(_layer_b_query(collection, query_vector, search_filter, "profile_filter"))

    if not attempts[-1][1] and profile.get("tone_da") and profile.get("dang_nguoi"):
        fallback_filter = Filter(
            must=[
                FieldCondition(
                    key="dang_nguoi",
                    match=MatchAny(any=[profile["dang_nguoi"], LAYER_B_WILDCARD_DANG]),
                )
            ]
        )
        attempts.append(_layer_b_query(collection, query_vector, fallback_filter, "drop_tone_filter"))

    if not attempts[-1][1] and search_filter:
        attempts.append(_layer_b_query(collection, query_vector, None, "no_filter"))

    chosen = None
    for stage, point in attempts:
        if point:
            chosen = point
            payload = point.payload
            print(f"{stage}: HIT score={score_text(point.score)}")
            print(f"  rule_key : {payload.get('rule_key')}")
            print(f"  style    : {payload.get('phong_cach')} | context: {payload.get('boi_canh')}")
            print(f"  body/tone: {payload.get('dang_nguoi')} | {payload.get('tone_da')}")
            print(f"  reason   : {compact(payload.get('ly_do_tu_van'), 160)}")
        else:
            print(f"{stage}: MISS")

    if not chosen:
        return {"attempts": [{"stage": stage, "hit": False} for stage, point in attempts], "chosen": None}

    return {
        "attempts": [{"stage": stage, "hit": bool(point), "score": getattr(point, "score", None)} for stage, point in attempts],
        "chosen": chosen.payload,
        "score": chosen.score,
    }


def debug_outfit_products(base_rule: dict, gender: str) -> dict:
    print("\n[Stage] Layer B details -> Layer A products")
    print("Loading outfit product retrieval modules...")
    from qdrant_client.http.models import FieldCondition, Filter, MatchAny

    from app.core.outfit import find_outfit_details, get_layer_a_categories
    from app.core.vector_store import diversity_filter_documents, get_product_vector_db, normalize_product_metadata

    details = find_outfit_details(base_rule, gender)
    if not details:
        print("No outfit detail rules found.")
        return {"details": {}, "products": {}}

    vector_db = get_product_vector_db()
    output = {}
    for layer_b_category, rule in details.items():
        product_type = rule["rule_key"].split("|")[1].strip()
        target_categories = get_layer_a_categories(layer_b_category, product_type)
        search_filter = None
        if target_categories:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.category",
                        match=MatchAny(any=target_categories),
                    )
                ]
            )

        product_query = f"{product_type} {base_rule['phong_cach']}"
        print(f"\nItem: {layer_b_category} | type={product_type} | mapped={target_categories}")
        pairs = vector_db.similarity_search_with_score(query=product_query, k=8, filter=search_filter)
        docs = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            doc.metadata["dense_rank"] = rank
            doc.metadata["dense_score"] = float(score)
            docs.append(normalize_product_metadata(doc))
        rows = [doc_row(doc, i) for i, doc in enumerate(docs, start=1)]
        print_rows("Layer A candidates for outfit item", rows)
        accepted = [doc for doc in docs if float(doc.metadata.get("dense_score", 0.0)) >= 0.30]
        final_docs = diversity_filter_documents(accepted, max_docs=3)
        final_rows = [doc_row(doc, i) for i, doc in enumerate(final_docs, start=1)]
        print_rows("Accepted outfit products", final_rows)
        output[layer_b_category] = {
            "product_type": product_type,
            "mapped_categories": target_categories,
            "raw": rows,
            "final": final_rows,
        }

    return {"details": details, "products": output}


def run_one_case(case: dict, args) -> dict:
    from app.core.intent import detect_gender
    from app.core.security import validate_user_query

    query = case["query"]
    profile = parse_profile(args.profile_json)
    profile.update(case.get("profile", {}))
    state = {"profile": profile}

    print("\n" + "=" * 88)
    print(f"QUERY: {query}")
    ok, validation_msg = validate_user_query(query)
    print(f"\n[Stage] Input validation: {'OK' if ok else 'BLOCKED'}")
    if not ok:
        print(validation_msg)
        return {"query": query, "validation_ok": False, "validation_message": validation_msg}

    decision = decide_route(query, args.router_mode, state=state)
    gender = case.get("gender") or profile.get("gender") or detect_gender(decision.rewrite_query or query)
    if gender == "unknown":
        gender = args.gender

    print("\n[Stage] Router")
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))
    print(f"gender={gender}")

    active_query = decision.rewrite_query or query
    result = {
        "query": query,
        "active_query": active_query,
        "route": asdict(decision),
        "gender": gender,
        "validation_ok": True,
    }

    if args.route_only:
        return result

    try:
        if decision.intent == "outfit" or args.force_outfit:
            rule_result = debug_layer_b_rule(active_query, gender, profile)
            result["layer_b_rule"] = rule_result
            if rule_result.get("chosen"):
                result["outfit_products"] = debug_outfit_products(rule_result["chosen"], gender)
        else:
            result["product_retrieval"] = debug_product_retrieval(
                active_query,
                k=args.k,
                page_size=args.page_size,
                threshold=args.threshold,
            )
    except Exception as exc:
        print(f"\n[ERROR] Stage failed: {type(exc).__name__}: {exc}")
        print("Check whether Qdrant, Ollama, and the ViFashionCLIP embedding service are running.")
        result["stage_error"] = {"type": type(exc).__name__, "message": str(exc)}

    expected_ids = {str(x).lower() for x in case.get("expected_product_ids", [])}
    if expected_ids and "product_retrieval" in result:
        final_ids = {
            str(row.get("product_id", "")).lower()
            for row in result["product_retrieval"]["final"]
            if row.get("product_id")
        }
        result["expected_hit"] = bool(expected_ids & final_ids)
        print(f"\n[Eval] expected_product_ids hit={result['expected_hit']} expected={sorted(expected_ids)} final={sorted(final_ids)}")

    return result


def load_cases(args) -> list[dict]:
    if args.query:
        return [{"query": args.query}]
    cases = []
    with Path(args.cases).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {args.cases}:{line_no}: {exc}") from exc
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Fashion RAG routing and retrieval stages.")
    parser.add_argument("--query", help="Single query to debug.")
    parser.add_argument("--cases", help="JSONL cases file. Each line needs at least {'query': '...'}")
    parser.add_argument("--json-out", help="Optional JSONL output path for machine-readable debug results.")
    parser.add_argument("--router-mode", choices=["full", "keyword"], default="full")
    parser.add_argument("--gender", choices=["female", "male"], default="female")
    parser.add_argument("--profile-json", help="Optional user profile JSON, e.g. '{\"dang_nguoi\":\"Dáng quả lê\"}'")
    parser.add_argument("--k", type=int, default=DEFAULT_PRODUCT_SEARCH_CANDIDATE_K)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PRODUCT_SEARCH_PAGE_SIZE)
    parser.add_argument("--threshold", type=float, default=None, help="Optional dense score cutoff for product search.")
    parser.add_argument("--force-outfit", action="store_true", help="Run outfit debug even if router does not choose outfit.")
    parser.add_argument("--route-only", action="store_true", help="Stop after validation/router; do not call retrieval services.")
    parser.add_argument("--check-services", action="store_true", help="Only check Qdrant, Ollama, and embedding service readiness.")
    parser.add_argument("--embedding-timeout", type=float, default=20.0, help="Remote ViFashionCLIP request timeout in seconds.")
    args = parser.parse_args()

    if args.check_services:
        report = check_services(args.embedding_timeout)
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[OK] Wrote service report: {out_path}")
        return

    if not args.query and not args.cases:
        parser.error("Provide --query or --cases")

    os.environ["VIFASHIONCLIP_SERVICE_TIMEOUT"] = str(args.embedding_timeout)

    results = [run_one_case(case, args) for case in load_cases(args)]

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"\n[OK] Wrote debug JSONL: {out_path}")


if __name__ == "__main__":
    main()
