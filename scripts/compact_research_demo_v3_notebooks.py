"""Compact Research Demo v3 notebooks into readable business-level sections."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "research_demo_v3_split"
ARCHIVE_DIR = NOTEBOOK_DIR / "archive"
LEGACY_NOTEBOOK = "05_layer_b_outfit_and_category_mapping.ipynb"
ARCHIVED_LEGACY_NOTEBOOK = "05_layer_b_outfit_and_category_mapping_legacy.ipynb"
GENERATED_CONTRACT_RE = re.compile(
    r"\n*### (?:Định nghĩa thành phần|Thực thi và kiểm tra):[^\n]*\n\n"
    r"- \*\*Tác dụng chính:\*\*[^\n]*\n"
    r"- \*\*Đầu vào \(Input\):\*\*[^\n]*\n"
    r"- \*\*Đầu ra \(Output\):\*\*[^\n]*\s*$",
    flags=re.MULTILINE,
)
NOTEBOOK_PURPOSES = {
    "01_environment_config_models.ipynb": "chuẩn bị môi trường, endpoint và các mô hình embedding nền",
    "02_product_data_pipeline.ipynb": "chuẩn hóa dữ liệu sản phẩm và index vào Qdrant",
    "03_layer_a_text_retrieval_debug.ipynb": "kiểm tra retrieval văn bản của Layer A",
    "04_image_retrieval_debug.ipynb": "kiểm tra retrieval ảnh và truy vấn ảnh kết hợp văn bản",
    "05_layer_b_outfit_and_category_mapping.ipynb": "đối chiếu luồng Layer B legacy",
    "05_layer_b_outfit_and_category_mapping_v2.ipynb": "kiểm tra rule, constraint và phối đồ Layer B V2",
    "06_vision_module_debug.ipynb": "quan sát ảnh và tạo profile candidate",
    "07_llm_prompts_chains_history.ipynb": "lắp prompt, chain và lịch sử hội thoại",
    "08_router_security_logging_eval.ipynb": "kiểm tra router, guardrail và evaluation",
    "09_chat_loop_demo.ipynb": "chạy vòng lặp chat theo state machine",
}
NOTEBOOK_TITLES = {
    "01_environment_config_models.ipynb": "01 - Môi Trường, Cấu Hình Và Mô Hình Nền",
    "02_product_data_pipeline.ipynb": "02 - Product Data Pipeline",
    "03_layer_a_text_retrieval_debug.ipynb": "03 - Layer A Text Retrieval Debug",
    "04_image_retrieval_debug.ipynb": "04 - Image Retrieval Debug",
    "05_layer_b_outfit_and_category_mapping.ipynb": "05 - Layer B Outfit Rule Debug (Legacy)",
    "05_layer_b_outfit_and_category_mapping_v2.ipynb": "05 V2 - Layer B Có Ràng Buộc Và Rerank",
    "06_vision_module_debug.ipynb": "06 - Vision Module Debug",
    "07_llm_prompts_chains_history.ipynb": "07 - LLM, Prompt, Chain Và History",
    "08_router_security_logging_eval.ipynb": "08 - Router, Security, Logging Và Eval",
    "09_chat_loop_demo.ipynb": "09 - Chat Loop State Machine",
}


def source_text(cell: dict) -> str:
    """Return a notebook cell source as one string.

    Args:
        cell: Raw Jupyter cell dictionary.

    Returns:
        Concatenated cell source.
    """
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def source_lines(text: str) -> list[str]:
    """Convert source text to Jupyter's line-list representation.

    Args:
        text: Cell source text.

    Returns:
        Lines retaining newline characters.
    """
    text = text.rstrip() + "\n" if text.strip() else ""
    return text.splitlines(keepends=True)


def semantic_fingerprint(cells: list[dict]) -> str:
    """Hash executable syntax while ignoring cell boundaries.

    Args:
        cells: Notebook cells in execution order.

    Returns:
        SHA-256 fingerprint of the combined AST.
    """
    code = "\n\n".join(source_text(cell) for cell in cells if cell.get("cell_type") == "code")
    payload = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def statement_inventory_fingerprint(cells: list[dict]) -> str:
    """Hash the multiset of top-level statements, independent of safe regrouping.

    Args:
        cells: Notebook cells.

    Returns:
        SHA-256 fingerprint of all top-level statements with duplicate counts retained.
    """
    statements: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse(source_text(cell))
        statements.extend(
            ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in tree.body
        )
    payload = "\n".join(sorted(statements))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def output_fingerprint(cells: list[dict]) -> str:
    """Hash stored output objects independent of code-cell grouping.

    Args:
        cells: Notebook cells.

    Returns:
        SHA-256 fingerprint of flattened outputs.
    """
    outputs = [
        output
        for cell in cells
        if cell.get("cell_type") == "code"
        for output in cell.get("outputs", [])
    ]
    payload = json.dumps(outputs, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_generated_contracts(cells: list[dict]) -> list[dict]:
    """Remove the previous per-code-cell Markdown boilerplate.

    Args:
        cells: Current notebook cells.

    Returns:
        Cells with original explanatory Markdown retained.
    """
    cleaned: list[dict] = []
    for cell in cells:
        updated = copy.deepcopy(cell)
        updated.get("metadata", {}).pop("refactor_role", None)
        if updated.get("cell_type") == "markdown":
            source = GENERATED_CONTRACT_RE.sub("", source_text(updated)).strip()
            if not source:
                continue
            updated["source"] = source_lines(source)
        cleaned.append(updated)
    without_pair_separators: list[dict] = []
    for index, cell in enumerate(cleaned):
        is_pair_separator = (
            cell.get("cell_type") == "markdown"
            and index > 0
            and index + 1 < len(cleaned)
            and cleaned[index - 1].get("cell_type") == "code"
            and cleaned[index + 1].get("cell_type") == "code"
            and cleaned[index - 1].get("metadata", {}).get("compact_role") == "definitions"
            and cleaned[index + 1].get("metadata", {}).get("compact_role") == "execution"
        )
        if not is_pair_separator:
            without_pair_separators.append(cell)
    return without_pair_separators


def merge_code_cells(cells: list[dict]) -> dict:
    """Merge adjacent code fragments without changing statement order.

    Args:
        cells: Adjacent code cells.

    Returns:
        One code cell containing the same source and outputs in order.
    """
    sources = [source_text(cell).strip() for cell in cells if source_text(cell).strip()]
    outputs = [output for cell in cells for output in cell.get("outputs", [])]
    counts = [cell.get("execution_count") for cell in cells if cell.get("execution_count") is not None]
    metadata: dict = {}
    for cell in cells:
        metadata.update(copy.deepcopy(cell.get("metadata", {})))
    metadata.pop("refactor_role", None)
    return {
        "cell_type": "code",
        "execution_count": counts[-1] if counts else None,
        "metadata": metadata,
        "outputs": outputs,
        "source": source_lines("\n\n".join(sources)),
    }


def merge_markdown_cells(cells: list[dict]) -> dict:
    """Merge adjacent explanatory Markdown blocks.

    Args:
        cells: Adjacent Markdown cells.

    Returns:
        One Markdown cell preserving source order.
    """
    sources = [source_text(cell).strip() for cell in cells if source_text(cell).strip()]
    return {
        "cell_type": "markdown",
        "metadata": copy.deepcopy(cells[0].get("metadata", {})),
        "source": source_lines("\n\n".join(sources)),
    }


def merge_adjacent_cells(cells: list[dict]) -> list[dict]:
    """Merge adjacent cells of the same type after boilerplate removal.

    Args:
        cells: Notebook cells.

    Returns:
        Compact cell sequence.
    """
    merged: list[dict] = []
    index = 0
    while index < len(cells):
        cell_type = cells[index].get("cell_type")
        end = index + 1
        while end < len(cells) and cells[end].get("cell_type") == cell_type:
            end += 1
        group = cells[index:end]
        if cell_type == "code":
            merged.append(merge_code_cells(group))
        elif cell_type == "markdown":
            merged.append(merge_markdown_cells(group))
        else:
            merged.extend(copy.deepcopy(group))
        index = end
    return merged


def statement_is_definition(node: ast.stmt) -> bool:
    """Classify top-level statements used by the safe trailing split.

    Args:
        node: Top-level Python statement.

    Returns:
        ``True`` for imports, functions, classes, and passive declarations.
    """
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = getattr(node, "value", None)
        return value is not None and not any(
            isinstance(child, (ast.Call, ast.Await, ast.Yield))
            or (isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr))
            for child in ast.walk(value)
        )
    return False


def assigned_names(node: ast.stmt) -> set[str]:
    """Collect top-level names assigned by a statement.

    Args:
        node: Top-level Python statement.

    Returns:
        Names created or rebound by the statement.
    """
    names: set[str] = set()
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        targets.append(node.target)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    for target in targets:
        names.update(child.id for child in ast.walk(target) if isinstance(child, ast.Name))
    return names


def evaluation_dependencies(node: ast.stmt) -> set[str]:
    """Collect names required while defining a passive statement.

    Args:
        node: Candidate definition statement.

    Returns:
        Names that must already exist when the statement is evaluated.
    """
    expressions: list[ast.AST] = []
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = getattr(node, "value", None)
        if value is not None:
            expressions.append(value)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        expressions.extend(node.decorator_list)
        expressions.extend(node.args.defaults)
        expressions.extend(value for value in node.args.kw_defaults if value is not None)
        if node.returns is not None:
            expressions.append(node.returns)
        expressions.extend(arg.annotation for arg in [*node.args.args, *node.args.kwonlyargs] if arg.annotation)
    elif isinstance(node, ast.ClassDef):
        expressions.extend(node.decorator_list)
        expressions.extend(node.bases)
        expressions.extend(keyword.value for keyword in node.keywords)
    return {
        child.id
        for expression in expressions
        for child in ast.walk(expression)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def statement_chunks(source: str, tree: ast.Module) -> list[str]:
    """Split source into comment-preserving top-level statement chunks.

    Args:
        source: Code cell source.
        tree: Parsed module for that source.

    Returns:
        Source chunks aligned with ``tree.body``.
    """
    lines = source.splitlines(keepends=True)
    starts = []
    for node in tree.body:
        decorators = [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
        starts.append(min([node.lineno, *decorators]) - 1)
    chunks: list[str] = []
    for index, start in enumerate(starts):
        actual_start = 0 if index == 0 else start
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        chunks.append("".join(lines[actual_start:end]).strip("\n"))
    return chunks


def split_definition_execution(cell: dict) -> list[dict]:
    """Regroup safe definitions before runtime statements within one section.

    Args:
        cell: Merged code cell.

    Returns:
        A collapsed definition cell and a visible execution cell when safe.
    """
    source = source_text(cell)
    tree = ast.parse(source)
    if not tree.body:
        return [cell]
    chunks = statement_chunks(source, tree)
    definition_chunks: list[str] = []
    execution_chunks: list[str] = []
    runtime_names: set[str] = set()
    runtime_seen = False

    for node, chunk in zip(tree.body, chunks):
        is_definition = statement_is_definition(node)
        depends_on_runtime = bool(evaluation_dependencies(node) & runtime_names)
        app_import_after_bootstrap = (
            runtime_seen
            and isinstance(node, ast.ImportFrom)
            and str(node.module or "").startswith("app")
        )
        if is_definition and not depends_on_runtime and not app_import_after_bootstrap:
            definition_chunks.append(chunk)
        else:
            execution_chunks.append(chunk)
            runtime_seen = True
            runtime_names.update(assigned_names(node))

    if not definition_chunks or not execution_chunks:
        return [cell]

    definition_source = "\n\n".join(definition_chunks)
    execution_source = "\n\n".join(execution_chunks)

    definition_cell = copy.deepcopy(cell)
    definition_cell["source"] = source_lines(definition_source)
    definition_cell["execution_count"] = None
    definition_cell["outputs"] = []
    definition_cell.setdefault("metadata", {})["compact_role"] = "definitions"

    execution_cell = copy.deepcopy(cell)
    execution_cell["source"] = source_lines(execution_source)
    execution_cell.setdefault("metadata", {})["compact_role"] = "execution"
    return [definition_cell, execution_cell]


def heading_from_markdown(source: str, fallback: str) -> str:
    """Choose one concise business-level heading from verbose Markdown.

    Args:
        source: Original Markdown source.
        fallback: Heading used when the source has none.

    Returns:
        Heading text without Markdown markers.
    """
    headings = []
    for line in source.splitlines():
        match = re.match(r"^#{1,4}\s+(.+?)\s*$", line.strip())
        if match:
            title = re.sub(r"`", "", match.group(1)).strip()
            if "Kết luận" not in title:
                headings.append(title)
    step_headings = [heading for heading in headings if re.search(r"BƯỚC|Bước", heading)]
    return (step_headings[0] if step_headings else headings[0] if headings else fallback)[:110]


def first_description(source: str, fallback: str) -> str:
    """Extract one useful prose sentence from a Markdown block.

    Args:
        source: Original Markdown source.
        fallback: Description used when prose is unavailable.

    Returns:
        Compact plain-language description.
    """
    candidates: list[str] = []
    in_fence = False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith(("#", "|", "-", ">")):
            continue
        if stripped.startswith(("**", "`")) and len(stripped) < 100:
            continue
        plain = re.sub(r"[*_`]", "", stripped)
        candidates.append(plain)
        if len(" ".join(candidates)) >= 180:
            break
    description = " ".join(candidates).strip() or fallback
    if len(description) > 240:
        description = description[:237].rsplit(" ", 1)[0] + "..."
    return description


def code_symbols(source: str) -> list[str]:
    """Extract representative symbols for the section contract.

    Args:
        source: Code source following the Markdown section.

    Returns:
        Up to five top-level symbol names.
    """
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.extend(target.id for target in targets if isinstance(target, ast.Name))
        if len(names) >= 5:
            break
    return names[:5]


def compact_markdown(notebook_name: str, original: str, code: str, section_index: int) -> str:
    """Replace verbose prose with one required three-line section contract.

    Args:
        notebook_name: Notebook filename.
        original: Original Markdown preceding the section.
        code: Code source executed by the section.
        section_index: One-based section number.

    Returns:
        Concise Markdown section.
    """
    purpose = NOTEBOOK_PURPOSES.get(notebook_name, "xử lý Multimodal RAG")
    symbols = code_symbols(code)
    fallback_title = f"Bước {section_index}: {purpose.capitalize()}"
    title = heading_from_markdown(original, fallback_title)
    effect = first_description(original, f"Thực hiện bước {purpose}.")
    symbol_text = ", ".join(f"`{name}`" for name in symbols) if symbols else "các lệnh trong bước"
    has_output = any(token in code for token in ("print(", "display(", "plt.show(", "pprint("))
    output_text = (
        f"{symbol_text} cùng output debug như shape, ID, score hoặc trạng thái trung gian."
        if has_output
        else f"{symbol_text} để các bước sau tiếp tục sử dụng."
    )
    warning = ""
    if "LEGACY REFERENCE" in original:
        warning = "\n> **Legacy reference:** Chỉ dùng để đối chiếu; luồng chính nằm ở notebook 05 V2.\n"
    return (
        f"### {title}\n"
        f"{warning}\n"
        f"- **Tác dụng chính:** {effect}\n"
        "- **Đầu vào (Input):** Cấu hình, dữ liệu hoặc đối tượng đã được chuẩn bị ở bước trước; "
        "tham số chi tiết nằm trong docstring của hàm.\n"
        f"- **Đầu ra (Output):** {output_text}\n"
    )


def compact_intro(notebook_name: str, source: str) -> str:
    """Keep a short notebook title and purpose statement.

    Args:
        notebook_name: Notebook filename.
        source: Original first Markdown block.

    Returns:
        Compact notebook introduction.
    """
    heading = NOTEBOOK_TITLES.get(notebook_name, notebook_name.removesuffix(".ipynb"))
    purpose = NOTEBOOK_PURPOSES.get(notebook_name, "trình diễn Multimodal RAG")
    legacy = (
        "\n> **Legacy reference:** Notebook này chỉ dùng để đối chiếu; hãy ưu tiên bản 05 V2.\n"
        if "LEGACY REFERENCE" in source or notebook_name == "05_layer_b_outfit_and_category_mapping.ipynb"
        else ""
    )
    return (
        f"# {heading}\n\n"
        f"Notebook này dùng để **{purpose}**. Phần triển khai dài được thu gọn mặc định; "
        "các cell thực thi vẫn giữ output cần thiết cho debug.\n"
        f"{legacy}"
    )


def apply_compact_markdown(notebook_name: str, cells: list[dict]) -> list[dict]:
    """Attach one concise Markdown contract to each business section.

    Args:
        notebook_name: Notebook filename.
        cells: Merged notebook cells.

    Returns:
        Cells with compact introductions and section descriptions.
    """
    result: list[dict] = []
    section_index = 0
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            result.append(copy.deepcopy(cell))
            continue
        if cell.get("cell_type") != "code":
            result.append(copy.deepcopy(cell))
            continue

        continues_definition_pair = (
            result
            and result[-1].get("cell_type") == "code"
            and result[-1].get("metadata", {}).get("compact_role") == "definitions"
            and cell.get("metadata", {}).get("compact_role") == "execution"
        )
        if continues_definition_pair:
            result.append(copy.deepcopy(cell))
            continue

        section_index += 1
        contract_code = source_text(cell)
        if (
            cell.get("metadata", {}).get("compact_role") == "definitions"
            and index + 1 < len(cells)
            and cells[index + 1].get("cell_type") == "code"
            and cells[index + 1].get("metadata", {}).get("compact_role") == "execution"
        ):
            contract_code += "\n\n" + source_text(cells[index + 1])
        if result and result[-1].get("cell_type") == "markdown":
            markdown = result[-1]
            if section_index == 1:
                intro = compact_intro(notebook_name, source_text(markdown))
                section = compact_markdown(notebook_name, source_text(markdown), contract_code, section_index)
                markdown["source"] = source_lines(intro + "\n" + section)
            else:
                markdown["source"] = source_lines(
                    compact_markdown(notebook_name, source_text(markdown), contract_code, section_index)
                )
        else:
            section = compact_markdown(notebook_name, "", contract_code, section_index)
            result.append({"cell_type": "markdown", "metadata": {}, "source": source_lines(section)})
        result.append(copy.deepcopy(cell))
    return result


def collapse_implementation_cells(cells: list[dict]) -> None:
    """Mark definition-heavy code cells as collapsed in common Jupyter frontends.

    Args:
        cells: Notebook cells modified in place.

    Returns:
        None: Cell metadata is updated in place.
    """
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        metadata = cell.setdefault("metadata", {})
        metadata.pop("collapsed", None)
        if isinstance(metadata.get("jupyter"), dict):
            metadata["jupyter"].pop("source_hidden", None)
            if not metadata["jupyter"]:
                metadata.pop("jupyter", None)
        if isinstance(metadata.get("tags"), list):
            metadata["tags"] = [tag for tag in metadata["tags"] if tag != "implementation"]
            if not metadata["tags"]:
                metadata.pop("tags", None)

        source = source_text(cell)
        tree = ast.parse(source)
        has_definitions = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
        has_runtime = any(
            isinstance(node, (ast.Expr, ast.For, ast.While, ast.Try, ast.With, ast.If, ast.Match))
            for node in tree.body
        )
        definition_only = has_definitions and not has_runtime
        if cell.get("metadata", {}).get("compact_role") == "definitions" or definition_only:
            metadata["collapsed"] = True
            metadata.setdefault("jupyter", {})["source_hidden"] = True
            tags = metadata.setdefault("tags", [])
            if "implementation" not in tags:
                tags.append("implementation")


def assign_cell_ids(notebook_name: str, cells: list[dict]) -> None:
    """Assign deterministic unique cell IDs.

    Args:
        notebook_name: Notebook filename.
        cells: Cells modified in place.

    Returns:
        None: Cell IDs are updated in place.
    """
    for index, cell in enumerate(cells):
        seed = f"compact:{notebook_name}:{index}:{cell.get('cell_type')}:{source_text(cell)}"
        cell["id"] = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def compact_notebook(path: Path) -> tuple[int, int]:
    """Compact one notebook and validate source/output preservation.

    Args:
        path: Notebook file to update.

    Returns:
        Old and new cell counts.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original_cells = copy.deepcopy(notebook.get("cells", []))
    if path.name == "00_INDEX.ipynb":
        for cell in original_cells:
            if cell.get("cell_type") != "markdown":
                continue
            source = source_text(cell).replace(
                "File không có hậu tố `_v2` được giữ làm **legacy reference** để đối chiếu quá trình cải tiến, "
                "không nên được import như implementation production.",
                "File không có hậu tố `_v2` hiện là trang điều hướng ngắn; bản đầy đủ được lưu tại "
                "`archive/05_layer_b_outfit_and_category_mapping_legacy.ipynb` để đối chiếu lịch sử.",
            )
            cell["source"] = source_lines(source)
        assign_cell_ids(path.name, original_cells)
        notebook["cells"] = original_cells
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return len(original_cells), len(original_cells)

    cells = strip_generated_contracts(original_cells)
    cells = merge_adjacent_cells(cells)
    safely_split: list[dict] = []
    for cell in cells:
        safely_split.extend(split_definition_execution(cell) if cell.get("cell_type") == "code" else [cell])
    cells = apply_compact_markdown(path.name, safely_split)
    collapse_implementation_cells(cells)
    assign_cell_ids(path.name, cells)

    if statement_inventory_fingerprint(original_cells) != statement_inventory_fingerprint(cells):
        raise RuntimeError(f"Executable statements changed unexpectedly: {path.name}")
    if output_fingerprint(original_cells) != output_fingerprint(cells):
        raise RuntimeError(f"Stored outputs changed unexpectedly: {path.name}")

    notebook["cells"] = cells
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1)
    json.loads(serialized)
    path.write_text(serialized + "\n", encoding="utf-8")
    return len(original_cells), len(cells)


def archive_legacy_notebook(path: Path) -> tuple[int, int]:
    """Move the executable legacy notebook behind a short navigation page.

    Args:
        path: Top-level legacy notebook path.

    Returns:
        Old and new top-level cell counts.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    old_count = len(notebook.get("cells", []))
    has_code = any(cell.get("cell_type") == "code" for cell in notebook.get("cells", []))
    if not has_code:
        return old_count, old_count

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / ARCHIVED_LEGACY_NOTEBOOK
    archive_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    pointer = (
        "# 05 - Layer B Outfit Rule Debug (Legacy)\n\n"
        "> Notebook này đã được chuyển sang khu vực lưu trữ để luồng đọc chính ngắn gọn hơn.\n\n"
        "- Luồng đang sử dụng: "
        "[`05_layer_b_outfit_and_category_mapping_v2.ipynb`](05_layer_b_outfit_and_category_mapping_v2.ipynb)\n"
        "- Bản đầy đủ để đối chiếu lịch sử: "
        "[`archive/05_layer_b_outfit_and_category_mapping_legacy.ipynb`]"
        "(archive/05_layer_b_outfit_and_category_mapping_legacy.ipynb)\n\n"
        "Không có code thực thi trong trang điều hướng này, vì vậy người đọc sẽ không vô tình chạy nhầm pipeline legacy.\n"
    )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": source_lines(pointer)}]
    assign_cell_ids(path.name, cells)
    notebook["cells"] = cells
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = max(5, int(notebook.get("nbformat_minor", 5)))
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return old_count, len(cells)


def main() -> None:
    """Compact every notebook in the Research Demo v3 split set."""
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        if path.name == LEGACY_NOTEBOOK:
            old_count, new_count = archive_legacy_notebook(path)
            print(f"[OK] {path.name}: {old_count} -> {new_count} cells (legacy archived)")
            continue
        old_count, new_count = compact_notebook(path)
        print(f"[OK] {path.name}: {old_count} -> {new_count} cells")


if __name__ == "__main__":
    main()
