"""Build searchable indexes over the FoodOn OWL ontology."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS

FOODON_OWL_URL = "http://purl.obolibrary.org/obo/foodon.owl"
OBO_BASE = "http://purl.obolibrary.org/obo/"
BFO = Namespace(OBO_BASE)

PREFERRED_ROOT_IDS = ("BFO_0000040",)  # material entity


def uri_to_id(uri: str) -> str:
    text = str(uri)
    if text.startswith(OBO_BASE):
        return text[len(OBO_BASE) :]
    return text.rsplit("/", 1)[-1]


def id_to_uri(node_id: str) -> str:
    if node_id.startswith("http://") or node_id.startswith("https://"):
        return node_id
    return f"{OBO_BASE}{node_id}"


def _norm_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _label_match_score(term: str, label: str) -> float:
    term_n, label_n = _norm_label(term), _norm_label(label)
    if not term_n or not label_n:
        return 0.0
    if term_n == label_n:
        return 1.0
    if label_n.startswith(term_n):
        return 0.92 + 0.08 * (len(term_n) / len(label_n))
    if term_n in label_n:
        return 0.75 + 0.2 * (len(term_n) / len(label_n))
    return difflib.SequenceMatcher(None, term_n, label_n).ratio()


class FoodOnIndex:
    """In-memory FoodOn class hierarchy with labels and descendant counts."""

    def __init__(
        self,
        labels: dict[str, str],
        children: dict[str, list[str]],
        parents: dict[str, list[str]],
    ):
        self.labels = labels
        self.children = children
        self.parents = parents
        self.descendant_counts = self._compute_descendant_counts()
        self.roots = self._compute_roots()

    @classmethod
    def from_graph(cls, graph: Graph) -> FoodOnIndex:
        labels: dict[str, str] = {}
        parents: dict[str, list[str]] = {}
        children: dict[str, list[str]] = {}

        for class_uri in graph.subjects(RDF.type, OWL.Class):
            node_id = uri_to_id(class_uri)
            for label in graph.objects(class_uri, RDFS.label):
                lang = getattr(label, "language", None)
                if lang and lang != "en":
                    continue
                labels.setdefault(node_id, str(label))

        for child, parent in graph.subject_objects(RDFS.subClassOf):
            child_id = uri_to_id(child)
            if child_id not in labels:
                continue
            if str(parent).startswith(str(OWL)):
                continue
            parent_id = uri_to_id(parent)
            parents.setdefault(child_id, []).append(parent_id)
            children.setdefault(parent_id, []).append(child_id)

        for parent_id, child_ids in children.items():
            children[parent_id] = sorted(
                set(child_ids), key=lambda cid: labels.get(cid, cid).lower()
            )

        return cls(labels, children, parents)

    @classmethod
    def from_owl(
        cls,
        owl_url: str = FOODON_OWL_URL,
        cache_path: Optional[Path] = None,
    ) -> FoodOnIndex:
        if cache_path and cache_path.exists():
            return cls.from_json(cache_path.read_text(encoding="utf-8"))

        graph = Graph()
        graph.parse(owl_url)
        index = cls.from_graph(graph)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(index.to_json(owl_url=owl_url), encoding="utf-8")
        return index

    @classmethod
    def from_json(cls, payload: str) -> FoodOnIndex:
        data = json.loads(payload)
        return cls(data["labels"], data["children"], data.get("parents", {}))

    def to_json(self, *, owl_url: str = FOODON_OWL_URL) -> str:
        return json.dumps(
            {
                "version": 1,
                "owl_url": owl_url,
                "labels": self.labels,
                "children": self.children,
                "parents": self.parents,
            },
            ensure_ascii=False,
        )

    def _compute_descendant_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node_id in self.labels:
            seen: set[str] = set()
            stack = list(self.children.get(node_id, []))
            while stack:
                child_id = stack.pop()
                if child_id in seen:
                    continue
                seen.add(child_id)
                stack.extend(self.children.get(child_id, []))
            counts[node_id] = len(seen)
        return counts

    def _compute_roots(self) -> list[str]:
        has_parent = set(self.parents)
        roots = [node_id for node_id in self.labels if node_id not in has_parent]
        roots.sort(key=lambda node_id: self.labels.get(node_id, node_id).lower())
        return roots

    def preferred_roots(self) -> list[str]:
        preferred = [node_id for node_id in PREFERRED_ROOT_IDS if node_id in self.labels]
        if preferred:
            return preferred
        return self.roots

    def node_summary(self, node_id: str) -> Optional[dict]:
        if node_id not in self.labels:
            return None
        child_ids = self.children.get(node_id, [])
        return {
            "id": node_id,
            "label": self.labels[node_id],
            "is_bfo": node_id.startswith("BFO_"),
            "descendant_count": self.descendant_counts.get(node_id, 0),
            "child_count": len(child_ids),
            "has_children": bool(child_ids),
        }

    def child_summaries(self, node_id: str) -> list[dict]:
        summaries = []
        for child_id in self.children.get(node_id, []):
            summary = self.node_summary(child_id)
            if summary:
                summaries.append(summary)
        summaries.sort(key=lambda row: row["label"].lower())
        return summaries

    def _choose_parent(self, child_id: str) -> Optional[str]:
        options = self.parents.get(child_id, [])
        if not options:
            return None
        if len(options) == 1:
            return options[0]

        def parent_rank(parent_id: str) -> tuple[int, int]:
            label = _norm_label(self.labels.get(parent_id, ""))
            score = 0
            if "food" in label:
                score += 4
            if "material" in label:
                score += 2
            if "product" in label:
                score += 1
            if "entity" in label:
                score += 1
            if parent_id.startswith("BFO_"):
                score += 1
            return (score, len(label))

        return max(options, key=parent_rank)

    def ancestry_path(self, node_id: str) -> list[str]:
        chain = [node_id]
        seen = {node_id}
        current = node_id
        while True:
            parent_id = self._choose_parent(current)
            if parent_id is None or parent_id in seen or parent_id not in self.labels:
                break
            seen.add(parent_id)
            chain.append(parent_id)
            current = parent_id
        chain.reverse()
        return chain

    def search(self, term: str, *, limit: int = 25, min_score: float = 0.45) -> list[dict]:
        if not term.strip():
            return []

        scored: list[tuple[float, str, str]] = []
        for node_id, label in self.labels.items():
            score = _label_match_score(term, label)
            if score >= min_score:
                scored.append((score, node_id, label))

        scored.sort(key=lambda row: (-row[0], row[2].lower()))
        results = []
        for score, node_id, label in scored[:limit]:
            results.append(
                {
                    "id": node_id,
                    "label": label,
                    "score": round(score, 3),
                    "is_bfo": node_id.startswith("BFO_"),
                    "descendant_count": self.descendant_counts.get(node_id, 0),
                    "child_count": len(self.children.get(node_id, [])),
                    "has_children": bool(self.children.get(node_id)),
                }
            )
        return results

    def labeled_ancestry(self, node_id: str) -> list[dict]:
        """Return ancestry chain as id/label rows (FoodOn-labeled nodes only)."""
        rows = []
        for ancestor_id in self.ancestry_path(node_id):
            if ancestor_id not in self.labels:
                continue
            summary = self.node_summary(ancestor_id)
            if summary:
                rows.append(summary)
        return rows

    def find_node_by_label(self, label: str) -> str | None:
        """Return first node id whose English label matches (normalized)."""
        target = _norm_label(label)
        if not target:
            return None
        for node_id, node_label in self.labels.items():
            if _norm_label(node_label) == target:
                return node_id
        return None

    def descendants_of(self, root_id: str, *, include_root: bool = True) -> set[str]:
        """All class ids in the subtree below ``root_id``."""
        if root_id not in self.labels:
            return set()
        seen: set[str] = set()
        if include_root:
            seen.add(root_id)
        stack = list(self.children.get(root_id, []))
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(self.children.get(node_id, []))
        return seen

    def leaf_ids_in(self, node_ids: set[str]) -> set[str]:
        """Subset of ``node_ids`` that have no children in the ontology."""
        return {
            node_id
            for node_id in node_ids
            if node_id in self.labels and not self.children.get(node_id)
        }

    def food_material_root_id(self) -> str | None:
        """Resolve the 'food material' class under material entity (BFO_0000040)."""
        preferred = "BFO_0000040"
        for child_id in self.children.get(preferred, []):
            if _norm_label(self.labels.get(child_id, "")) == _norm_label("food material"):
                return child_id
        return self.find_node_by_label("food material")

    def food_material_subtree_leaves(self) -> set[str]:
        """Leaf FoodOn classes under material entity → food material."""
        root = self.food_material_root_id()
        if root is None:
            return set()
        subtree = self.descendants_of(root, include_root=True)
        return self.leaf_ids_in(subtree)


def choose_foodon_match(
    index: FoodOnIndex,
    term: str,
    *,
    limit: int = 25,
    min_score: float = 0.55,
) -> Optional[dict]:
    """Pick the best FoodOn class for a search term, preferring exact and leaf hits."""
    hits = index.search(term, limit=limit, min_score=min_score)
    if not hits:
        return None

    term_n = _norm_label(term)
    exact = [hit for hit in hits if _norm_label(hit["label"]) == term_n]
    if exact:
        leaves = [hit for hit in exact if not hit["has_children"]]
        return leaves[0] if leaves else exact[0]

    top = hits[0]
    if top["score"] >= 0.95:
        return top

    leaves = [hit for hit in hits if not hit["has_children"]]
    strong_leaves = [hit for hit in leaves if hit["score"] >= 0.9]
    if strong_leaves:
        return strong_leaves[0]
    if leaves:
        return leaves[0]
    return top


def fdc_description_search_terms(fdc_description: str) -> list[str]:
    """Derive FoodOn search phrases from a USDA FDC description string."""
    text = (fdc_description or "").strip()
    if not text:
        return []

    parts = [part.strip().lower() for part in text.split(",") if part.strip()]
    joined = " ".join(parts)
    terms: list[str] = []

    if len(parts) >= 3 and parts[:2] == ["spices", "pepper"]:
        terms.append(f"{parts[2]} pepper")
    elif parts and parts[0] == "cheese" and len(parts) >= 2:
        cheese = parts[1]
        modifier = parts[2] if len(parts) > 2 else ""
        if modifier:
            terms.append(f"{cheese} cheese ({modifier})")
        terms.append(f"{cheese} cheese")
    elif parts and parts[0] == "egg":
        if "yolk" in parts:
            terms.extend(["egg yolk", "chicken egg yolk"])
        elif "white" in parts:
            terms.extend(["egg white", "chicken egg white"])
        else:
            terms.extend(["chicken egg (raw)", "chicken egg"])
    elif len(parts) >= 2 and parts[:2] == ["oil", "olive"]:
        terms.append("olive oil")
    elif parts and "bacon" in parts:
        terms.extend(["bacon (raw)", "bacon"])
    elif parts and (parts[0].startswith("alcoholic beverage") or "wine" in parts):
        if "white" in parts:
            terms.append("white wine")
        elif "red" in parts:
            terms.append("red wine")
    elif parts and parts[0] == "salt":
        terms.append("table salt")
    elif parts and parts[0] == "lemons":
        terms.extend(["lemon (raw)", "lemon"])
    elif parts and (
        parts[0] == "lemon juice" or (parts[0] == "lemon" and any("juice" in part for part in parts))
    ):
        terms.extend(["lemon juice", "lemon (raw)", "lemon"])
    elif parts and parts[0] == "limeade":
        # Some resolved lines use this FDC string for lemon juice in recipes.
        terms.extend(["lemon juice", "lemon (raw)", "lemon"])
    elif parts and parts[0] == "garlic":
        terms.extend(["garlic clove", "garlic (raw)", "garlic"])
    elif parts and parts[0] == "parsley":
        terms.extend(["parsley leaf (raw)", "parsley"])
    elif parts and parts[0] == "onions":
        terms.append("onion")
    elif parts and parts[0] == "cream":
        if any("light" in part for part in parts):
            terms.append("light cream")
        elif any(part in {"heavy", "whipping"} or "heavy" in part or "whipping" in part for part in parts):
            terms.append("heavy cream")
        else:
            terms.append("cream")
    elif parts and parts[0] == "squash" and "zucchini" in parts:
        terms.extend(["zucchini squash", "zucchini"])
    elif "pancetta" in joined:
        terms.append("pancetta")
    elif "pecorino" in joined or "parmigiano" in joined or "reggiano" in joined:
        if "grated" in joined or "pecorino" in joined:
            terms.extend(["romano cheese (grated)", "pecorino romano", "parmesan cheese (grated)"])
        terms.extend(["parmigiano reggiano", "romano cheese"])
    elif text.upper() == text and len(parts) == 1:
        terms.append(parts[0])
    elif joined == "spaghetti":
        terms.append("spaghetti")

    terms.append(joined)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def map_fdc_description_to_foodon(
    index: FoodOnIndex,
    fdc_description: str,
    *,
    min_score: float = 0.55,
) -> Optional[dict]:
    """Map one FDC description to the best FoodOn class (leaf when possible)."""
    best: Optional[dict] = None
    best_term: Optional[str] = None

    for term in fdc_description_search_terms(fdc_description):
        candidate = choose_foodon_match(index, term, min_score=min_score)
        if candidate is None:
            continue
        if best is None or candidate["score"] > best["score"]:
            best = dict(candidate)
            best_term = term

    if best is None:
        return None

    best["search_term"] = best_term
    best["fdc_description"] = fdc_description
    best["ancestry"] = index.labeled_ancestry(best["id"])
    return best
