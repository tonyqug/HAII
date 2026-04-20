from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .repository import Repository
from .utils import (
    best_snippet,
    clamp_quality,
    estimate_tokens,
    json_dumps,
    quality_rank,
    significant_terms,
    stable_bundle_id,
    stable_citation_id,
    unique_preserve_order,
)


ANNOTATION_PSEUDO_MATERIAL_PREFIX = "annotation_material:"
ANNOTATION_PSEUDO_SLIDE_PREFIX = "annotation_slide:"
ANNOTATION_MATERIAL_TITLE = "Workspace Annotations"


@dataclass
class Candidate:
    source_type: str  # slide | annotation
    workspace_id: str
    material_id: Optional[str]
    slide_id: Optional[str]
    slide_number: Optional[int]
    material_title: str
    text: str
    extraction_quality: str
    support_type: str
    annotation_id: Optional[str] = None
    score: float = 0.0
    scope: Optional[str] = None
    role: Optional[str] = None

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (-self.score, self.material_title.lower(), self.slide_number or 0, self.slide_id or self.annotation_id or "")

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.text)


def material_source_view_url(settings: Settings, material_id: str) -> str:
    return f"{settings.api_base_url}/v1/materials/{material_id}/source"


def slide_preview_url(settings: Settings, material_id: str, slide_id: str) -> str:
    return f"{settings.api_base_url}/v1/materials/{material_id}/slides/{slide_id}/preview"


def slide_source_open_url(settings: Settings, material_id: str, slide_id: str) -> str:
    return f"{settings.api_base_url}/v1/materials/{material_id}/slides/{slide_id}/source"


def annotation_material_id(workspace_id: str) -> str:
    return f"{ANNOTATION_PSEUDO_MATERIAL_PREFIX}{workspace_id}"


def annotation_slide_id(annotation_id: str) -> str:
    return f"{ANNOTATION_PSEUDO_SLIDE_PREFIX}{annotation_id}"


def annotation_preview_url(settings: Settings, workspace_id: str, annotation_id: str) -> str:
    return f"{settings.api_base_url}/v1/workspaces/{workspace_id}/annotations/{annotation_id}/preview"


def annotation_source_open_url(settings: Settings, workspace_id: str, annotation_id: str) -> str:
    return f"{settings.api_base_url}/v1/workspaces/{workspace_id}/annotations/{annotation_id}/source"


def public_material(material: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "material_id": material["material_id"],
        "workspace_id": material["workspace_id"],
        "title": material["title"],
        "role": material["role"],
        "kind": material["kind"],
        "processing_status": material["processing_status"],
        "page_count": int(material["page_count"] or 0),
        "created_at": material["created_at"],
        "quality_summary": {
            "overall": clamp_quality(material.get("quality_overall")),
            "notes": material.get("quality_notes"),
        },
        "source_view_url": material_source_view_url(settings, material["material_id"]),
    }


def public_material_detail(material: dict[str, Any], settings: Settings) -> dict[str, Any]:
    item = public_material(material, settings)
    item.update(
        {
            "original_filename": material.get("original_filename"),
            "extraction_notes": material.get("extraction_notes"),
            "slide_count": int(material.get("slide_count") or material.get("page_count") or 0),
            "page_count": int(material.get("page_count") or 0),
            "ready_for_retrieval": bool(material.get("ready_for_retrieval")),
            "previews_available": bool(material.get("previews_available")),
        }
    )
    return item


def public_slide(material_id: str, slide: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "slide_id": slide["slide_id"],
        "slide_number": int(slide["slide_number"]),
        "title_guess": slide.get("title_guess"),
        "preview_url": slide_preview_url(settings, material_id, slide["slide_id"]),
        "source_open_url": slide_source_open_url(settings, material_id, slide["slide_id"]),
        "extraction_quality": clamp_quality(slide["extraction_quality"]),
        "has_text": bool(slide.get("has_text")),
    }


def public_slide_detail(material_id: str, slide: dict[str, Any], settings: Settings) -> dict[str, Any]:
    item = public_slide(material_id, slide, settings)
    item.update(
        {
            "material_id": material_id,
            "extracted_text": slide.get("extracted_text", ""),
            "quality_notes": slide.get("quality_notes"),
        }
    )
    return item


def public_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_id": annotation["annotation_id"],
        "annotation_type": annotation["annotation_type"],
        "scope": annotation["scope"],
        "material_id": annotation.get("material_id"),
        "slide_id": annotation.get("slide_id"),
        "text": annotation["text"],
        "created_at": annotation["created_at"],
    }


class RetrievalEngine:
    def __init__(self, settings: Settings, repo: Repository):
        self.settings = settings
        self.repo = repo

    def _workspace_materials(self, workspace_id: str, material_ids: Optional[List[str]] = None) -> list[dict[str, Any]]:
        materials = self.repo.list_materials(workspace_id)
        materials = [m for m in materials if m["processing_status"] == "ready"]
        if material_ids:
            wanted = set(material_ids)
            materials = [m for m in materials if m["material_id"] in wanted]
        return materials

    def _annotation_sets(self, workspace_id: str) -> dict[str, list[dict[str, Any]]]:
        annotations = self.repo.list_annotations(workspace_id)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            buckets[annotation["annotation_type"]].append(annotation)
        return buckets

    def _exclusion_state(self, workspace_id: str) -> tuple[bool, set[str], set[str]]:
        excluded_annotations = self._annotation_sets(workspace_id).get("exclude_from_grounding", [])
        workspace_excluded = any(a["scope"] == "workspace" for a in excluded_annotations)
        excluded_materials = {a["material_id"] for a in excluded_annotations if a.get("material_id") and a["scope"] == "material"}
        excluded_slides = {a["slide_id"] for a in excluded_annotations if a.get("slide_id") and a["scope"] == "slide"}
        return workspace_excluded, excluded_materials, excluded_slides

    def _focus_annotations(self, workspace_id: str) -> list[dict[str, Any]]:
        return self._annotation_sets(workspace_id).get("focus", [])

    def _annotation_candidates(
        self,
        *,
        workspace_id: str,
        include_annotations: bool,
        selected_material_ids: set[str],
        excluded_materials: set[str],
        excluded_slides: set[str],
    ) -> list[Candidate]:
        if not include_annotations:
            return []
        out: list[Candidate] = []
        for annotation in self.repo.list_annotations(workspace_id):
            if annotation["annotation_type"] not in {"user_correction", "study_note"}:
                continue
            if annotation["scope"] == "material" and annotation.get("material_id") not in selected_material_ids:
                continue
            if annotation["scope"] == "slide" and annotation.get("material_id") not in selected_material_ids:
                continue
            if annotation.get("material_id") in excluded_materials:
                continue
            if annotation.get("slide_id") in excluded_slides:
                continue
            out.append(
                Candidate(
                    source_type="annotation",
                    workspace_id=workspace_id,
                    material_id=annotation_material_id(workspace_id),
                    slide_id=annotation_slide_id(annotation["annotation_id"]),
                    slide_number=int(annotation.get("virtual_slide_number") or 0),
                    material_title=ANNOTATION_MATERIAL_TITLE,
                    text=annotation["text"],
                    extraction_quality="high",
                    support_type="supplemental_note",
                    annotation_id=annotation["annotation_id"],
                    scope=annotation["scope"],
                )
            )
        return out

    def _slide_candidates(
        self,
        *,
        materials: list[dict[str, Any]],
        min_extraction_quality: str,
        excluded_materials: set[str],
        excluded_slides: set[str],
    ) -> list[Candidate]:
        min_rank = quality_rank(min_extraction_quality)
        out: list[Candidate] = []
        for material in materials:
            if material["material_id"] in excluded_materials:
                continue
            for slide in self.repo.list_slides(material["material_id"]):
                if slide["slide_id"] in excluded_slides:
                    continue
                if quality_rank(slide["extraction_quality"]) < min_rank:
                    continue
                if not (slide.get("extracted_text") or "").strip():
                    continue
                out.append(
                    Candidate(
                        source_type="slide",
                        workspace_id=material["workspace_id"],
                        material_id=material["material_id"],
                        slide_id=slide["slide_id"],
                        slide_number=int(slide["slide_number"]),
                        material_title=material["title"],
                        text=slide.get("extracted_text", ""),
                        extraction_quality=clamp_quality(slide["extraction_quality"]),
                        support_type="explicit",
                        role=material["role"],
                    )
                )
        return out

    def _tfidf_scores(self, query_text: str, candidates: list[Candidate]) -> list[float]:
        if not candidates:
            return []
        query_text = (query_text or "").strip()
        if not query_text:
            return [0.0] * len(candidates)
        docs = [query_text] + [candidate.text for candidate in candidates]
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            query_vec = matrix[0]
            doc_vecs = matrix[1:]
            scores = (doc_vecs @ query_vec.T).toarray().ravel()
            return [float(score) for score in scores]
        except Exception:
            q_terms = Counter(significant_terms(query_text))
            scores: list[float] = []
            for candidate in candidates:
                c_terms = Counter(significant_terms(candidate.text))
                overlap = sum(min(q_terms[t], c_terms[t]) for t in q_terms)
                scores.append(float(overlap))
            return scores

    def _focus_boost(self, candidate: Candidate, query_text: str, focus_annotations: list[dict[str, Any]], include_annotations: bool) -> float:
        if not include_annotations:
            return 0.0
        q_terms = set(significant_terms(query_text))
        c_terms = set(significant_terms(candidate.text))
        boost = 0.0
        for annotation in focus_annotations:
            a_terms = set(significant_terms(annotation["text"]))
            scope = annotation["scope"]
            if scope == "slide" and annotation.get("slide_id") == candidate.slide_id:
                boost += 0.35
            elif scope == "material" and annotation.get("material_id") == candidate.material_id:
                boost += 0.18
            elif scope == "workspace":
                if q_terms and a_terms and q_terms.intersection(a_terms):
                    boost += 0.08
            if a_terms and c_terms and a_terms.intersection(c_terms):
                boost += 0.04
        return boost

    def _candidate_score(
        self,
        *,
        candidate: Candidate,
        base_score: float,
        query_text: str,
        focus_annotations: list[dict[str, Any]],
        include_annotations: bool,
        retrieval_mode: str,
    ) -> float:
        score = base_score
        score += self._focus_boost(candidate, query_text, focus_annotations, include_annotations)
        if candidate.source_type == "annotation":
            score += 0.06 if retrieval_mode in {"broad", "coverage"} else 0.02
        if candidate.support_type == "explicit":
            score += {"high": 0.12, "medium": 0.05, "low": 0.0}[candidate.extraction_quality]
        else:
            score += 0.08
        return score

    def _rank_candidates(
        self,
        *,
        query_text: str,
        candidates: list[Candidate],
        include_annotations: bool,
        workspace_id: str,
        retrieval_mode: str,
    ) -> list[Candidate]:
        if not candidates:
            return []
        focus_annotations = self._focus_annotations(workspace_id)
        base_scores = self._tfidf_scores(query_text, candidates)
        for candidate, base in zip(candidates, base_scores, strict=False):
            candidate.score = self._candidate_score(
                candidate=candidate,
                base_score=base,
                query_text=query_text,
                focus_annotations=focus_annotations,
                include_annotations=include_annotations,
                retrieval_mode=retrieval_mode,
            )
        return sorted(candidates, key=lambda candidate: candidate.sort_key)

    def _diversified_select(self, ranked: list[Candidate], top_k: int, mode: str) -> list[Candidate]:
        if top_k <= 0:
            return []
        if mode == "precision":
            return ranked[:top_k]
        selected: list[Candidate] = []
        remaining = ranked.copy()
        material_counts: Counter[str] = Counter()
        while remaining and len(selected) < top_k:
            best_idx = 0
            best_value = float("-inf")
            for idx, candidate in enumerate(remaining):
                material_key = candidate.material_id or candidate.annotation_id or ""
                penalty = 0.0
                penalty += material_counts[material_key] * (0.22 if mode == "coverage" else 0.12)
                for existing in selected:
                    if existing.material_id == candidate.material_id and existing.slide_number and candidate.slide_number:
                        delta = abs(existing.slide_number - candidate.slide_number)
                        if delta == 0:
                            penalty += 0.5
                        elif delta <= 1:
                            penalty += 0.18 if mode == "coverage" else 0.08
                        elif delta <= 3:
                            penalty += 0.08 if mode == "coverage" else 0.03
                value = candidate.score - penalty
                if value > best_value:
                    best_value = value
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            material_key = chosen.material_id or chosen.annotation_id or ""
            material_counts[material_key] += 1
        return selected

    def _make_citation(self, candidate: Candidate, query_text: str) -> dict[str, Any]:
        confidence = candidate.extraction_quality if candidate.source_type == "slide" else "high"
        snippet = best_snippet(candidate.text, query_text)
        if candidate.source_type == "slide":
            citation_id = stable_citation_id(
                candidate.source_type,
                candidate.material_id or "",
                candidate.slide_id or "",
                candidate.support_type,
                snippet,
            )
            self.repo.upsert_citation(
                citation_id=citation_id,
                workspace_id=candidate.workspace_id,
                source_type="slide",
                material_id=candidate.material_id,
                slide_id=candidate.slide_id,
                slide_number=candidate.slide_number,
                annotation_id=None,
                snippet_text=snippet,
                support_type=candidate.support_type,
                confidence=confidence,
            )
            return {
                "citation_id": citation_id,
                "material_id": candidate.material_id,
                "material_title": candidate.material_title,
                "slide_id": candidate.slide_id,
                "slide_number": candidate.slide_number,
                "snippet_text": snippet,
                "support_type": candidate.support_type,
                "confidence": confidence,
                "preview_url": slide_preview_url(self.settings, candidate.material_id or "", candidate.slide_id or ""),
                "source_open_url": slide_source_open_url(self.settings, candidate.material_id or "", candidate.slide_id or ""),
            }
        citation_id = stable_citation_id(
            candidate.source_type,
            candidate.workspace_id,
            candidate.annotation_id or "",
            candidate.support_type,
            snippet,
        )
        self.repo.upsert_citation(
            citation_id=citation_id,
            workspace_id=candidate.workspace_id,
            source_type="annotation",
            material_id=None,
            slide_id=None,
            slide_number=candidate.slide_number,
            annotation_id=candidate.annotation_id,
            snippet_text=snippet,
            support_type=candidate.support_type,
            confidence=confidence,
        )
        return {
            "citation_id": citation_id,
            "material_id": annotation_material_id(candidate.workspace_id),
            "material_title": ANNOTATION_MATERIAL_TITLE,
            "slide_id": annotation_slide_id(candidate.annotation_id or ""),
            "slide_number": candidate.slide_number,
            "snippet_text": snippet,
            "support_type": candidate.support_type,
            "confidence": confidence,
            "preview_url": annotation_preview_url(self.settings, candidate.workspace_id, candidate.annotation_id or ""),
            "source_open_url": annotation_source_open_url(self.settings, candidate.workspace_id, candidate.annotation_id or ""),
        }

    def search(
        self,
        *,
        workspace_id: str,
        material_ids: list[str],
        query_text: str,
        top_k: int,
        retrieval_mode: str,
        include_annotations: bool,
        min_extraction_quality: str,
    ) -> dict[str, Any]:
        workspace_excluded, excluded_materials, excluded_slides = self._exclusion_state(workspace_id)
        if workspace_excluded:
            return {"query_text": query_text, "evidence_items": []}
        materials = self._workspace_materials(workspace_id, material_ids)
        selected_material_ids = {material["material_id"] for material in materials}
        candidates = self._slide_candidates(
            materials=materials,
            min_extraction_quality=min_extraction_quality,
            excluded_materials=excluded_materials,
            excluded_slides=excluded_slides,
        )
        candidates.extend(
            self._annotation_candidates(
                workspace_id=workspace_id,
                include_annotations=include_annotations,
                selected_material_ids=selected_material_ids,
                excluded_materials=excluded_materials,
                excluded_slides=excluded_slides,
            )
        )
        ranked = self._rank_candidates(
            query_text=query_text,
            candidates=candidates,
            include_annotations=include_annotations,
            workspace_id=workspace_id,
            retrieval_mode=retrieval_mode,
        )
        selected = self._diversified_select(ranked, top_k, retrieval_mode)
        evidence_items = []
        for rank, candidate in enumerate(selected, start=1):
            citation = self._make_citation(candidate, query_text)
            evidence_items.append(
                {
                    "rank": rank,
                    "text": candidate.text,
                    "extraction_quality": candidate.extraction_quality,
                    "citation": citation,
                }
            )
        return {"query_text": query_text, "evidence_items": evidence_items}

    def bundle(
        self,
        *,
        workspace_id: str,
        material_ids: list[str],
        query_text: Optional[str],
        bundle_mode: str,
        token_budget: int,
        max_items: int,
        include_annotations: bool,
    ) -> dict[str, Any]:
        query_text = query_text or ""
        workspace_excluded, excluded_materials, excluded_slides = self._exclusion_state(workspace_id)
        bundle_id = stable_bundle_id(
            json_dumps(
                {
                    "workspace_id": workspace_id,
                    "material_ids": material_ids,
                    "query_text": query_text,
                    "bundle_mode": bundle_mode,
                    "token_budget": token_budget,
                    "max_items": max_items,
                    "include_annotations": include_annotations,
                }
            )
        )
        if workspace_excluded:
            return {
                "bundle_id": bundle_id,
                "workspace_id": workspace_id,
                "material_ids": material_ids,
                "query_text": query_text or None,
                "bundle_mode": bundle_mode,
                "items": [],
                "summary": {"total_items": 0, "total_slides": 0, "low_confidence_item_count": 0},
            }

        materials = self._workspace_materials(workspace_id, material_ids)
        selected_material_ids = {material["material_id"] for material in materials}
        if bundle_mode == "full_material":
            ordered_materials = []
            by_id = {m["material_id"]: m for m in materials}
            for material_id in unique_preserve_order(material_ids):
                if material_id in by_id:
                    ordered_materials.append(by_id[material_id])
            for material in materials:
                if material not in ordered_materials:
                    ordered_materials.append(material)
            candidates: list[Candidate] = []
            for material in ordered_materials:
                if material["material_id"] in excluded_materials:
                    continue
                for slide in self.repo.list_slides(material["material_id"]):
                    if slide["slide_id"] in excluded_slides:
                        continue
                    text = slide.get("extracted_text", "")
                    if not text.strip():
                        continue
                    candidates.append(
                        Candidate(
                            source_type="slide",
                            workspace_id=workspace_id,
                            material_id=material["material_id"],
                            slide_id=slide["slide_id"],
                            slide_number=int(slide["slide_number"]),
                            material_title=material["title"],
                            text=text,
                            extraction_quality=clamp_quality(slide["extraction_quality"]),
                            support_type="explicit",
                        )
                    )
            ordered = candidates
        else:
            candidates = self._slide_candidates(
                materials=materials,
                min_extraction_quality="low",
                excluded_materials=excluded_materials,
                excluded_slides=excluded_slides,
            )
            candidates.extend(
                self._annotation_candidates(
                    workspace_id=workspace_id,
                    include_annotations=include_annotations,
                    selected_material_ids=selected_material_ids,
                    excluded_materials=excluded_materials,
                    excluded_slides=excluded_slides,
                )
            )
            ranked = self._rank_candidates(
                query_text=query_text,
                candidates=candidates,
                include_annotations=include_annotations,
                workspace_id=workspace_id,
                retrieval_mode=bundle_mode,
            )
            if bundle_mode == "coverage":
                ordered = self._diversified_select(ranked, max_items or len(ranked), "coverage")
            else:
                ordered = ranked

        items = []
        total_tokens = 0
        for idx, candidate in enumerate(ordered, start=1):
            if max_items and len(items) >= max_items:
                break
            candidate_tokens = candidate.token_count
            if token_budget and items and total_tokens + candidate_tokens > token_budget:
                break
            if token_budget and not items and candidate_tokens > token_budget:
                # still return one item so callers are not left empty-handed
                pass
            citation = self._make_citation(candidate, query_text)
            items.append(
                {
                    "item_id": f"{bundle_id}:item:{idx}",
                    "material_id": citation["material_id"],
                    "material_title": citation["material_title"],
                    "slide_id": citation["slide_id"],
                    "slide_number": citation["slide_number"],
                    "text": candidate.text,
                    "extraction_quality": candidate.extraction_quality,
                    "citation": citation,
                }
            )
            total_tokens += candidate_tokens

        summary = {
            "total_items": len(items),
            "total_slides": len({(item["material_id"], item["slide_id"]) for item in items}),
            "low_confidence_item_count": sum(1 for item in items if item["citation"]["confidence"] == "low"),
        }
        return {
            "bundle_id": bundle_id,
            "workspace_id": workspace_id,
            "material_ids": material_ids,
            "query_text": query_text or None,
            "bundle_mode": bundle_mode,
            "items": items,
            "summary": summary,
        }

    def resolve_citations(self, citation_ids: Iterable[str]) -> list[dict[str, Any]]:
        requested = list(citation_ids)
        rows = self.repo.get_citations(requested)
        by_id = {row["citation_id"]: row for row in rows}
        resolved: list[dict[str, Any]] = []
        for citation_id in requested:
            row = by_id.get(citation_id)
            if not row:
                continue
            if row["source_type"] == "slide":
                material = self.repo.get_material(row["material_id"])
                slide = self.repo.get_slide(row["material_id"], row["slide_id"])
                if not material or not slide:
                    continue
                resolved.append(
                    {
                        "citation_id": citation_id,
                        "material_id": row["material_id"],
                        "material_title": material["title"],
                        "slide_id": row["slide_id"],
                        "slide_number": int(row["slide_number"]),
                        "snippet_text": row["snippet_text"],
                        "support_type": row["support_type"],
                        "confidence": row["confidence"],
                        "preview_url": slide_preview_url(self.settings, row["material_id"], row["slide_id"]),
                        "source_open_url": slide_source_open_url(self.settings, row["material_id"], row["slide_id"]),
                    }
                )
            elif row["source_type"] == "annotation":
                annotation = self.repo.get_annotation(row["workspace_id"], row["annotation_id"])
                if not annotation:
                    continue
                resolved.append(
                    {
                        "citation_id": citation_id,
                        "material_id": annotation_material_id(row["workspace_id"]),
                        "material_title": ANNOTATION_MATERIAL_TITLE,
                        "slide_id": annotation_slide_id(row["annotation_id"]),
                        "slide_number": int(annotation.get("virtual_slide_number") or row.get("slide_number") or 0),
                        "snippet_text": row["snippet_text"],
                        "support_type": row["support_type"],
                        "confidence": row["confidence"],
                        "preview_url": annotation_preview_url(self.settings, row["workspace_id"], row["annotation_id"]),
                        "source_open_url": annotation_source_open_url(self.settings, row["workspace_id"], row["annotation_id"]),
                    }
                )
        return resolved
