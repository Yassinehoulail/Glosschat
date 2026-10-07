"""Declarative query planner for the structured glossary dataset.

The planner returns operations, not a single intent. The backend remains the
authority that executes those operations over the complete JSON catalogue.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TargetEntities(BaseModel):
    kpi_names: list[str] = Field(default_factory=list)
    glossary_names: list[str] = Field(default_factory=list)
    families: list[str] = Field(default_factory=list)
    source_systems: list[str] = Field(default_factory=list)
    calculation_modes: list[str] = Field(default_factory=list)


class MatchingStrategy(BaseModel):
    kpi_name_match: Literal["none", "exact", "normalized_exact", "fuzzy", "semantic", "hybrid"] = "none"
    glossary_match: Literal["none", "exact", "normalized_exact", "fuzzy"] = "none"
    use_semantic_similarity: bool = False
    semantic_fields: list[str] = Field(default_factory=list)


class Scope(BaseModel):
    all_glossaries: bool = False
    exclude_glossaries: list[str] = Field(default_factory=list)
    exhaustive: bool = False


class SimilarityAnalysis(BaseModel):
    enabled: bool = False
    compare_name: bool = False
    compare_definition: bool = False
    compare_formula: bool = False
    compare_source_system: bool = False
    cluster_similar_kpis: bool = False


class Aggregation(BaseModel):
    enabled: bool = False
    group_by: list[str] = Field(default_factory=list)
    count: bool = False
    distinct: bool = False


class QualityChecks(BaseModel):
    enabled: bool = False
    missing_definition: bool = False
    missing_formula: bool = False
    duplicate_kpi_names: bool = False


class PostProcessing(BaseModel):
    compare_results: bool = False
    explain_differences: bool = False
    propose_standard_definition: bool = False
    propose_standard_formula: bool = False
    include_sources: bool = False


class QueryPlan(BaseModel):
    standalone_question: str
    target_entities: TargetEntities = Field(default_factory=TargetEntities)
    matching_strategy: MatchingStrategy = Field(default_factory=MatchingStrategy)
    scope: Scope = Field(default_factory=Scope)
    select_fields: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    similarity_analysis: SimilarityAnalysis = Field(default_factory=SimilarityAnalysis)
    aggregation: Aggregation = Field(default_factory=Aggregation)
    quality_checks: QualityChecks = Field(default_factory=QualityChecks)
    post_processing: PostProcessing = Field(default_factory=PostProcessing)


def deterministic_plan(question: str) -> QueryPlan:
    """Build a safe plan without requiring an LLM or vector retrieval."""
    q = question.casefold()
    # Glossary questions are exhaustive by default. Counting is handled
    # separately by Python and remains the only non-LLM operation.
    exhaustive = True
    if "combien" in q or "count" in q:
        return QueryPlan(standalone_question=question, scope=Scope(all_glossaries=True, exhaustive=True), operations=["collect_all_matches", "count_matches"], aggregation=Aggregation(enabled=True, count=True), post_processing=PostProcessing(include_sources=True))
    if "formule" in q and ("toutes" in q or "diff" in q):
        return QueryPlan(standalone_question=question, scope=Scope(all_glossaries=True, exhaustive=True), operations=["collect_all_matches", "normalize_formulas", "group_equivalent_formulas", "compare_formula_groups"], post_processing=PostProcessing(compare_results=True, explain_differences=True, propose_standard_formula=True, include_sources=True))
    if any(x in q for x in ("définition", "definition")):
        return QueryPlan(standalone_question=question, scope=Scope(all_glossaries=True, exhaustive=True), operations=["collect_all_matches"], post_processing=PostProcessing(include_sources=True))
    if any(x in q for x in ("même indicateur", "similarité", "similitude")):
        return QueryPlan(standalone_question=question, matching_strategy=MatchingStrategy(kpi_name_match="hybrid", use_semantic_similarity=True, semantic_fields=["Nom KPI", "Définition", "Formule de calcul"]), scope=Scope(all_glossaries=True, exhaustive=True), operations=["load_all_kpis", "normalize_kpi_names", "normalize_formulas", "compute_name_similarity", "compute_definition_similarity", "compute_formula_similarity", "cluster_similar_kpis"], similarity_analysis=SimilarityAnalysis(enabled=True, compare_name=True, compare_definition=True, compare_formula=True, cluster_similar_kpis=True), post_processing=PostProcessing(compare_results=True, explain_differences=True, include_sources=True))
    return QueryPlan(standalone_question=question, scope=Scope(all_glossaries=True, exhaustive=True), operations=["collect_all_matches"])
