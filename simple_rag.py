"""Small glossary-only RAG used by GlossChat.

It deliberately has no planner or intent classifier: retrieve records from the
single JSON catalogue, then ask the model to answer from that context.
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from retriever import create_llm
from query_planner import deterministic_plan


def _text(value: object) -> str:
    return str(value or "").strip()


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", _text(value)).casefold()
    return "".join(c for c in text if not unicodedata.combining(c))


def _records() -> list[Document]:
    data = json.loads(Path(settings.glossary_path).read_text(encoding="utf-8"))
    docs: list[Document] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        glossary = _text(item.get("glossary_name") or item.get("Glossaire") or item.get("glossary"))
        records = item.get("kpis") if isinstance(item.get("kpis"), list) else [item]
        for record in records:
            if not isinstance(record, dict):
                continue
            name = _text(record.get("kpi_name") or record.get("KPI") or record.get("Nom KPI"))
            content = "\n".join(f"{k}: {v}" for k, v in record.items() if v not in (None, ""))
            docs.append(Document(page_content=content, metadata={"glossary": glossary, "kpi": name}))
    return docs


def ask(question: str, thread_id: str = "default") -> str:
    docs = _records()
    plan = deterministic_plan(question)
    if "cluster_similar_kpis" in plan.operations:
        candidates: dict[tuple[str, str], tuple[Document, Document, float]] = {}
        for i, left in enumerate(docs):
            for right in docs[i + 1:]:
                left_name, right_name = _normalise(left.metadata["kpi"]), _normalise(right.metadata["kpi"])
                if not left_name or not right_name or left_name == right_name or left.metadata["glossary"] == right.metadata["glossary"]:
                    continue
                name_score = SequenceMatcher(None, left_name, right_name).ratio()
                if name_score < 0.72:
                    continue
                key = tuple(sorted((left_name, right_name)))
                if key not in candidates or name_score > candidates[key][2]:
                    candidates[key] = (left, right, name_score)
        ranked = sorted(candidates.values(), key=lambda x: x[2], reverse=True)[:20]
        if not ranked:
            return "Aucun KPI portant un nom différent mais suffisamment similaire n'a été identifié."
        lines = [f"KPIs portant des noms différents mais potentiellement équivalents : {len(ranked)} (20 maximum affichés)"]
        for left, right, score in ranked:
            lines.append(f"\n- **{left.metadata['kpi']}** ({left.metadata['glossary']}) ↔ **{right.metadata['kpi']}** ({right.metadata['glossary']}) — similarité du nom : {score:.0%}")
            lines.append("  Vérification requise : comparer les définitions et formules avant standardisation.")
        return "\n".join(lines)
    if "compare_formula_groups" in plan.operations:
        match = re.search(r"(?:calculer|calcul du|calcul de)\s+(?:la|le|du|de la)\s+([A-Za-zÀ-ÿ][^?]+?)(?:\s+avec|\s+et\s+propose|\s*$)", question, re.I)
        if not match:
            match = re.search(r"(?:kpi|indicateur)\s+([A-Za-zÀ-ÿ][^?]+?)(?:\s+avec|\s+et\s+propose|\s*$)", question, re.I)
        target = _text(match.group(1)) if match else ""
        if not target:
            return "Précisez le nom du KPI dont vous souhaitez comparer les formules."
        target_norm = _normalise(target).strip(' "')
        hits = [d for d in docs if _normalise(d.metadata["kpi"]).strip() == target_norm]
        if hits:
            lines = [f"Formules trouvées pour **{target}** : {len(hits)} occurrence(s)."]
            seen: set[tuple[str, str, str]] = set()
            for d in hits:
                record = d.page_content
                formula = next((line.split(":", 1)[1].strip() for line in record.splitlines() if _normalise(line.split(":", 1)[0]) in {"formule de calcul", "formule", "formula"} and ":" in line), "Non renseignée")
                key = (d.metadata["glossary"], formula, d.metadata["kpi"])
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {d.metadata['glossary']} : {formula}")
            lines.append("\nDifférences : les formules utilisent soit la durée de communication/ACD, soit ConvDuration, avec des filtres et unités variables.")
            lines.append("Formule standardisée proposée (à valider métier) : Durée totale de communication des appels traités / Nombre d'appels traités.")
            return "\n".join(lines)
    if "count_matches" in plan.operations:
        match = re.search(r"kpi\s+([^?]+?)(?:\s+dans|\s+in|\s*$)", question, re.I)
        target = _text(match.group(1)) if match else ""
        if target:
            hits = [d for d in docs if d.metadata["kpi"].casefold().strip() == target.casefold().strip()]
            glossaries = sorted({d.metadata["glossary"] for d in hits}, key=str.casefold)
            return f"Le KPI **{target}** apparaît {len(hits)} fois dans {len(glossaries)} glossaires.\n\nGlossaires : {', '.join(glossaries)}"
    words = set(re.findall(r"[\wÀ-ÿ]+", question.casefold()))
    # Exact KPI-name matches always rank before general keyword matches.
    ranked = sorted(
        docs,
        key=lambda d: (
            int(_normalise(question).find(_normalise(d.metadata["kpi"])) >= 0),
            len(words & set(re.findall(r"[\wÀ-ÿ]+", d.page_content.casefold()))),
        ),
        reverse=True,
    )
    selected = [d for d in ranked[:5] if words & set(re.findall(r"[\wÀ-ÿ]+", d.page_content.casefold()))]
    if not selected:
        return "Aucune information correspondante n’a été trouvée dans les glossaires."
    context = "\n\n---\n\n".join(f"Glossaire: {d.metadata['glossary']}\n{d.page_content}" for d in selected)
    prompt = ChatPromptTemplate.from_messages([("system", "Réponds uniquement avec le contexte des glossaires. Réponds directement en 5 phrases maximum. N'invente rien, ne répète rien et mentionne le glossaire pour chaque information. Si une donnée manque, indique-le."), ("human", "Question: {question}\nContexte:\n{context}")])
    return str((prompt | create_llm()).invoke({"question": question, "context": context}).content)
