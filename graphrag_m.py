#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import json
import logging
import warnings
from collections import deque
from typing import Any, List, Optional, Union

from pydantic import ValidationError

from neo4j_graphrag.exceptions import (
    RagInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.generation.types import RagInitModel, RagResultModel, RagSearchModel
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import LLMMessage, RetrieverResult
from neo4j_graphrag.utils.logging import prettify

logger = logging.getLogger(__name__)


class GraphRAG:
    """Performs a GraphRAG search using a specific retriever
    and LLM.

    Example:

    .. code-block:: python

      import neo4j
      from neo4j_graphrag.retrievers import VectorRetriever
      from neo4j_graphrag.llm.openai_llm import OpenAILLM
      from neo4j_graphrag.generation import GraphRAG

      driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

      retriever = VectorRetriever(driver, "vector-index-name", custom_embedder)
      llm = OpenAILLM()
      graph_rag = GraphRAG(retriever, llm)
      graph_rag.search(query_text="Find me a book about Fremen")

    Args:
        retriever (Retriever): The retriever used to find relevant context to pass to the LLM.
        llm (LLMInterface): The LLM used to generate the answer.
        prompt_template (RagTemplate): The prompt template that will be formatted with context and user question and passed to the LLM.

    Raises:
        RagInitializationError: If validation of the input arguments fail.
    """

    COLOR_RESET = "\033[0m"
    COLOR_CODES = {
        "stage": "\033[95m",
        "query": "\033[94m",
        "node": "\033[96m",
        "decision": "\033[93m",
        "evidence": "\033[92m",
        "warning": "\033[91m",
    }

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        try:
            validated_data = RagInitModel(
                retriever=retriever,
                llm=llm,
                prompt_template=prompt_template,
            )
        except ValidationError as e:
            raise RagInitializationError(e.errors())
        self.retriever = validated_data.retriever
        self.llm = validated_data.llm
        self.prompt_template = validated_data.prompt_template
        self.driver = getattr(self.retriever, "driver", None)
        
    def search_nodes_relationships_only(
        self,
        query_text: str = "",
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: Optional[bool] = None,
        response_fallback: Optional[str] = None,
    ) -> RagResultModel:
        """Graph-first search that keeps the LLM focused on nodes and relationships."""

        if return_context is None:
            warnings.warn(
                "The default value of 'return_context' will change from 'False' to 'True' in a future version.",
                DeprecationWarning,
            )
            return_context = False

        retriever_config = retriever_config or {}
        strategy_config = {}
        if isinstance(retriever_config, dict) and "graph_strategy" in retriever_config:
            strategy_config = retriever_config.get("graph_strategy") or {}
            retriever_config = {
                key: value for key, value in retriever_config.items() if key != "graph_strategy"
            }

        try:
            validated_data = RagSearchModel(
                query_text=query_text,
                examples=examples,
                retriever_config=retriever_config,
                return_context=return_context,
                response_fallback=response_fallback,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())

        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages

        query = self._build_query(validated_data.query_text, message_history)
        retriever_result: RetrieverResult = self.retriever.search(
            query_text=query, **validated_data.retriever_config
        )

        if len(retriever_result.items) == 0 and response_fallback is not None:
            answer = response_fallback
        else:
            max_iterations = int(strategy_config.get("max_iterations", 3))
            thought_prefix = strategy_config.get("thought_prefix", "Iteration")

            formatted_snippets: list[str] = []
            top_node_hint: Optional[str] = None

            for idx, item in enumerate(retriever_result.items, start=1):
                metadata = getattr(item, "metadata", {}) or {}
                node_name = metadata.get("name") or metadata.get("id") or metadata.get("title")
                if top_node_hint is None and node_name:
                    top_node_hint = str(node_name)

                labels = metadata.get("labels") or metadata.get("label")
                relationships = (
                    metadata.get("relationships")
                    or metadata.get("relationship")
                    or metadata.get("edges")
                )

                def _to_text(value: Any) -> str:
                    if value is None:
                        return ""
                    if isinstance(value, (list, tuple, set)):
                        return ", ".join(str(v) for v in value)
                    return str(value)

                snippet_header = [
                    f"Snippet {idx}",
                    f"node={node_name or 'unknown'}",
                ]
                if labels:
                    snippet_header.append(f"labels={_to_text(labels)}")
                if relationships:
                    snippet_header.append(f"relationships={_to_text(relationships)}")
                snippet_body = item.content.strip()
                formatted_snippets.append(" | ".join(snippet_header) + f"\n{snippet_body}")

            if not formatted_snippets:
                formatted_snippets.append("No graph snippets were retrieved.")

            context_block = "\n\n".join(formatted_snippets)
            examples_block = (
                f"\n\nExamples:\n{validated_data.examples}" if validated_data.examples else ""
            )
            top_node_text = top_node_hint or "unknown"

            reasoning_prompt = f"""
You are a graph traversal assistant. Only reason about nodes and relationships explicitly grounded in the retrieved snippets.

User question: "{query_text}"
Top candidate node to anchor the exploration: "{top_node_text}".
You may perform up to {max_iterations} reasoning hops. For each hop output a bullet beginning with "{thought_prefix} <index>" that explains which relationship you investigate next and why.
Stop early if enough evidence exists to answer the user. When finished provide a concise answer that cites the specific nodes/relationships you relied on. If the context is insufficient, explicitly say so.

Graph snippets:
{context_block}
{examples_block}
"""

            logger.debug("GraphRAG nodes+relationships prompt=%s", reasoning_prompt)
            llm_response = self.llm.invoke(
                input=reasoning_prompt,
                message_history=message_history,
                system_instruction="You analyze property graphs by focusing on node-level evidence and relationship chains.",
            )
            answer = llm_response.content

        result: dict[str, Any] = {
            "answer": answer,
        }
        if return_context:
            result["retriever_result"] = retriever_result
            result["retriever_query"] = query
        return RagResultModel(**result)

    def search(
        self,
        query_text: str = "",
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: Optional[bool] = None,
        response_fallback: Optional[str] = None,
    ) -> RagResultModel:
        """
        .. warning::
            The default value of 'return_context' will change from 'False' to 'True' in a future version.


        This method performs a full RAG search:
            1. Retrieval: context retrieval
            2. Augmentation: prompt formatting
            3. Generation: answer generation with LLM


        Args:
            query_text (str): The user question.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            examples (str): Examples added to the LLM prompt.
            retriever_config (Optional[dict]): Parameters passed to the retriever.
                search method; e.g.: top_k
            return_context (bool): Whether to append the retriever result to the final result (default: False).
            response_fallback (Optional[str]): If not null, will return this message instead of calling the LLM if context comes back empty.

        Returns:
            RagResultModel: The LLM-generated answer.

        """
        if return_context is None:
            warnings.warn(
                "The default value of 'return_context' will change from 'False' to 'True' in a future version.",
                DeprecationWarning,
            )
            return_context = False
        try:
            validated_data = RagSearchModel(
                query_text=query_text,
                examples=examples,
                retriever_config=retriever_config or {},
                return_context=return_context,
                response_fallback=response_fallback,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        query = self._build_query(validated_data.query_text, message_history)
        retriever_result: RetrieverResult = self.retriever.search(
            query_text=query, **validated_data.retriever_config
        )
        if len(retriever_result.items) == 0 and response_fallback is not None:
            answer = response_fallback
        else:
            context = "\n".join(item.content for item in retriever_result.items)
            prompt = self.prompt_template.format(
                query_text=query_text, context=context, examples=validated_data.examples
            )
            print(f"RAG: retriever_result={prettify(retriever_result)}")
            print(f"RAG: prompt={prompt}")
            llm_response = self.llm.invoke(
                prompt,
                message_history,
                system_instruction=self.prompt_template.system_instructions,
            )
            answer = llm_response.content
        result: dict[str, Any] = {
            "answer": answer,
        }
        if return_context:
            result["retriever_result"] = retriever_result
            result["retriever_query"] = query
        return RagResultModel(**result)

    def search_recursive_graph(
        self,
        query_text: str = "",
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: Optional[bool] = None,
        response_fallback: Optional[str] = None,
    ) -> RagResultModel:
        """LLM-guided recursive node exploration with explicit relationship hops."""

        if return_context is None:
            warnings.warn(
                "The default value of 'return_context' will change from 'False' to 'True' in a future version.",
                DeprecationWarning,
            )
            return_context = False

        retriever_config = retriever_config or {}
        recursive_cfg = retriever_config.pop("recursive_strategy", {}) or {}

        max_depth = int(recursive_cfg.get("max_depth", 3))
        max_followups = int(recursive_cfg.get("max_followups", 3))
        relationship_limit = int(recursive_cfg.get("relationship_limit", 0))

        try:
            validated_data = RagSearchModel(
                query_text=query_text,
                examples=examples,
                retriever_config=retriever_config,
                return_context=return_context,
                response_fallback=response_fallback,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())

        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages

        def _structured_llm_call(prompt: str, schema_hint: str) -> dict[str, Any]:
            try:
                response = self.llm.invoke(
                    input=f"{prompt}\nReturn JSON with shape: {schema_hint}",
                    message_history=message_history,
                    system_instruction="You must return valid JSON only.",
                )
                return json.loads(response.content)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Recursive search parsing failed: %s", exc)
                self._color_log("warning", f"JSON parsing failed: {exc}")
                return {}

        def _identify_subjects(question: str) -> tuple[str, List[str]]:
            prompt = f"""
Rewrite the user question for graph lookup and list the most relevant subject node names.
Respond with JSON: {{"focus_question": "...", "subjects": ["..."]}}

Question: "{question}"
"""
            data = _structured_llm_call(
                prompt,
                '{"focus_question": "...", "subjects": ["..."]}',
            )
            focus = data.get("focus_question") if isinstance(data, dict) else None
            subjects = data.get("subjects", []) if isinstance(data, dict) else []
            subjects = [s.strip() for s in subjects if isinstance(s, str) and s.strip()]
            if not subjects:
                subjects = [question]
            return (focus.strip() if isinstance(focus, str) and focus.strip() else question, subjects)

        def _fetch_relationships_for_node(node_name: str, limit: int | None) -> List[dict[str, Any]]:
            if not self.driver:
                return []
            cypher_lines = [
                "MATCH (n)",
                "WHERE toLower(n.name) = toLower($name)",
                "WITH n LIMIT 1",
                "MATCH (n)-[r]-(m)",
                "WHERE NOT (type(r) = 'FROM_CHUNK' AND (m.name IS NULL OR trim(m.name) = ''))",
                "RETURN n.name AS node,",
                "       labels(n) AS node_labels,",
                "       type(r) AS rel_type,",
                "       CASE WHEN id(startNode(r)) = id(n) THEN 'OUT' ELSE 'IN' END AS direction,",
                "       m.name AS neighbor,",
                "       labels(m) AS neighbor_labels",
            ]
            if limit is not None:
                cypher_lines.append("LIMIT $limit")
            cypher = "\n".join(cypher_lines)
            try:
                with self.driver.session() as session:
                    params = {"name": node_name}
                    if limit is not None:
                        params["limit"] = limit
                    records = session.run(cypher, **params)
                    return [record.data() for record in records]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Relationship fetch failed for %s: %s", node_name, exc)
                self._color_log("warning", f"Failed to fetch relationships for {node_name}: {exc}")
                return []

        vector_top_k = int(
            recursive_cfg.get(
                "vector_top_k",
                validated_data.retriever_config.get("top_k", 5)
                if isinstance(validated_data.retriever_config, dict)
                else 5,
            )
        )
        vector_max_queries = int(recursive_cfg.get("vector_max_queries", 2))
        vector_auto_fallback = bool(recursive_cfg.get("vector_auto_fallback", True))
        fallback_queries_cfg = recursive_cfg.get("vector_fallback_queries")
        fallback_queries: List[str] = []
        if isinstance(fallback_queries_cfg, str):
            candidate = fallback_queries_cfg.strip()
            if candidate:
                fallback_queries.append(candidate)
        elif isinstance(fallback_queries_cfg, (list, tuple)):
            for candidate in fallback_queries_cfg:
                if isinstance(candidate, str):
                    cleaned = candidate.strip()
                    if cleaned:
                        fallback_queries.append(cleaned)

        vector_calls_made = 0

        focus_question, seed_subjects = _identify_subjects(validated_data.query_text)
        if not fallback_queries:
            fallback_queries = [focus_question, validated_data.query_text]
        fallback_queries = [q.strip() for q in fallback_queries if isinstance(q, str) and q.strip()]
        self._color_log("stage", f"Focus question: {focus_question}")
        self._color_log("stage", f"Seed subjects: {seed_subjects}")

        evidence: List[str] = []
        reasoning_trace: List[str] = []
        provisional_answer: Optional[str] = None

        queue: deque[tuple[str, int]] = deque((subject, 1) for subject in seed_subjects)
        visited: set[str] = set()

        def _vector_retrieve(query_text: str) -> list[str]:
            nonlocal vector_calls_made
            if vector_max_queries <= 0:
                return []
            if vector_calls_made >= vector_max_queries:
                self._color_log(
                    "warning",
                    f"Vector retrieval budget exhausted; skipping query '{query_text}'",
                )
                return []
            if not isinstance(validated_data.retriever_config, dict):
                config = {}
            else:
                config = dict(validated_data.retriever_config)
            if vector_top_k > 0:
                config["top_k"] = vector_top_k
            vector_calls_made += 1
            try:
                result = self.retriever.search(query_text=query_text, **config)
            except Exception as exc:  # noqa: BLE001
                self._color_log("warning", f"Vector retrieval failed for '{query_text}': {exc}")
                return []
            snippets = []
            for item in result.items:
                snippet = item.content.strip()
                if snippet:
                    snippets.append(snippet)
            if snippets:
                self._color_log(
                    "query",
                    f"Vector retrieval for '{query_text}' returned {len(snippets)} snippet(s)",
                )
            return snippets

        while queue:
            node_name, depth = queue.popleft()
            node_key = node_name.lower()
            if node_key in visited or depth > max_depth:
                continue
            visited.add(node_key)
            self._color_log("node", f"[Depth {depth}] Exploring node '{node_name}'")

            rel_limit = relationship_limit if relationship_limit > 0 else None
            relationships = _fetch_relationships_for_node(node_name, rel_limit)
            if not relationships:
                self._color_log("node", f"No relationships found for {node_name}")
                reasoning_trace.append(f"Depth {depth}: No relationships for {node_name}")
                continue

            formatted_relationships: List[str] = []
            for rel in relationships:
                neighbor = rel.get("neighbor") or "unknown"
                rel_type = rel.get("rel_type") or "UNKNOWN"
                direction = rel.get("direction") or "OUT"

                start_node = node_name
                end_node = neighbor
                if direction == "IN":
                    start_node, end_node = neighbor, node_name

                formatted = f"{start_node} -[{rel_type}]-> {end_node}"
                formatted_relationships.append(formatted)
                evidence.append(
                    f"{start_node} -[{rel_type} ({direction})]-> {end_node}; neighbor_labels={rel.get('neighbor_labels')}"
                )

            relationships_block = "\n".join(formatted_relationships)
            self._color_log("evidence", relationships_block)

            doc_prompt = f"""
Original question: "{validated_data.query_text}"
Relationships currently known for node "{node_name}":
{relationships_block}

Decide if additional document retrieval is needed. Respond with JSON:
{{
  "needs_retrieval": bool,
  "reason": "...",
  "queries": ["embedding search text", ...]
}}

If the relationships do not explicitly answer the question or any uncertainty remains, you MUST set "needs_retrieval" to true and propose targeted embedding queries.
"""

            doc_decision = _structured_llm_call(
                doc_prompt,
                '{"needs_retrieval": false, "reason": "...", "queries": ["..."]}',
            )
            if isinstance(doc_decision, dict):
                self._color_log("decision", f"Doc retrieval decision: {doc_decision}")
            if isinstance(doc_decision, dict) and doc_decision.get("needs_retrieval"):
                queries = doc_decision.get("queries", []) or []
                for doc_query in queries:
                    if vector_calls_made >= vector_max_queries:
                        break
                    if isinstance(doc_query, str) and doc_query.strip():
                        snippets = _vector_retrieve(doc_query.strip())
                        for snippet in snippets:
                            evidence.append(f"[Vector:{doc_query.strip()}] {snippet}")

            decision_prompt = f"""
Original question: "{validated_data.query_text}"
Current node: "{node_name}"
Depth {depth}/{max_depth}

Relationships:\n{relationships_block}

Respond with JSON: {{
  "reason": "...",
  "answer_hint": "optional answer fragment grounded in data",
  "follow_up_nodes": ["neighbor name", ...]
}}
You should consider adding more follow-up nodes to explore if you think it could improve the answer.
"""

            print("Decision prompt:", decision_prompt)

            decision = _structured_llm_call(
                decision_prompt,
                '"reason": "...", "answer_hint": "...", "follow_up_nodes": ["..."]}',
            )

            stop = bool(decision.get("stop")) if isinstance(decision, dict) else False
            reason = decision.get("reason") if isinstance(decision, dict) else ""
            if isinstance(decision, dict) and decision.get("answer_hint"):
                provisional_answer = decision.get("answer_hint")
            followups = decision.get("follow_up_nodes", []) if isinstance(decision, dict) else []
            self._color_log(
                "decision",
                f"Decision at depth {depth}: stop={stop}, reason={reason}, followups={followups}",
            )
            reasoning_trace.append(
                f"Depth {depth}: stop={stop}, reason={reason}, followups={followups}"
            )

            if stop:
                break

            for follow in followups[:max_followups]:
                if isinstance(follow, str) and follow.strip():
                    queue.append((follow.strip(), depth + 1))

        if vector_auto_fallback and vector_calls_made == 0 and vector_max_queries > 0:
            self._color_log(
                "decision",
                "Vector fallback triggered because no document lookups were executed during recursion.",
            )
            for candidate in fallback_queries:
                if vector_calls_made >= vector_max_queries:
                    break
                snippets = _vector_retrieve(candidate)
                for snippet in snippets:
                    evidence.append(f"[Vector:{candidate}] {snippet}")
                if snippets:
                    break

        if not evidence and response_fallback is not None:
            answer = response_fallback
        else:
            evidence_text = "\n".join(evidence) if evidence else "No structured evidence captured."
            reasoning_text = "\n".join(reasoning_trace)
            final_prompt = f"""
Answer the user's question using only the provided graph evidence.

User question: "{validated_data.query_text}"
Internal reasoning trace:
{reasoning_text}

Graph evidence:
{evidence_text}

Provisional answer (if any): {provisional_answer or 'None'}

Provide a concise, factual answer. State explicitly if the answer cannot be determined.
{f"Examples:\n{validated_data.examples}" if validated_data.examples else ""}
"""

            llm_response = self.llm.invoke(
                input=final_prompt,
                message_history=message_history,
                system_instruction="Only use supplied evidence to answer.",
            )
            answer = llm_response.content

        return RagResultModel(answer=answer)

    def _build_query(
        self,
        query_text: str,
        message_history: Optional[List[LLMMessage]] = None,
    ) -> str:
        summary_system_message = "You are a summarization assistant. Summarize the given text in no more than 300 words."
        if message_history:
            summarization_prompt = self._chat_summary_prompt(
                message_history=message_history
            )
            summary = self.llm.invoke(
                input=summarization_prompt,
                system_instruction=summary_system_message,
            ).content
            return self.conversation_prompt(summary=summary, current_query=query_text)
        return query_text

    def _chat_summary_prompt(self, message_history: List[LLMMessage]) -> str:
        message_list = [
            f"{message['role']}: {message['content']}" for message in message_history
        ]
        history = "\n".join(message_list)
        return f"""
Summarize the message history:

{history}
"""

    def conversation_prompt(self, summary: str, current_query: str) -> str:
        return f"""
Message Summary:
{summary}

Current Query:
{current_query}
"""

    def _color_log(self, category: str, message: str) -> None:
        color = self.COLOR_CODES.get(category, "")
        reset = self.COLOR_RESET if color else ""
        print(f"{color}{message}{reset}")
