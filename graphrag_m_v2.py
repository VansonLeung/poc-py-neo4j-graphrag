from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, List, Optional, Sequence, Tuple, Union

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.exceptions import (
    RagInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.generation.types import RagInitModel, RagResultModel, RagSearchModel
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import ToolCallResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.tool import (
    IntegerParameter,
    ObjectParameter,
    StringParameter,
    Tool,
)
from neo4j_graphrag.types import LLMMessage, RawSearchResult, RetrieverResult

logger = logging.getLogger(__name__)


class GraphAdapter(ABC):
    """Abstraction over concrete graph stores for node and relationship lookups."""

    @abstractmethod
    def get_node(
        self,
        node_name: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Return a single node by name with optional property filters."""

    @abstractmethod
    def search_nodes(
        self,
        node_name: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[dict[str, Any]]:
        """Search nodes by fuzzy name match and/or filters."""

    @abstractmethod
    def get_relationships(
        self,
        node_name: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 25,
    ) -> List[dict[str, Any]]:
        """Fetch relationships for a node, applying optional filters."""


class Neo4jGraphAdapter(GraphAdapter):
    """Default implementation backed by a Neo4j driver."""

    def __init__(self, driver: neo4j.Driver, database: Optional[str] = None):
        self.driver = driver
        self.database = database

    def get_node(
        self,
        node_name: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if not node_name:
            raise ValueError("node_name is required for node lookup")
        where_clause, params = self._build_filter_clause(filters, alias="n")
        params["node_name"] = node_name
        query = [
            "MATCH (n)",
            "WHERE toLower(n.name) = toLower($node_name)",
        ]
        if where_clause:
            query.append(f"AND {where_clause}")
        query.append("RETURN n.name AS name, labels(n) AS labels, properties(n) AS properties")
        query.append("LIMIT 1")
        record = self._run_single("\n".join(query), params)
        return record

    def search_nodes(
        self,
        node_name: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[dict[str, Any]]:
        limit = limit if limit and limit > 0 else 10
        where_parts = []
        params: dict[str, Any] = {"limit": limit}
        if node_name:
            where_parts.append("toLower(n.name) CONTAINS toLower($search_name)")
            params["search_name"] = node_name
        filter_clause, filter_params = self._build_filter_clause(filters, alias="n")
        if filter_clause:
            where_parts.append(filter_clause)
            params.update(filter_params)
        query = ["MATCH (n)"]
        if where_parts:
            query.append("WHERE " + " AND ".join(where_parts))
        query.append("RETURN n.name AS name, labels(n) AS labels, properties(n) AS properties")
        query.append("LIMIT $limit")
        return self._run_many("\n".join(query), params)

    def get_relationships(
        self,
        node_name: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 25,
    ) -> List[dict[str, Any]]:
        if not node_name:
            raise ValueError("node_name is required to fetch relationships")
        limit = limit if limit and limit > 0 else 25
        params: dict[str, Any] = {"node_name": node_name, "limit": limit}
        rel_filters = filters or {}
        rel_type = rel_filters.get("relationship_type")
        neighbor_name = rel_filters.get("neighbor_name")
        direction = rel_filters.get("direction")
        clauses = ["toLower(n.name) = toLower($node_name)"]
        clauses.append(
            "NOT (type(r) = 'FROM_CHUNK' AND (m.name IS NULL OR trim(m.name) = ''))"
        )
        if rel_type:
            clauses.append("type(r) = $rel_type")
            params["rel_type"] = rel_type
        if neighbor_name:
            clauses.append("toLower(m.name) CONTAINS toLower($neighbor_name)")
            params["neighbor_name"] = neighbor_name
        query = [
            "MATCH (n)-[r]-(m)",
            "WHERE " + " AND ".join(clauses),
            "WITH n, r, m",
            "RETURN n.name AS source, type(r) AS rel_type,",
            "       CASE WHEN elementId(startNode(r)) = elementId(n) THEN 'OUT' ELSE 'IN' END AS direction,",
            "       m.name AS target, labels(m) AS target_labels",
            "LIMIT $limit",
        ]
        records = self._run_many("\n".join(query), params)
        if direction in {"OUT", "IN"}:
            filtered = [rec for rec in records if rec.get("direction") == direction]
            return filtered
        return records

    def _build_filter_clause(
        self,
        filters: Optional[dict[str, Any]],
        alias: str,
    ) -> Tuple[str, dict[str, Any]]:
        if not filters:
            return "", {}
        clauses: List[str] = []
        params: dict[str, Any] = {}
        for idx, (key, value) in enumerate(filters.items()):
            if value is None:
                continue
            param_key = f"{alias}_filter_{idx}"
            clauses.append(f"{alias}.{key} = ${param_key}")
            params[param_key] = value
        return " AND ".join(clauses), params

    def _run_single(self, query: str, params: dict[str, Any]) -> Optional[dict[str, Any]]:
        records = self._run_many(query, params)
        return records[0] if records else None

    def _run_many(self, query: str, params: dict[str, Any]) -> List[dict[str, Any]]:
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                return [record.data() for record in result]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph query failed: %s", exc)
            raise


class GraphRAGv2:
    """Tool-call driven GraphRAG variant with graph + vector utilities."""

    TOOL_RESULT_PREFIX = "Tool Result"
    COLOR_RESET = "\033[0m"
    COLOR_CODES = {
        "stage": "\033[95m",
        "prompt": "\033[94m",
        "tool": "\033[96m",
        "result": "\033[92m",
        "warning": "\033[91m",
        "fallback": "\033[93m",
    }

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        graph_adapter: Optional[GraphAdapter] = None,
        prompt_template: RagTemplate = RagTemplate(),
        max_tool_turns: int = 15,
        vector_call_limit: int = 15,
        controller_instruction: Optional[str] = None,
    ):
        try:
            validated_data = RagInitModel(
                retriever=retriever,
                llm=llm,
                prompt_template=prompt_template,
            )
        except ValidationError as exc:
            raise RagInitializationError(exc.errors())
        self.retriever = validated_data.retriever
        self.llm = validated_data.llm
        self.prompt_template = validated_data.prompt_template
        self.max_tool_turns = max(1, max_tool_turns)
        self.vector_call_limit = max(1, vector_call_limit)
        self.controller_instruction = (
            controller_instruction
            or "You are a graph-first assistant. Decide whether to invoke a tool or return a final grounded answer."
        )
        self.graph_adapter = graph_adapter or self._build_default_graph_adapter()
        if not self.graph_adapter:
            raise RagInitializationError("Graph adapter is required for GraphRAGv2")
        self.tools = self._build_tools()

    def search_with_tool_calls(
        self,
        query_text: str = "",
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: Optional[bool] = None,
        response_fallback: Optional[str] = None,
    ) -> RagResultModel:
        if return_context is None:
            return_context = False
        retriever_config = retriever_config or {}
        try:
            validated_data = RagSearchModel(
                query_text=query_text,
                examples=examples,
                retriever_config=retriever_config,
                return_context=return_context,
                response_fallback=response_fallback,
            )
        except ValidationError as exc:
            raise SearchValidationError(exc.errors())

        history: List[LLMMessage]
        if isinstance(message_history, MessageHistory):
            history = list(message_history.messages)
        else:
            history = list(message_history or [])
        if query_text:
            history.append({"role": "user", "content": query_text})
            self._color_log("stage", f"User question queued: {query_text}")

        tool_observations: List[str] = []
        self._vector_calls_used = 0

        answer: Optional[str] = None
        for turn_index in range(1, self.max_tool_turns + 1):
            controller_prompt = self._build_controller_prompt(
                validated_data.query_text,
                turn_index,
            )
            self._color_log(
                "prompt",
                f"Controller prompt (turn {turn_index}): {controller_prompt.strip()}",
            )

            if turn_index == 1:
                self._color_log(
                    "stage",
                    "First turn: initial vector_search tool call.",
                )
                response = ToolCallResponse(
                  tool_calls=[
                    {
                        "index": 0,
                        "name": "vector_search",
                        "arguments": {
                            "query_text": validated_data.query_text,
                            "top_k": min(5, self.vector_call_limit - self._vector_calls_used),
                        },
                    }
                  ]
                )
            else:
                response = self.llm.invoke_with_tools(
                    input=controller_prompt,
                    tools=self.tools,
                    message_history=history,
                    system_instruction=self.controller_instruction,
                )
                if not response.tool_calls:
                    self._color_log("result", "LLM responded without tool calls; finalizing answer.")
                    answer = response.content or ""
                    break

            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                arguments = tool_call.arguments or {}
                self._color_log(
                    "tool",
                    f"Invoking tool '{tool_name}' with args={json.dumps(arguments, ensure_ascii=False)}",
                )
                execution_payload = self._dispatch_tool(tool_name, arguments)
                serialized = self._serialize_tool_result(tool_name, execution_payload)
                history.append({"role": "user", "content": serialized})
                tool_observations.append(serialized)
                self._color_log("result", serialized)

        if answer is None:
            self._color_log(
                "fallback",
                "Maximum tool turns reached without direct answer; entering fallback summarization.",
            )
            fallback_prompt = self._build_fallback_prompt(
                validated_data.query_text,
                tool_observations,
                validated_data.examples,
            )
            self._color_log("prompt", f"Fallback prompt: {fallback_prompt.strip()}")
            llm_response = self.llm.invoke(
                input=fallback_prompt,
                message_history=history,
                system_instruction="Summarize tool findings into a final answer grounded in prior tool outputs.",
            )
            answer = llm_response.content or response_fallback or "No answer produced."
            if not llm_response.content:
                self._color_log("warning", "Fallback LLM produced empty content; using response_fallback/text.")

        result: dict[str, Any] = {"answer": answer}
        if return_context:
            result["tool_messages"] = tool_observations
        return RagResultModel(**result)

    def _build_controller_prompt(self, query_text: str, turn_index: int) -> str:
        remaining = max(self.max_tool_turns - turn_index + 1, 0)
        return (
            f"Original question: {query_text}\n"
            f"You are on turn {turn_index}/{self.max_tool_turns}. Remaining tool turns: {remaining}.\n"
            "Use graph_workspace to explore nodes for relationships, and vector_search to pull supporting documents.\n"
            "When you spot gaps, call think_new_questions to brainstorm targeted follow-up queries."
        )

    def _build_fallback_prompt(
        self,
        query_text: str,
        observations: List[str],
        examples: str,
    ) -> str:
        obs_text = "\n".join(observations) if observations else "No tool outputs captured."
        examples_text = f"\nExamples:\n{examples}" if examples else ""
        return (
            "Consolidate the collected tool outputs into a final answer.\n"
            f"Question: {query_text}\n"
            f"Tool outputs:\n{obs_text}{examples_text}\nProvide a concise, factual answer."
        )

    def _build_tools(self) -> Sequence[Tool]:
        filters_param = ObjectParameter(
            description="Optional property filters as key/value pairs.",
            properties={},
            additional_properties=True,
        )
        graph_tool = Tool(
            name="graph_workspace",
            description=(
                "Access the knowledge graph. Retrieve node by (seed) subject as node name for their relationships to explore connections."
            ),
            execute_func=lambda **_: None,
            parameters=ObjectParameter(
                description="Graph access configuration",
                properties={
                    "node_name": StringParameter(
                        description="Target node name / subject name / seed subject name (case-insensitive)",
                        required=True,
                    ),
                    "filters": filters_param,
                },
                required_properties=["node_name"],
                additional_properties=False,
            ),
        )
        vector_tool = Tool(
            name="vector_search",
            description="Retrieve supporting documents via raw vector similarity (no LLM re-ranking).",
            execute_func=lambda **_: None,
            parameters=ObjectParameter(
                description="Vector search parameters",
                properties={
                    "query_text": StringParameter(
                        description="The embedding query text",
                        required=True,
                    ),
                    "top_k": IntegerParameter(
                        description="Number of results to fetch",
                        minimum=1,
                        required=False,
                    ),
                    "filters": filters_param,
                },
                required_properties=["query_text"],
                additional_properties=False,
            ),
        )
        think_tool = Tool(
            name="think_new_questions",
            description=(
                "Given the user's goal and the latest graph/vector findings, synthesize new investigative "
                "questions using semantic search and using the best-fit language / localization that could uncover the remaining evidence from the knowledge database."
            ),
            execute_func=lambda **_: None,
            parameters=ObjectParameter(
                description="Inputs needed to brainstorm new follow-up queries",
                properties={
                    "context_summary": StringParameter(
                        description="Concise summary of known findings or tool outputs",
                        required=True,
                    ),
                    "goal_hint": StringParameter(
                        description="Reminder of the original user question or focus",
                        required=False,
                    ),
                    "max_questions": IntegerParameter(
                        description="Maximum number of follow-up questions to generate (default 1)",
                        minimum=1,
                        required=False,
                    ),
                    "best_fit_locale": StringParameter(
                        description="Preferred language / locale of the question to search for an answer using semantics search (default: English)",
                        required=False,
                    ),
                },
                required_properties=["context_summary"],
                additional_properties=False,
            ),
        )
        self._tool_handlers = {
            graph_tool.get_name(): self._execute_graph_tool,
            vector_tool.get_name(): self._execute_vector_tool,
            think_tool.get_name(): self._execute_think_new_questions,
        }
        return [graph_tool, vector_tool, think_tool]

    def _dispatch_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            logger.warning("Unknown tool requested: %s", tool_name)
            self._color_log("warning", f"Unknown tool requested: {tool_name}")
            return {"error": f"Unsupported tool: {tool_name}"}
        try:
            return handler(**arguments)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool %s failed: %s", tool_name, exc)
            self._color_log("warning", f"Tool {tool_name} failed: {exc}")
            return {"error": str(exc)}

    def _execute_graph_tool(
        self,
        node_name: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        # data_node = self.graph_adapter.get_node(node_name=node_name or "", filters=filters)

        # data_search = self.graph_adapter.search_nodes(
        #     node_name=node_name,
        #     filters=filters,
        #     limit=limit or 10,
        # )

        data_relationships = self.graph_adapter.get_relationships(
            node_name=node_name or "",
            filters=filters,
            limit=limit or 500,
        )
        formatted_relationships, relationship_evidence = self._format_relationships(
            data_relationships,
            anchor=node_name or None,
        )

        # result = {
        #     "mode": mode_key,
        #     "node": data_node,
        #     "node_search": data_search,
        #     "node_relationships": data_relationships,
        #     "formatted_relationships": formatted_relationships,
        #     "relationship_evidence": relationship_evidence,
        # }

        return f"""Node: {node_name}

Relationships:
{formatted_relationships}
        """

    def _execute_vector_tool(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not query_text:
            raise ValueError("query_text is required for vector search")
        if self._vector_calls_used >= self.vector_call_limit:
            self._color_log("warning", "Vector search budget exhausted; skipping vector tool execution.")
            return {"error": "Vector search budget exhausted"}
        self._vector_calls_used += 1
        search_kwargs = {"query_text": query_text}
        if top_k:
            search_kwargs["top_k"] = top_k
        if filters:
            search_kwargs["filters"] = filters
        result_items: Optional[List[Any]] = None

        get_results_fn = getattr(self.retriever, "get_search_results", None)
        if callable(get_results_fn):
            try:
                raw_result: RawSearchResult = get_results_fn(
                    query_text=query_text,
                    top_k=search_kwargs.get("top_k", 5),
                    filters=filters,
                )
                result_items = self._raw_result_to_items(raw_result)
            except TypeError as exc:
                self._color_log(
                    "warning",
                    f"get_search_results signature mismatch, falling back to raw/search: {exc}",
                )
            except Exception as exc:  # noqa: BLE001
                self._color_log("warning", f"get_search_results failed: {exc}")

        if result_items is None:
            raw_fn = getattr(self.retriever, "raw_vector_search", None)
            if callable(raw_fn):
                result_items = raw_fn(**search_kwargs)
            else:
                logger.warning(
                    "Retriever %s lacks raw_vector_search; falling back to search().",
                    self.retriever.__class__.__name__,
                )
                self._color_log(
                    "warning",
                    f"Retriever {self.retriever.__class__.__name__} lacks raw_vector_search; using standard search().",
                )
                fallback_kwargs = search_kwargs.copy()
                retriever_result = self.retriever.search(**fallback_kwargs)
                result_items = retriever_result

        normalized = self._normalize_vector_result(result_items)
        doc_identifiers = self._extract_document_ids(normalized)
        seed_subjects_scoring = self._suggest_seed_subjects(doc_identifiers, top_n=1000)
        if seed_subjects_scoring:
            self._color_log("result", f"Vector-derived seed subjects: {seed_subjects_scoring}")
            
        ranked_seed_subjects_scoring = dict(
            sorted(
                seed_subjects_scoring.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        
        subject_relationships = {}
        for subject in ranked_seed_subjects_scoring.keys():
            relationships = self._execute_graph_tool(
                node_name=subject,
                limit=500,
            )
            subject_relationships[subject] = relationships

        return {
            "top_k": top_k or 5,
            "results": normalized,
            "seed_subjects_scoring": ranked_seed_subjects_scoring,
            "subject_relationships": subject_relationships,
        }

    def _execute_think_new_questions(
        self,
        context_summary: str,
        goal_hint: Optional[str] = None,
        max_questions: Optional[int] = None,
        best_fit_locale: Optional[str] = None,
    ) -> dict[str, Any]:
        summary = (context_summary or "").strip()
        if not summary:
            raise ValueError("context_summary is required for think_new_questions")
        try:
            parsed_limit = int(max_questions) if max_questions is not None else 3
        except (TypeError, ValueError):
            parsed_limit = 3
        limit = max(1, min(parsed_limit, 10))
        goal_text = (str(goal_hint or "").strip()) or "Not provided"
        prompt = f"""
You are an investigative assistant that proposes new graph or document search queries.

Original user goal: {goal_text}
Known findings so far:
{summary}

Recommend up to {limit} new, highly specific follow-up queries that would uncover the missing details needed to fully answer the goal. Avoid generic research questions; be precise about the entities or relationships you want to inspect.

Use the following language for the queries: {best_fit_locale or "English"}.
Return JSON only with the following shape:
{{
  "queries": [
    {{"query": "...", "rationale": "why this helps"}}
  ]
}}
"""
        llm_response = self.llm.invoke(
            input=prompt,
            system_instruction="Return valid JSON listing concrete follow-up queries.",
        )
        raw_text = llm_response.content or ""
        suggestions: List[dict[str, str]] = []
        if raw_text:
            try:
                parsed = json.loads(raw_text)
                candidate_list = parsed.get("queries") if isinstance(parsed, dict) else None
                if isinstance(candidate_list, list):
                    for entry in candidate_list:
                        if not isinstance(entry, dict):
                            continue
                        query = (entry.get("query") or entry.get("question") or "").strip()
                        rationale = (entry.get("rationale") or entry.get("reason") or "").strip()
                        if not query:
                            continue
                        suggestions.append({"query": query, "rationale": rationale})
                        if len(suggestions) >= limit:
                            break
            except Exception:  # noqa: BLE001
                pass

        if not suggestions and raw_text:
            for line in raw_text.splitlines():
                cleaned = line.lstrip("-*0123456789. ").strip()
                if not cleaned:
                    continue
                suggestions.append({"query": cleaned, "rationale": "Derived from free-form response"})
                if len(suggestions) >= limit:
                    break

        query_documents: List[dict[str, Any]] = []
        for suggestion in suggestions:
            query_text = suggestion.get("query")
            if not query_text:
                continue
            self._color_log(
                "tool",
                f"think_new_questions triggering vector_search for follow-up query: {query_text}",
            )
            try:
                vector_payload = self._execute_vector_tool(query_text=query_text, top_k=3)
            except Exception as exc:  # noqa: BLE001
                vector_payload = {"error": str(exc)}
            query_documents.append(vector_payload)
            #     {
            #         # "query": query_text,
            #         # "rationale": suggestion.get("rationale"),
            #         "vector_results": vector_payload,
            #     }
            # )

        extracts = []
        cumulative_seed_subjects_scoring: Counter[str] = Counter()
        nodeset_subject_relationships: dict[str, Any] = {}

        for vector_payload in query_documents:

            results = vector_payload.get("results") or []
            seed_subjects_scoring = vector_payload.get("seed_subjects_scoring") or {}
            subject_relationships = vector_payload.get("subject_relationships") or {}

            cumulative_seed_subjects_scoring.update(seed_subjects_scoring)

            for subject, relationships in subject_relationships.items():
                if subject not in nodeset_subject_relationships:
                    nodeset_subject_relationships[subject] = []
                    nodeset_subject_relationships[subject] = relationships

            for result in results:
              self._color_log("result", f"Processing vector search result: {result}")
              extracts.append(result)
                # content = result.get("content") or ""
                # text = content.text.strip()
                # extracts.append(text)


        ranked_seed_subjects_scoring = dict(
            sorted(
                cumulative_seed_subjects_scoring.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        return {
          "results": extracts,
          "seed_subjects_scoring": ranked_seed_subjects_scoring,
          "nodeset_subject_relationships": nodeset_subject_relationships,
        }
        # return {
        #     # "goal_hint": goal_text,
        #     # "suggested_queries": suggestions,
        #     "documents": query_documents,
        #     # "raw_response": raw_text,
        # }

    def _normalize_vector_result(self, data: Any) -> List[dict[str, Any]]:
        items: List[dict[str, Any]] = []
        if isinstance(data, RetrieverResult):
            source = data.items
        elif isinstance(data, list):
            source = data
        elif hasattr(data, "items"):
            source = getattr(data, "items")
        else:
            return items
        for entry in source:
            if hasattr(entry, "content"):
                content = getattr(entry, "content")
                metadata = getattr(entry, "metadata", None)
            elif isinstance(entry, dict):
                content = entry.get("content") or json.dumps(entry)
                metadata = entry.get("metadata")
            else:
                content = str(entry)
                metadata = None
            items.append({"content": content, "metadata": metadata})
        return items

    def _raw_result_to_items(self, raw_result: RawSearchResult) -> List[Any]:
        formatter_fn = None
        if hasattr(self.retriever, "get_result_formatter"):
            try:
                formatter_fn = self.retriever.get_result_formatter()
            except Exception as exc:  # noqa: BLE001
                self._color_log("warning", f"Failed to obtain record formatter: {exc}")
                formatter_fn = None
        formatted_items: List[Any] = []
        for record in raw_result.records:
            if formatter_fn:
                try:
                    formatted_items.append(formatter_fn(record))
                    continue
                except Exception as exc:  # noqa: BLE001
                    self._color_log("warning", f"Record formatting failed: {exc}")
            formatted_items.append({"content": str(record), "metadata": None})
        return formatted_items

    def _extract_document_ids(self, results: List[dict[str, Any]]) -> List[Tuple[str, Optional[str]]]:
        doc_ids: List[Tuple[str, Optional[str]]] = []
        for item in results:
            metadata = item.get("metadata") if isinstance(item, dict) else None
            if not isinstance(metadata, dict):
                continue
            element_identifier = metadata.get("element_id") or metadata.get("elementId")
            if element_identifier is not None:
                element_text = str(element_identifier).strip()
                if element_text:
                    doc_ids.append((element_text, None))
                    continue
            for property_key in ("id", "chunk_id", "document_id"):
                value = metadata.get(property_key)
                if value is None:
                    continue
                value_text = str(value).strip()
                if value_text:
                    doc_ids.append((value_text, property_key))
                    break
        return doc_ids

    def _suggest_seed_subjects(
        self,
        doc_identifiers: List[Tuple[str, Optional[str]]],
        top_n: int = 10,
    ) -> Counter[str]:
        driver = getattr(self.retriever, "driver", None)
        if not driver or not doc_identifiers:
            return Counter()
        counter: Counter[str] = Counter()
        for doc_id, property_key in doc_identifiers:
            neighbors = self._fetch_neighbors_for_document(
                driver,
                doc_id,
                property_key=property_key,
            )
            for neighbor in neighbors:
                if neighbor:
                    counter[neighbor] += 1
        if top_n and top_n > 0:
            trimmed: Counter[str] = Counter()
            for name, score in counter.most_common(top_n):
                trimmed[name] = score
            return trimmed
        return counter

    def _fetch_neighbors_for_document(
        self,
        driver: neo4j.Driver,
        doc_id: str,
        property_key: Optional[str] = None,
        limit: int = 50,
    ) -> List[str]:
        query = """
    CALL () {
  WITH $doc_id AS doc_id
  MATCH (chunk)
  WHERE elementId(chunk) = doc_id
  RETURN chunk
  UNION
  WITH $doc_id AS doc_id, $property_key AS property_key
  WHERE property_key IS NOT NULL
  MATCH (chunk)
  WHERE chunk[property_key] = doc_id
  RETURN chunk
}
WITH DISTINCT chunk
OPTIONAL MATCH (chunk)-[:FROM_CHUNK]->(origin)
WITH coalesce(origin, chunk) AS anchor
MATCH (anchor)-[]-(neighbor)
WHERE neighbor.name IS NOT NULL AND trim(neighbor.name) <> ""
RETURN DISTINCT neighbor.name AS name
LIMIT $limit
"""
        try:
            with driver.session(database=getattr(self.retriever, "neo4j_database", None)) as session:
                records = session.run(
                    query,
                    doc_id=doc_id,
                    property_key=property_key,
                    limit=limit,
                )
                return [record["name"] for record in records if record.get("name")]
        except Exception as exc:  # noqa: BLE001
            self._color_log("warning", f"Neighbor lookup failed for doc {doc_id}: {exc}")
            return []

    def _serialize_tool_result(self, tool_name: str, payload: str | dict[str, Any]) -> str:
        if isinstance(payload, str):
            return f"{self.TOOL_RESULT_PREFIX}: {tool_name} - {payload}"
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        return f"{self.TOOL_RESULT_PREFIX}: {tool_name} - {serialized}"

    def _build_default_graph_adapter(self) -> Optional[GraphAdapter]:
        driver = getattr(self.retriever, "driver", None)
        if driver is None:
            return None
        database = getattr(self.retriever, "neo4j_database", None)
        return Neo4jGraphAdapter(driver=driver, database=database)

    def _format_relationships(
        self,
        relationships: Optional[List[dict[str, Any]]],
        anchor: Optional[str] = None,
    ) -> tuple[List[str], List[str]]:
        if not relationships:
            return [], []
        formatted: List[str] = []
        evidence: List[str] = []
        anchor_name = anchor or "unknown"
        for rel in relationships:
            neighbor = rel.get("neighbor") or rel.get("target") or "unknown"
            rel_type = rel.get("rel_type") or "UNKNOWN"
            direction = rel.get("direction") or "OUT"
            source_name = rel.get("node") or rel.get("source") or anchor_name
            start_node = source_name or anchor_name
            end_node = neighbor or "unknown"
            if direction == "IN":
                start_node = neighbor or "unknown"
                end_node = source_name or anchor_name
            formatted.append(f"{start_node} -[{rel_type}]-> {end_node}")
            evidence.append(
                f"{start_node} -[{rel_type} ({direction})]-> {end_node}; neighbor_labels={rel.get('neighbor_labels')}"
            )
        return formatted, evidence

    def _color_log(self, category: str, message: str) -> None:
        color = self.COLOR_CODES.get(category, "")
        reset = self.COLOR_RESET if color else ""
        print(f"{color}{message}{reset}")
