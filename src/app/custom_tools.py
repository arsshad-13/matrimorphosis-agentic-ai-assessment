"""Tasks 7 and 8: custom knowledge-base tools with observability."""

import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log


_meter = metrics.get_meter("agent")

KBSTATS_CALLS = _meter.create_counter(
    "kbstats.calls",
    unit="1",
    description="Knowledge-base statistics tool calls",
)

OUTLINE_CALLS = _meter.create_counter(
    "outline.calls",
    unit="1",
    description="Document outline tool calls",
)


def _document_names(documents: list[dict]) -> str:
    """Format valid document names for an error message."""
    return ", ".join(f'"{d["filename"]}"' for d in documents) or "(none)"


def _unknown_document(doc_name: str, documents: list[dict]) -> ToolException:
    """Create the required safe unknown-document error."""
    return ToolException(
        f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}'. "
        f"Documents in this user's knowledge base: {_document_names(documents)}. "
        "Next step: retry with one of these names."
    )


def build_custom_tools(kb: KnowledgeBase) -> list:

    @tool
    async def kb_stats(
        runtime: ToolRuntime[Ctx],
        top_keywords: int = 5,
    ) -> str:
        """Show statistics about the calling user's knowledge base.
        Use this when the user asks how many documents, sections or chunks
        are stored, or asks for the most common document keywords.

        Args:
            top_keywords: Number of top keywords to display (1-20).
        """
        started = time.perf_counter()
        status = "error"

        try:
            if not isinstance(top_keywords, int) or not 1 <= top_keywords <= 20:
                raise ToolException(
                    "ERROR[INVALID_INPUT]: top_keywords must be between 1 and 20. "
                    "Next step: provide an integer from 1 to 20."
                )

            user_id = runtime.context.user_id
            stats = await asyncio.to_thread(kb.stats, user_id)

            if stats["documents"] == 0:
                status = "ok"
                return "The knowledge base is empty."

            keywords = ", ".join(stats["keywords"][:top_keywords])

            status = "ok"
            return (
                f'{stats["documents"]} documents, '
                f'{stats["sections"]} sections, '
                f'{stats["chunks"]} chunks. '
                f"Top keywords: {keywords}"
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            KBSTATS_CALLS.add(1, {"status": status})
            log.info(
                "kb_stats_done",
                tool="kb_stats",
                status=status,
                duration_ms=duration_ms,
            )

    @tool
    async def document_outline(
        doc_name: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Show the structure and section headings of a document in the
        calling user's knowledge base. Use this when the user asks what
        a document contains, how it is organised, or which section covers
        a topic. Use the outline to identify the relevant section before
        searching its contents.

        Args:
            doc_name: Document filename or identifiable name from the
                      calling user's knowledge-base document list.
        """
        started = time.perf_counter()
        status = "error"
        section_count = 0

        try:
            user_id = runtime.context.user_id

            documents = await asyncio.to_thread(kb.documents, user_id)
            exact_name = await asyncio.to_thread(
                kb.resolve_doc_name, user_id, doc_name
            )

            if exact_name is None:
                raise _unknown_document(doc_name, documents)

            document = next(
                (d for d in documents if d["filename"] == exact_name),
                None,
            )

            if document is None:
                raise _unknown_document(doc_name, documents)

            sections = await asyncio.to_thread(
                kb.outline,
                user_id,
                document["doc_id"],
            )

            section_count = len(sections)
            lines = [
                f"Outline of '{exact_name}' ({section_count} sections):"
            ]

            for index, section in enumerate(sections, start=1):
                page_start = section.get("page_start")
                page_end = section.get("page_end")

                if page_start is None:
                    page_text = ""
                elif page_start == page_end:
                    page_text = f" (p.{page_start})"
                else:
                    page_text = f" (p.{page_start}-{page_end})"

                lines.append(
                    f'{index}. [L{section["level"]}] '
                    f'{section["heading"]}'
                    f'{page_text}: {section["summary"]}'
                )

            status = "ok"
            return "\n".join(lines)

        except Exception as exc:
            raise to_tool_exception(exc) from exc

        finally:
            duration_ms = round((time.perf_counter() - started) * 1000)
            OUTLINE_CALLS.add(1, {"status": status})
            log.info(
                "document_outline_done",
                tool="document_outline",
                status=status,
                duration_ms=duration_ms,
                sections=section_count,
            )

    @tool
    async def delete_document(
        doc_name: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Delete one document from the calling user's knowledge base.
        Use this when the user explicitly asks to remove a document.

        Args:
            doc_name: Filename or identifiable name of the document
                      in the calling user's knowledge-base list.
        """
        try:
            user_id = runtime.context.user_id

            documents = await asyncio.to_thread(kb.documents, user_id)
            exact_name = await asyncio.to_thread(
                kb.resolve_doc_name, user_id, doc_name
            )

            if exact_name is None:
                raise _unknown_document(doc_name, documents)

            document = next(
                (d for d in documents if d["filename"] == exact_name),
                None,
            )

            if document is None:
                raise _unknown_document(doc_name, documents)

            deleted = await asyncio.to_thread(
                kb.delete_document,
                user_id,
                document["doc_id"],
            )

            if not deleted:
                raise _unknown_document(doc_name, documents)

            remaining = await asyncio.to_thread(kb.documents, user_id)

            return (
                f"Deleted '{exact_name}'. "
                f"{len(remaining)} documents left."
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

    @tool
    async def update_document(
        doc_name: str,
        source: str,
        runtime: ToolRuntime[Ctx],
    ) -> str:
        """Replace an existing knowledge-base document with a newly uploaded
        version. The new file is ingested first; the old document is removed
        only after successful ingestion. If the new content already exists
        in the knowledge base, no changes are made.

        Use this when the user explicitly asks to update or replace a document.

        Args:
            doc_name: Existing document filename or identifiable name in
                      the calling user's knowledge base.
            source: file_id of the newly uploaded replacement file.
        """
        try:
            user_id = runtime.context.user_id
            session_id = runtime.context.session_id

            documents = await asyncio.to_thread(kb.documents, user_id)
            exact_name = await asyncio.to_thread(
                kb.resolve_doc_name, user_id, doc_name
            )

            if exact_name is None:
                raise _unknown_document(doc_name, documents)

            old_document = next(
                (d for d in documents if d["filename"] == exact_name),
                None,
            )

            if old_document is None:
                raise _unknown_document(doc_name, documents)

            # Resolve the new upload from the current chat session.
            path, _uploaded_filename = await asyncio.to_thread(
                files.resolve,
                session_id,
                source.strip(),
            )

            # Ingest the replacement under the original document filename.
            # The old document remains untouched if ingestion fails.
            result, existing = await ingest_file(
                kb,
                user_id,
                path,
                exact_name,
                runtime.config,
            )

            if existing:
                return f"'{exact_name}' is already upto date."

            # Delete the previous version only after successful ingestion.
            deleted = await asyncio.to_thread(
                kb.delete_document,
                user_id,
                old_document["doc_id"],
            )

            if not deleted:
                raise ToolException(
                    "ERROR[UNKNOWN_DOCUMENT]: The previous document could "
                    "not be found during replacement. "
                    "Next step: check the knowledge base and retry."
                )

            return (
                f"Updated '{exact_name}': "
                f"{len(result.sections)} sections, "
                f"{result.n_chunks} chunks."
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

    for custom_tool in (
        kb_stats,
        document_outline,
        delete_document,
        update_document,
    ):
        custom_tool.handle_tool_error = True

    return [
        kb_stats,
        document_outline,
        delete_document,
        update_document,
    ]
