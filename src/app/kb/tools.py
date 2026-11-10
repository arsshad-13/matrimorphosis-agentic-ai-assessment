
"""The two knowledge-base tools.

Task 2: search_knowledge_base is asynchronous; KB search runs in a worker thread.
Task 6: knowledge-base ownership is based on user_id, while uploaded files
are resolved using session_id.
"""

import asyncio

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import KB_SEARCHES

LOW_CONFIDENCE = 0.30


def build_kb_tools(kb: KnowledgeBase) -> list:
    @tool
    async def ingest_document(source: str, runtime: ToolRuntime[Ctx]) -> str:
        """Add a document to the user's knowledge base so it can be searched later.
        Reads PDF, DOCX, TXT, MD and images (scanned pages are OCR'd), splits them
        into sections, and indexes them. Call it whenever the user uploads a file
        or gives a document link, then tell the user in one sentence what it contains.

        Args:
            source: the file_id from the upload message (e.g. "a1b2c3d4e5"),
                    or a public http(s) URL to a document
        """
        session_id = runtime.context.session_id
        user_id = runtime.context.user_id

        try:
            # Uploaded files are stored under the chat session.
            if source.startswith(("http://", "https://")):
                _, path = await files.download(session_id, source)
                filename = path.name.split("__", 1)[1]
            else:
                path, filename = files.resolve(session_id, source.strip())

            # The indexed document belongs to the authenticated user,
            # not just the chat session.
            result, existing = await ingest_file(
                kb,
                user_id,
                path,
                filename,
                runtime.config,
            )

        except Exception as exc:
            raise to_tool_exception(exc) from exc

        if existing:
            return (
                f"'{existing['filename']}' is already in the knowledge base "
                f"({existing['n_sections']} sections)."
            )

        p = result.profile
        heads = "; ".join(s.heading for s in result.sections[:8])
        ocr = (
            f", {result.ocr_pages} page(s) read with OCR"
            if result.ocr_pages
            else ""
        )

        return (
            f"Ingested '{filename}': {p.title} [{p.doc_type}, {p.language}]. "
            f"{p.summary} {len(result.sections)} sections, "
            f"{result.n_chunks} chunks{ocr}. Sections: {heads}. "
            f"Keywords: {', '.join(p.keywords)}."
        )

    @tool
    async def search_knowledge_base(
        query: str,
        runtime: ToolRuntime[Ctx],
        k: int = 4,
        doc_name: str = "",
    ) -> str:
        """Search the documents uploaded by this user across all their chat sessions.
        Use it BEFORE other tools or your own knowledge whenever the KNOWLEDGE BASE
        overview/keywords in the system prompt suggest the question can be answered
        from those documents. Returns numbered passages, each with a bracketed source
        that must be copied exactly when citing.

        Args:
            query: what to look for, phrased as a full question or topic
            k: number of passages to return (1-6)
            doc_name: exact file name from the KNOWLEDGE BASE document list.
                      ALWAYS pass it when the user names or hints at a specific
                      document; leave empty to search all documents.
        """
        user_id = runtime.context.user_id
        exact = None

        if doc_name:
            exact = kb.resolve_doc_name(user_id, doc_name)

            if exact is None:
                names = (
                    ", ".join(
                        f'"{d["filename"]}"'
                        for d in kb.documents(user_id)
                    )
                    or "(none)"
                )
                raise ToolException(
                    f"ERROR[UNKNOWN_DOCUMENT]: No document matches "
                    f"'{doc_name}'. Documents in this user's knowledge base: "
                    f"{names}. Next step: retry with one of these names, "
                    "or without doc_name."
                )

        try:
            hits = await asyncio.to_thread(
                kb.search,
                user_id,
                query,
                k=max(1, min(k, 6)),
                doc_name=exact,
            )
        except Exception as exc:
            raise to_tool_exception(exc) from exc

        best = max((h.relevance for h in hits), default=0.0)

        KB_SEARCHES.add(
            1,
            {"found": str(best >= LOW_CONFIDENCE).lower()},
        )

        if not hits:
            return (
                "No passages found: the user's knowledge base is empty "
                "or the document name does not exist."
            )

        head = (
            f"{len(hits)} passages"
            f"{f' from {exact}' if exact else ''}, "
            f"best relevance {best:.2f}."
        )

        if best < LOW_CONFIDENCE:
            head += (
                " LOW CONFIDENCE: nothing clearly matches. "
                "Say the documents do not cover this unless a passage below answers it."
            )
        else:
            head += (
                " Answer only from these passages. After each claim, "
                "copy the bracketed source of the passage you used, exactly."
            )

        body = "\n\n".join(
            f"Passage {i} [{h.cite}] relevance {h.relevance:.2f}\n{h.text}"
            for i, h in enumerate(hits, start=1)
        )

        return f"{head}\n\n{body}"

    for t in (ingest_document, search_knowledge_base):
        t.handle_tool_error = True

    return [ingest_document, search_knowledge_base]
