from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        context_blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = (
                metadata.get("source_url")
                or metadata.get("source")
                or metadata.get("doc_id")
                or result.get("id")
                or "không rõ nguồn"
            )
            context_blocks.append(
                f"[{index}] Nguồn: {source}\n{result['content']}"
            )

        context = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là trợ lý trả lời câu hỏi dựa trên cơ sở tri thức.\n"
            "Chỉ sử dụng thông tin trong phần NGỮ CẢNH bên dưới. "
            "Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ rằng không tìm thấy "
            "đủ thông tin. Khi có thể, hãy trích dẫn nguồn bằng số [1], [2], ...\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI:\n{question}\n\n"
            "TRẢ LỜI:"
        )
        return self.llm_fn(prompt)
