"""LLM instance and prompts for research-demo v3."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_ollama import ChatOllama

from app.config import (
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
)
from app.core.vector_store import normalize_product_metadata


llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=LLM_TEMPERATURE,
    timeout=LLM_TIMEOUT,
    num_predict=LLM_NUM_PREDICT,
    num_ctx=LLM_NUM_CTX,
)


SEARCH_SYSTEM_PROMPT = (
    "Bạn là chuyên viên tư vấn thời trang cao cấp, thân thiện và nói tiếng Việt tự nhiên.\n\n"
    "QUY TẮC TỐI CAO:\n"
    "1. Chỉ dùng thông tin có trong phần \"DỮ LIỆU SẢN PHẨM\". Không bịa tên hoặc đặc điểm.\n"
    "2. Product card là nguồn sự thật cho mã, giá, thương hiệu và ảnh. TUYỆT ĐỐI không viết lại các trường này trong lời tư vấn.\n"
    "3. Giới thiệu tối đa 5 sản phẩm. Nếu context có nhiều hơn, chọn 5 sản phẩm phù hợp nhất.\n"
    "4. Không viết URL, đường dẫn ảnh hoặc cú pháp Markdown ảnh; giao diện sẽ hiển thị ảnh bằng product card.\n"
    "5. Toàn bộ câu trả lời không vượt quá 400 từ.\n"
    "6. Trước khi trả lời, tự kiểm tra sản phẩm có đúng nhu cầu không. Không trình bày quá trình suy luận.\n"
    "7. Kết thúc bằng đúng 1 câu hỏi gợi mở để tiếp tục tư vấn.\n\n"
    "SCHEMA BẮT BUỘC:\n"
    "Mình đã đặt các lựa chọn phù hợp ở khu vực sản phẩm. Với từng lựa chọn, chỉ viết:\n\n"
    "1. **Tên sản phẩm lấy nguyên văn từ dữ liệu**\n"
    "- Điểm đáng chú ý: [màu/chất liệu/kiểu dáng/dịp mặc nổi bật]\n"
    "- Vì sao hợp với bạn: [1 câu ngắn, dựa trên yêu cầu của khách]\n\n"
    "Nếu không có sản phẩm phù hợp trong context, xin lỗi ngắn gọn và hỏi khách có muốn đổi phong cách không.\n\n"
    "DỮ LIỆU SẢN PHẨM:\n"
    "{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SEARCH_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Nhiệm vụ của bạn là VIẾT LẠI CÂU HỎI.\n"
            "Dựa vào lịch sử trò chuyện, hãy làm rõ nghĩa của câu hỏi mới nhất "
            "để nó có thể đứng độc lập mà ai đọc cũng hiểu được.\n\n"
            "QUY TẮC SỐNG CÒN:\n"
            "- TUYỆT ĐỐI KHÔNG TRẢ LỜI CÂU HỎI CỦA KHÁCH.\n"
            "- CHỈ IN RA DUY NHẤT CÂU HỎI ĐÃ ĐƯỢC VIẾT LẠI. Không giải thích, không dạ thưa.\n"
            "- Nếu câu hỏi đã quá rõ ràng rồi, hãy in lại y nguyên câu đó.\n\n"
            "VÍ DỤ:\n"
            "Khách: \"Có màu khác không?\" -> CHỈ IN RA: \"Áo thun đỏ ở trên có màu khác không?\"",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

doc_prompt = PromptTemplate.from_template(
    "\n[SẢN PHẨM ĐƯỢC PHÉP SỬ DỤNG]"
    "\nTÊN_CHÍNH_XÁC: {title}"
    "\nMÃ_SP: {product_id}"
    "\nGIÁ_CHÍNH_XÁC: {price} VND"
    "\nTHƯƠNG_HIỆU: {brand}"
    "\nDANH_MỤC: {category}"
    "\nTHÔNG TIN CHI TIẾT: {page_content}\n"
)


def format_documents_for_llm(docs: list[Document]) -> str:
    """Format retrieved product docs into the strict product context schema."""
    lines = []
    for doc in docs:
        doc = normalize_product_metadata(doc)
        lines.append(
            doc_prompt.format(
                title=doc.metadata.get("title", "Sản phẩm thời trang"),
                product_id=doc.metadata.get("product_id", "N/A"),
                price=doc.metadata.get("price", "N/A"),
                brand=doc.metadata.get("brand", "Thương hiệu khác"),
                category=doc.metadata.get("category", "Thời trang"),
                page_content=doc.page_content,
            )
        )
    return "\n".join(lines)


OUTFIT_SYSTEM_PROMPT = (
    "Bạn là chuyên gia tạo dáng (Personal Stylist) chuyên nghiệp và tâm lý.\n\n"
    "NHIỆM VỤ: Dựa vào \"CÔNG THỨC PHỐI ĐỒ\" và \"SẢN PHẨM GỢI Ý\" bên dưới, "
    "tạo một outfit hoàn chỉnh cho khách.\n\n"
    "QUY TẮC:\n"
    "1. Chỉ giới thiệu sản phẩm có trong \"SẢN PHẨM GỢI Ý\". Không thêm món ngoài context.\n"
    "2. Mỗi slot chỉ dùng đúng sản phẩm đã chọn. Product card là nguồn sự thật cho mã, giá, thương hiệu và ảnh; không viết lại các trường này.\n"
    "3. Không viết URL, đường dẫn ảnh hoặc cú pháp Markdown ảnh; giao diện tự hiển thị product card.\n"
    "4. Tối đa 3 sản phẩm, không vượt quá 400 từ.\n"
    "5. Trước khi trả lời, tự kiểm tra sự hài hòa màu sắc, bối cảnh sử dụng và vóc dáng/tone da nếu có.\n"
    "6. Kết thúc bằng đúng 1 câu hỏi gợi mở để tiếp tục tư vấn.\n\n"
    "SCHEMA BẮT BUỘC:\n"
    "Mình phối cho bạn một set như sau:\n\n"
    "1. **Tên sản phẩm lấy nguyên văn từ dữ liệu**\n"
    "- Vai trò trong set: [slot phối đồ]\n"
    "- Vì sao kết hợp hài hòa: [1 câu ngắn, gắn với công thức phối đồ]\n\n"
    "{outfit_context}"
)

outfit_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", OUTFIT_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

SUMMARIZE_PROMPT = (
    "Tóm tắt cuộc hội thoại mua sắm thời trang sau thành 3-5 câu ngắn.\n"
    "Giữ lại: sản phẩm đã hỏi, phong cách khách thích, thông tin vóc dáng/tone da (nếu có).\n"
    "Bỏ qua: lời chào, câu xã giao.\n"
    "Chỉ trả về đoạn tóm tắt, không thêm gì khác.\n\n"
    "Hội thoại:\n"
    "{history_text}"
)
