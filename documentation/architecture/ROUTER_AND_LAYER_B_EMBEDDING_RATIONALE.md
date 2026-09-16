# Router Và Lý Do Layer B Dùng BGE-M3

> **Tài liệu lý do thiết kế cũ:** Taxonomy route bên dưới có tên legacy. Kiến trúc production mới nhất nằm tại [`05_INTENT_ROUTER_DECISION.md`](05_INTENT_ROUTER_DECISION.md); không dùng numeric confidence trong file này như router accuracy.

## Intent chính

Hệ thống nên giữ ít route chính, nhưng mỗi route có `action` và `entities` để mở rộng:

- `product_search`: tìm sản phẩm bằng text.
- `image_product_search`: người dùng gửi ảnh và muốn tìm sản phẩm giống ảnh.
- `outfit_advice`: tư vấn phối đồ từ text.
- `image_outfit_advice`: người dùng gửi ảnh sản phẩm đơn lẻ và hỏi phối đồ.
- `profile_inquiry`: hỏi hoặc cập nhật thông tin profile.
- `greeting`, `chitchat`, `out_of_scope`, `clarify`.

Với ảnh đơn lẻ và không có text, mặc định là `image_product_search`. Với ảnh kèm câu như "món này phối sao", "mặc với gì", "mix với gì", router chuyển sang `image_outfit_advice`.

## Luồng ảnh phối đồ

```text
ảnh sản phẩm đơn lẻ
  -> FashionCLIP image retrieval
  -> đọc top candidates từ metadata Layer A
  -> tính confidence: top score, score gap, category agreement
  -> nếu đủ chắc: tạo base item context từ metadata
  -> nếu yếu: mới gọi VLM để caption/enrich ảnh
  -> build query cho Layer B
  -> BGE-M3 search outfit rule
  -> lấy rule chi tiết
  -> Layer A retrieval lấy sản phẩm thật để phối cùng
```

VLM không nên là bước mặc định đầu tiên vì latency qua VastAI có thể cao. VLM phù hợp làm fallback khi image retrieval lẫn category, score thấp, hoặc user hỏi thuộc tính visual sâu.

## Vì sao không dùng ViFashionCLIP cho Layer B

ViFashionCLIP vẫn có khả năng semantic, nhưng semantic của nó được học để nối **text mô tả sản phẩm** với **ảnh thời trang**. Nó rất hợp cho Layer A:

```text
"giày sneaker trắng nữ"
  -> gần ảnh/sản phẩm sneaker trắng trong catalog
```

Layer B lại truy hồi đối tượng khác: rule stylist dạng text dài, gồm bối cảnh, phong cách, dáng người, tone da và lý do tư vấn:

```text
"dáng quả lê, da ngăm, đi làm mùa đông, phong cách smart casual"
  -> rule phối đồ phù hợp
```

BGE-M3 là text embedding tổng quát, phù hợp hơn cho truy hồi ngữ nghĩa câu dài và rule tư vấn. Vì vậy kiến trúc chọn:

```text
Layer A = product/image retrieval -> ViFashionCLIP/FashionCLIP
Layer B = stylist-rule retrieval  -> BGE-M3
```

## Không Nhồi Mọi Trường Vào Vector Layer B

Layer B vẫn không nên embed tất cả field. Công thức vector nên giữ các trường ngữ nghĩa tự do:

```text
rule_key + phong_cach + boi_canh + ly_do_tu_van
```

Các field `dang_nguoi` và `tone_da` là categorical/profile slots, nên dùng payload filter với wildcard như `Mọi vóc dáng`, `Mọi tone da` và các biến thể bắt đầu bằng các cụm này. Field `goi_y_phoi_cung` là output của rule để bung outfit chi tiết, không phải tín hiệu chính để semantic search rule chủ đạo.

Thiết kế đúng là:

```text
vector search   -> hiểu ý định, phong cách, bối cảnh, món đồ
payload filter  -> cá nhân hoá theo dáng người và tone da
payload output  -> dùng goi_y_phoi_cung để lấy các món phối cùng
```

Nếu hội đồng hỏi "ViFashionCLIP cũng hiểu semantic mà?", câu trả lời là:

> Đúng, ViFashionCLIP có hiểu semantic, nhưng đó là semantic được tối ưu cho tương quan ảnh-sản phẩm. Layer B không truy hồi ảnh hay sản phẩm, mà truy hồi tri thức phối đồ dạng văn bản dài. Do đó dùng BGE-M3 giúp matching theo ý định, bối cảnh, dáng người và tone da tốt hơn. Hai model không trùng vai trò; chúng phục vụ hai không gian truy hồi khác nhau.

Để giảm chi phí, hệ thống chỉ lazy-load BGE-M3 khi route là outfit và pre-index Layer B sẵn trong Qdrant.
