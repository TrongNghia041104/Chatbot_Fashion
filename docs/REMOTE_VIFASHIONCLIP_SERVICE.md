# Chạy ViFashionCLIP Trên Vast.ai GPU Qua Remote Embedding Service

Tài liệu này dành cho kiến trúc:

```text
Local FastAPI chatbot
    -> gọi http://localhost:18080/embed
    -> SSH tunnel
    -> ViFashionCLIP embedding service trên Vast.ai GPU
```

Ollama vẫn dùng tunnel port `11434` như trước. ViFashionCLIP dùng thêm tunnel port `18080`.

---

## 1. Vì sao cần service này?

`qwen3`, `qwen2.5vl`, `bge-m3` chạy qua Ollama trên Vast.ai.

Nhưng ViFashionCLIP không phải Ollama model. Nó là PyTorch checkpoint:

```text
Vietnamese/vifashionclip_aiteamvn_embedding_v2_projection_336k/stage2_last_layers/best_stage2_model.pt
```

Nếu backend chạy local, PyTorch sẽ dùng GPU local nếu có. Nếu local không có CUDA, nó dùng CPU:

```text
Device: cpu
```

Remote embedding service giải quyết việc đó bằng cách chạy riêng ViFashionCLIP trên Vast.ai GPU.

---

## 2. Port sử dụng

| Port local | Trỏ tới | Công dụng |
|---:|---|---|
| `11434` | Ollama trên Vast.ai | LLM, vision, BGE-M3 |
| `18080` | ViFashionCLIP service trên Vast.ai | Layer A text embedding |

---

## 3. Trên Vast.ai: chuẩn bị project

Bạn cần có thư mục `Chatbot_Fashion` trên Vast.ai, gồm ít nhất:

```text
Chatbot_Fashion/
├── app/
├── scripts/
├── requirements.txt
└── Vietnamese/
    └── vifashionclip_aiteamvn_embedding_v2_projection_336k/
        └── stage2_last_layers/
            └── best_stage2_model.pt
```

Nếu chưa có code/checkpoint trên Vast.ai, copy project lên Vast.ai trước.

---

## 4. Trên Vast.ai: cài dependency

```bash
cd /path/to/Chatbot_Fashion
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Kết quả mong muốn:

```text
True
NVIDIA ...
```

---

## 5. Trên Vast.ai: chạy embedding service

```bash
cd /path/to/Chatbot_Fashion
source venv/bin/activate
python scripts/vifashionclip_embedding_service.py --host 0.0.0.0 --port 18080 --preload
```

Nếu thành công, bạn sẽ thấy:

```text
[INFO] Preloading ViFashionCLIP model...
[OK] ViFashionCLIP loaded: ...
     Device: cuda | Dim: 512
Uvicorn running on http://0.0.0.0:18080
```

`--preload` làm request đầu tiên không bị đứng lâu do load checkpoint.

---

## 6. Trên local: mở SSH tunnel cho embedding service

Mở terminal mới trên local:

```powershell
ssh -p <SSH_PORT> root@<VAST_IP> -L 18080:localhost:18080 -N
```

Terminal này phải giữ mở trong lúc dùng chatbot.

Bạn vẫn cần tunnel Ollama:

```powershell
ssh -p <SSH_PORT> root@<VAST_IP> -L 11434:localhost:11434 -N
```

Có thể mở 2 terminal riêng, hoặc gộp hai port vào một lệnh:

```powershell
ssh -p <SSH_PORT> root@<VAST_IP> -L 11434:localhost:11434 -L 18080:localhost:18080 -N
```

---

## 7. Trên local: kiểm tra service

```powershell
Invoke-RestMethod http://localhost:18080/health
```

Kết quả mong muốn:

```text
status         : ok
model_loaded   : True
device         : cuda
cuda_available : True
```

Warmup/test embedding:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:18080/warmup
```

---

## 8. Trên local: chạy chatbot dùng remote embedding

Trong terminal chạy backend local:

```powershell
cd "D:\KHÓA LUẬN\WORKSPACE\Chatbot_Fashion"
.\venv\Scripts\activate

$env:PRODUCT_EMBEDDING_BACKEND="remote"
$env:VIFASHIONCLIP_SERVICE_URL="http://localhost:18080"
$env:ENABLE_PRODUCT_RERANKER="false"

python main.py
```

`ENABLE_PRODUCT_RERANKER=false` là khuyến nghị ban đầu để giảm latency. Khi hệ thống đã ổn, có thể bật lại reranker sau.

---

## 9. Cách biết local backend đã dùng remote chưa

Khi search sản phẩm lần đầu, log local phải là:

```text
[INFO] Using remote ViFashionCLIP embedding service: http://localhost:18080
```

Không nên còn dòng:

```text
[OK] ViFashionCLIP loaded: ... Device: cpu
```

Nếu thấy `Device: cpu` ở local backend, nghĩa là backend đang dùng local checkpoint, chưa dùng remote.

---

## 10. Lỗi thường gặp

### Local báo remote embedding service unavailable

Kiểm tra tunnel:

```powershell
Invoke-RestMethod http://localhost:18080/health
```

Nếu lỗi, tunnel `18080` chưa mở hoặc service trên Vast.ai chưa chạy.

### Service trên Vast.ai vẫn báo `device: cpu`

Kiểm tra PyTorch CUDA trên Vast.ai:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu `False`, instance/template PyTorch chưa có CUDA đúng, hoặc bạn đang dùng Python env không có torch CUDA.

### Request đầu tiên vẫn lâu

Chạy service với `--preload`, hoặc gọi:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:18080/warmup
```

### Product search vẫn chậm

Tắt reranker trước:

```powershell
$env:ENABLE_PRODUCT_RERANKER="false"
```

Reranker cũng là PyTorch model và nếu chạy local CPU thì sẽ kéo chậm.

---

## 11. Tóm tắt lệnh cần mở mỗi lần chạy

Trên Vast.ai:

```bash
cd /path/to/Chatbot_Fashion
source venv/bin/activate
ollama serve &
python scripts/vifashionclip_embedding_service.py --host 0.0.0.0 --port 18080 --preload
```

Trên local, terminal tunnel:

```powershell
ssh -p <SSH_PORT> root@<VAST_IP> -L 11434:localhost:11434 -L 18080:localhost:18080 -N
```

Trên local, terminal backend:

```powershell
cd "D:\KHÓA LUẬN\WORKSPACE\Chatbot_Fashion"
.\venv\Scripts\activate
$env:PRODUCT_EMBEDDING_BACKEND="remote"
$env:VIFASHIONCLIP_SERVICE_URL="http://localhost:18080"
$env:ENABLE_PRODUCT_RERANKER="false"
python main.py
```
