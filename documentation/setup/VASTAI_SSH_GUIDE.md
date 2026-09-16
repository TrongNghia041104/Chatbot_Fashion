# Hướng dẫn SSH vào máy thuê Vast.ai

File này ghi lại toàn bộ các bước để kết nối SSH thành công vào máy thuê trên Vast.ai, cấu hình tunnel cho Ollama, và sửa lỗi thường gặp khi dùng VS Code Remote SSH.

Mục tiêu cuối cùng là:

- SSH được vào máy Vast.ai.
- Máy local gọi được Ollama trên Vast.ai qua `localhost:11434`.
- VS Code Remote SSH kết nối được bằng tên ngắn, ví dụ `VuTrongNghia`.

---

## 1. Hiểu nhanh: SSH và SSH tunnel là gì?

Khi thuê GPU trên Vast.ai, model Ollama chạy trên máy GPU đó, không chạy trực tiếp trên laptop. Vì vậy laptop cần một đường kết nối an toàn để nói chuyện với máy Vast.ai.

SSH có 2 việc chính:

1. Đăng nhập vào máy Vast.ai để cài và chạy lệnh.
2. Tạo tunnel để chuyển cổng trên máy local sang cổng trên máy Vast.ai.

Ví dụ tunnel:

```bash
ssh -p 13431 root@115.72.131.186 -L 11434:localhost:11434
```

Ý nghĩa:

- `ssh`: dùng SSH để kết nối.
- `-p 13431`: port SSH do Vast.ai cấp.
- `root@115.72.131.186`: đăng nhập user `root` vào IP `115.72.131.186`.
- `-L 11434:localhost:11434`: nối `localhost:11434` trên laptop tới `localhost:11434` trên máy Vast.ai.

Vì Ollama mặc định chạy ở port `11434`, nên chatbot local sẽ gọi:

```text
http://localhost:11434
```

nhưng thực tế request được chuyển sang Ollama đang chạy trên GPU Vast.ai.

---

## 2. Lấy thông tin SSH từ Vast.ai

Vào Vast.ai:

1. Mở trang **Instances**.
2. Chọn instance đang chạy.
3. Mở phần SSH hoặc **Manage SSH Keys for Instance**.
4. Copy dòng **Direct ssh connect**.

Ví dụ Vast.ai có thể hiện:

```bash
ssh -p 13431 root@115.72.131.186 -L 8080:localhost:8080
```

Trong đó:

- `13431` là SSH port.
- `115.72.131.186` là IP của máy Vast.ai.
- `root` là user đăng nhập.
- `-L 8080:localhost:8080` là tunnel port 8080, thường chỉ cần nếu chạy service web trên port 8080.

Đối với chatbot dùng Ollama, cần thêm port `11434`.

---

## 3. SSH thử lần đầu bằng PowerShell

Mở PowerShell thường trên Windows, sau đó chạy:

```powershell
ssh -p 13431 root@115.72.131.186
```

Lưu ý: thay `13431` và `115.72.131.186` bằng port/IP hiện tại của instance Vast.ai.

Lần đầu kết nối, SSH sẽ hỏi:

```text
The authenticity of host '[115.72.131.186]:13431' can't be established.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Gõ:

```text
yes
```

rồi nhấn Enter.

Vì sao phải làm bước này?

SSH cần lưu fingerprint của máy Vast.ai vào file `known_hosts`. Đây là cách SSH ghi nhớ: "máy IP/port này đã được mình tin tưởng". Nếu bỏ qua bước này, VS Code Remote SSH có thể không hiện được ô hỏi `yes`, rồi báo lỗi:

```text
Host key verification failed.
SSH server closed unexpectedly.
```

Sau khi SSH vào được máy Vast.ai, có thể thoát ra bằng:

```bash
exit
```

---

## 4. Sửa file SSH config trên Windows

Mở file:

```text
C:\Users\DELL\.ssh\config
```

Thêm hoặc sửa thành:

```sshconfig
Host VuTrongNghia
    HostName 115.72.131.186
    User root
    Port 13431
    LocalForward 11434 localhost:11434
    LocalForward 8080 localhost:8080
```

Ý nghĩa từng dòng:

- `Host VuTrongNghia`: tên ngắn để kết nối. Sau này chỉ cần chạy `ssh VuTrongNghia`.
- `HostName 115.72.131.186`: IP máy Vast.ai.
- `User root`: user đăng nhập.
- `Port 13431`: SSH port do Vast.ai cấp.
- `LocalForward 11434 localhost:11434`: tunnel Ollama, rất quan trọng cho chatbot.
- `LocalForward 8080 localhost:8080`: tunnel thêm port 8080, chỉ cần nếu dùng service trên Vast.ai ở port 8080.

Nếu Vast.ai cấp instance mới, IP và port thường sẽ đổi. Khi đó chỉ cần sửa:

```sshconfig
HostName <IP_MOI>
Port <PORT_MOI>
```

Ví dụ Vast.ai cấp lệnh mới:

```bash
ssh -p 32523 root@120.238.149.205
```

thì sửa config thành:

```sshconfig
Host VuTrongNghia
    HostName 120.238.149.205
    User root
    Port 32523
    LocalForward 11434 localhost:11434
    LocalForward 8080 localhost:8080
```

---

## 5. Kết nối bằng tên ngắn

Sau khi có config, mở PowerShell và chạy:

```powershell
ssh VuTrongNghia
```

Nếu vào được terminal của máy Vast.ai là SSH đã thành công.

Nếu muốn mở bằng VS Code:

1. Mở tab Remote Explorer.
2. Chọn SSH.
3. Bấm vào `VuTrongNghia`.
4. VS Code sẽ mở một cửa sổ remote vào máy Vast.ai.

---

## 6. Cài Ollama trên Vast.ai

Sau khi SSH vào máy Vast.ai, chạy:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Sau đó khởi động Ollama:

```bash
ollama serve &
```

Kiểm tra Ollama đã chạy chưa:

```bash
ollama list
```

Vì sao cần `ollama serve &`?

- `ollama serve` mở server Ollama ở port `11434`.
- Dấu `&` giúp tiến trình chạy nền, để terminal vẫn tiếp tục dùng được.
- Nếu Ollama chưa chạy, tunnel vẫn mở nhưng gọi `localhost:11434` sẽ bị lỗi connection refused.

---

## 7. Tải model cần dùng

Trên máy Vast.ai, chạy lần lượt:

```bash
ollama pull bge-m3
ollama pull qwen3:4b-instruct
ollama pull qwen2.5vl:3b
```

Kiểm tra lại:

```bash
ollama list
```

Các model cần thấy:

```text
bge-m3
qwen3:4b-instruct
qwen2.5vl:3b
```

Vì sao tải model trên Vast.ai?

Model cần GPU để chạy nhanh. Laptop chỉ gửi request qua tunnel, còn phần xử lý nặng nằm trên máy Vast.ai.

---

## 8. Mở SSH tunnel cho Ollama

Có 2 cách.

### Cách 1: Dùng file config

Nếu file `C:\Users\DELL\.ssh\config` đã có dòng:

```sshconfig
LocalForward 11434 localhost:11434
```

thì chỉ cần chạy:

```powershell
ssh VuTrongNghia
```

Giữ terminal này mở trong lúc dùng chatbot.

### Cách 2: Dùng lệnh trực tiếp

Nếu không dùng file config, chạy:

```powershell
ssh -p 13431 root@115.72.131.186 -L 11434:localhost:11434
```

Nếu cần cả port 8080:

```powershell
ssh -p 13431 root@115.72.131.186 -L 11434:localhost:11434 -L 8080:localhost:8080
```

Nếu chỉ muốn mở tunnel, không muốn vào shell remote:

```powershell
ssh -p 13431 root@115.72.131.186 -L 11434:localhost:11434 -N
```

Ý nghĩa `-N`:

- Chỉ mở tunnel.
- Không mở terminal shell.
- Terminal sẽ đứng im, đó là bình thường.

---

## 9. Test tunnel từ máy local

Mở một PowerShell khác trên laptop, chạy:

```powershell
curl http://localhost:11434/api/tags
```

Nếu thành công, sẽ thấy JSON có dạng:

```json
{
  "models": []
}
```

hoặc có danh sách model:

```json
{
  "models": [
    ...
  ]
}
```

Nếu thấy danh sách model, nghĩa là:

- Laptop gọi được `localhost:11434`.
- Tunnel SSH đang hoạt động.
- Ollama trên Vast.ai đang trả lời.

---

## 10. Chạy chatbot local

Sau khi tunnel Ollama đã hoạt động, quay về project local và chạy các phần còn lại:

```powershell
docker-compose up -d
```

Sau đó chạy backend:

```powershell
python main.py
```

Mở web:

```text
http://localhost:8000
```

---

## 11. Xử lý lỗi thường gặp

### Lỗi 1: Host key verification failed

Log thường gặp:

```text
The authenticity of host '[115.72.131.186]:13431' can't be established.
Host key verification failed.
SSH server closed unexpectedly.
```

Cách sửa:

```powershell
ssh-keygen -R "[115.72.131.186]:13431"
ssh -p 13431 root@115.72.131.186
```

Khi được hỏi:

```text
Are you sure you want to continue connecting?
```

gõ:

```text
yes
```

Vì sao phải làm vậy?

SSH đang nghi ngờ fingerprint cũ không khớp với máy hiện tại. Vast.ai có thể cấp lại IP/port hoặc máy khác, nên fingerprint thay đổi. Lệnh `ssh-keygen -R` xóa fingerprint cũ khỏi `known_hosts`, sau đó SSH lại để lưu fingerprint mới.

### Lỗi 2: VS Code không hiện ô hỏi yes/no

Log có thể có:

```text
debug1: read_passphrase: can't open conin$: No such device or address
```

Cách sửa:

Không xử lý trong VS Code trước. Hãy mở PowerShell thường và chạy:

```powershell
ssh -p 13431 root@115.72.131.186
```

Gõ `yes` để lưu host key. Sau đó quay lại VS Code Remote SSH.

Vì sao?

VS Code đôi khi không chuyển được prompt xác nhận host key. PowerShell thường xử lý bước này rõ ràng hơn.

### Lỗi 3: channel open failed: connect failed: Connection refused

Lỗi thường gặp:

```text
channel open failed: connect failed: Connection refused
```

Nguyên nhân thường là Ollama chưa chạy trên Vast.ai.

Cách sửa:

SSH vào Vast.ai:

```powershell
ssh VuTrongNghia
```

Rồi chạy:

```bash
ollama serve &
```

Test lại từ máy local:

```powershell
curl http://localhost:11434/api/tags
```

### Lỗi 4: Model not found

Nếu chatbot báo thiếu model, SSH vào Vast.ai và kiểm tra:

```bash
ollama list
```

Nếu thiếu model thì tải lại:

```bash
ollama pull bge-m3
ollama pull qwen3:4b-instruct
ollama pull qwen2.5vl:3b
```

### Lỗi 5: SSH vào instance cũ

Nếu Vast.ai cấp instance mới, file config cũ có thể vẫn trỏ về IP/port cũ.

Mở:

```text
C:\Users\DELL\.ssh\config
```

Sửa lại:

```sshconfig
HostName <IP_MOI>
Port <PORT_MOI>
```

IP/port mới lấy từ dòng **Direct ssh connect** trên Vast.ai.

---

## 12. Checklist nhanh mỗi lần thuê máy mới

Mỗi lần thuê instance mới trên Vast.ai, làm checklist này:

```text
[ ] Copy Direct ssh connect từ Vast.ai.
[ ] Lấy IP và port mới.
[ ] Sửa C:\Users\DELL\.ssh\config.
[ ] Chạy PowerShell: ssh VuTrongNghia.
[ ] Nếu hỏi authenticity, gõ yes.
[ ] Trên Vast.ai chạy: ollama serve &
[ ] Kiểm tra: ollama list.
[ ] Nếu thiếu model, chạy ollama pull.
[ ] Giữ SSH/tunnel mở.
[ ] Trên local test: curl http://localhost:11434/api/tags.
[ ] Chạy docker-compose up -d.
[ ] Chạy python main.py.
[ ] Mở http://localhost:8000.
```

---

## 13. Mẫu config chuẩn

Mẫu nên dùng trong:

```text
C:\Users\DELL\.ssh\config
```

```sshconfig
Host VuTrongNghia
    HostName <IP_VASTAI>
    User root
    Port <PORT_VASTAI>
    LocalForward 11434 localhost:11434
    LocalForward 8080 localhost:8080
```

Ví dụ thực tế:

```sshconfig
Host VuTrongNghia
    HostName 115.72.131.186
    User root
    Port 13431
    LocalForward 11434 localhost:11434
    LocalForward 8080 localhost:8080
```

Sau đó kết nối bằng:

```powershell
ssh VuTrongNghia
```

---

## 14. Câu thần chú cần nhớ

Nếu chỉ nhớ một dòng, hãy nhớ dòng này:

```powershell
ssh -p <PORT_VASTAI> root@<IP_VASTAI> -L 11434:localhost:11434
```

Trong đó:

- `<PORT_VASTAI>` lấy từ Vast.ai.
- `<IP_VASTAI>` lấy từ Vast.ai.
- `11434` là port Ollama.

Nếu dùng file config thì chỉ cần:

```powershell
ssh VuTrongNghia
```

