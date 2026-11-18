# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603258 - Nguyễn Thành Đạt

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Gán từng xe khi box cao ≥ 20 px **và** tách được biên với xe bên cạnh. Phần còn lại của khối (≥ 5 xe không tách được) vẽ một `ignore_region`. Áp dụng như nhau cho gần hay xa camera. | img_018, img_029 | CAL-03 |
| 7 | Có đặt `conf` tối thiểu để chấp nhận box máy không cần sửa? | **Không đặt.** `conf` chỉ để xếp thứ tự duyệt. Lô này accepted 90,4% (480/531) mà chỉ bắt 2/12 bẫy, 3/3 bẫy box lỏng lọt: box điểm cao vẫn phải kiểm 4 cạnh. | img_013, img_034, img_005 | CAL-04 |

Chưa chốt: câu 1 (xe đạp điện), 3 (hàng nhô ra), 4 (ngưỡng 20% người trong xe có mui), 5 (kính xe trong mask), 6 (box đêm chồng nhau). Lý do: lô này không gặp đủ ca để quyết có ảnh ví dụ.

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_007, img_032 | Xích lô đỗ bên đường, không người | YOLO26 gọi `motorbike` (0.57, 0.37 ở img_007); YOLO26 không có lớp `three_wheeler` | Luôn `three_wheeler`, kể cả khi đỗ, không người, nhìn từ sau | Mục 2: `three_wheeler` gồm xích lô đỗ không người; dấu hiệu nhận: 3 bánh + ghế khách phía trước (CAL-01) |
| img_002 | Quầy hàng có mái, mặt tiền, pickup cạnh tuk-tuk | YOLOE gán `three_wheeler` 0.40, 0.37, 0.37; cả lô batch xoá 12/35 box `three_wheeler` | Chỉ giữ khi thấy bánh xe hoặc ghế khách / thùng xe gắn bánh | Mục 4.3: `three_wheeler` phải có bằng chứng nhìn thấy; quầy hàng có mái **không** phải `three_wheeler` (CAL-02) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu mở 2 (khối xe gần camera) và 7 (không dùng `conf` để bỏ qua kiểm); thêm quy tắc nhận diện `three_wheeler` (xích lô đỗ, loại quầy hàng có mái) | CAL-01, CAL-02, CAL-03, CAL-04 |
