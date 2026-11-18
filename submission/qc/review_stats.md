# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 531; đối tượng sau duyệt: 487; ngưỡng accepted: IoU >= 0.9
- Đã **loại 23 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 480 | 480/531 = 90.4% | 531 box máy |
| edited (sửa) | 4 | 4/531 = 0.8% | 531 box máy |
| deleted (xoá) | 47 | 47/531 = 8.9% | 531 box máy |
| added (vẽ thêm) | 3 | 3/487 = 0.6% | 487 đối tượng sau duyệt |

Trong số deleted + added, **2 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 1.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 283 | 0 | 30 | 0 |
| bicycle | 4 | 2 | 0 | 0 |
| motorbike | 116 | 1 | 3 | 2 |
| car | 47 | 0 | 2 | 0 |
| bus | 3 | 0 | 0 | 0 |
| truck | 5 | 0 | 0 | 1 |
| three_wheeler | 22 | 1 | 12 | 0 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 66 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 60 | 23 | 6 | 72.3% (KTC 95%: 62–81%) | 90.9% (KTC 95%: 82–96%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 2/12 = 16.7% (KTC 95%: 5–45%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 0/3
  - wrong_class: 1/3
  - extra: 1/3
  - loose: 0/3
  - lọt: img_018.jpg (missing), img_026.jpg (wrong_class), img_035.jpg (extra), img_021.jpg (missing), img_034.jpg (loose), img_017.jpg (wrong_class), img_013.jpg (loose), img_029.jpg (missing), img_005.jpg (loose), img_006.jpg (extra)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
