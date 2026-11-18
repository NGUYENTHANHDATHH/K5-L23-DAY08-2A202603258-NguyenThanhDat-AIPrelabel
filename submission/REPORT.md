# Báo cáo Day 08 — <<2A202603258 - Nguyễn Thành Đạt>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp                                              | Model + cỡ                                            | Ngưỡng                                                                      | Vì sao (dẫn `run_id` trong `model_log.csv`)                                                      |
| ------------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| person / bicycle / motorbike / car / bus / truck | YOLO26s (`--size s`, chạy CPU)                        | person 0.15 · bicycle 0.3 · motorbike 0.3 · car / bus / truck: mặc định 0.3 | Ngưỡng từ `threshold_sweep` (xem `qc/threshold_sweep.md`). Tour: r01, r02 trong `model_log.csv`. |
| three_wheeler                                    | YOLOE-s (`yoloe-26s-seg.pt`), prompt: `three_wheeler` | 0.2                                                                         | Tour: r03 trong `model_log.csv`.                                                                 |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,bicycle=0.3,motorbike=0.3,three_wheeler=0.2" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` → 630 box (person 366, motorbike 138, car 59, three_wheeler 44, bicycle 12, truck 7, bus 4) trên 28 ảnh (24 batch + 4 gold).
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 (`sam2.1_b.pt`), đã chạy ở r04 trên CPU với 10,4 giây/ảnh. Lý do khác (chất lượng mask, RAM): <<bạn điền>>
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? YOLO26 không có lớp `three_wheeler`: tuk-tuk xanh img_002 không có box nào ở cả 0.25 lẫn 0.5 (r01, r02); xích lô img_007 bị gọi `motorbike` (r01). Sửa bằng model open-vocab YOLOE cho riêng `three_wheeler` (r03) rồi gộp bằng `merge_prelabels.py`, và người duyệt đổi nhãn xích lô từ `motorbike` sang `three_wheeler`.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm thấp mà đúng: person 0.25 / 0.31 / 0.33 (người bị che sau tuk-tuk, img_002) mất khi lên 0.5 (r02). Điểm khá mà sai: YOLOE `three_wheeler` 0.40 và 0.37 x2 ở img_002 là quầy hàng / mặt tiền / pickup (r03). Box đúng lớp nhưng lỏng: tuk-tuk 0.93 img_002 lấn sang người lái xe máy.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: <<>>

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số   | Tỉ lệ | Mẫu số                  |
| ---------- | ---- | ----- | ----------------------- |
| accepted   | 480 | 90,4% | 531 box máy |
| edited     | 4 | 0,8% | 531 box máy |
| deleted    | 47 | 8,9% | 531 box máy |
| added      | 3 | 0,6% | 487 đối tượng sau duyệt |

Nguồn: `qc/review_stats.md` (đã loại 23 đối tượng quanh bẫy). Theo lớp: person bị xoá 30/313 box, three_wheeler bị xoá 12/35 box (tỉ lệ xoá cao nhất, 34%).

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** 72,3% (KTC 95%: 62–81%) / 90,9% (KTC 95%: 82–96%); TP 60, FP 23, FN 6 (mẫu số: 66 đối tượng gold trên 4 ảnh). Lạc quan vì ngưỡng chọn trên chính 4 ảnh này.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 2/12 = 16,7% (KTC 95%: 5–45%; ngưỡng đạt 80%); loại nào lọt: missing 3/3 lọt, loose 3/3 lọt, wrong_class 2/3 lọt, extra 2/3 lọt (10 bẫy lọt: img_005, 006, 013, 017, 018, 021, 026, 029, 034, 035); vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 30 phút / 4 ảnh gold = 7,5 phút/ảnh; duyệt 20 phút / 24 ảnh batch ≈ 0,8 phút/ảnh (guideline gợi ý ~3 phút/ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh                                    | SAM (phiên bản)       | Số click                                               | Mask gồm người lái / hàng hoá?                                                             | Lỗi quan sát                                                                                                                                    | File trong `masks/`                          |
| -------------------------------------- | --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| img_002 (tuk-tuk xanh)                 | SAM 2.1 `sam2.1_b.pt` | 3 (2 dương thân + mui, 1 âm người lái xe máy bên phải) | Không gồm tài xế tuk-tuk (nằm ngoài mask); khoang khách lẫn người đi bộ phía sau qua khung | Mask bám thân, mui, bánh tốt; vùng khoang hở lẫn nền phía sau                                                                                   | `tuktuk_img002.jpg`                          |
| img_032 (xích lô đỗ, tiền cảnh)        | SAM 2.1 `sam2.1_b.pt` | 4 (3 dương: đệm, gầm, bánh; 1 âm người đứng cạnh)      | Không có người lái; gồm đệm / vải trên ghế                                                 | Mask trọn khung, 3 bánh, đệm; không lan sang người đứng cạnh nhờ click âm                                                                       | `xichlo_img032.jpg`                          |
| img_007 (xích lô đỗ bên trái, xa, nhỏ) | SAM 2.1 `sam2.1_b.pt` | 2 dương (lần 1) → 3 dương + box (lần 2)                | Không                                                                                      | Lần 1 chỉ ra **đệm ghế đỏ**, mất khung và bánh (0,7% ảnh). Lần 2 thêm box: phủ đủ hơn nhưng **lan sang xích lô đỗ sát bên** (2 xe thành 1 mask) | `xichlo_img007.jpg`, `xichlo_img007_box.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- _(Tuỳ chọn)_ Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): claude, dùng để tham khảo và đọc tài liệu
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng: Windows 11, CPU (không GPU), Python 3.12, CVAT Docker (server 2.74.1). Vướng mắc: PowerShell chặn script (đã đổi `RemoteSigned`); `cvat-cli` báo lệch phiên bản SDK 2.83 / server 2.74.1; hàm auto-annotate báo `motorbike` không có trong Task 21 (Project thiếu nhãn). <<bạn điền cách đã xử lý>>
- Dùng cache `data/cache/` thay vì tự chạy? không (đã tự chạy các lệnh model trên máy).
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
