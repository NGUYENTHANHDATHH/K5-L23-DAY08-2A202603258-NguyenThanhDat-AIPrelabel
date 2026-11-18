# Báo cáo Day 08 — 2A202603258 - Nguyễn Thành Đạt

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp                                              | Model + cỡ                                            | Ngưỡng                                                                      | Vì sao (dẫn `run_id` trong `model_log.csv`)                                                      |
| ------------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| person / bicycle / motorbike / car / bus / truck | YOLO26s (`--size s`, chạy CPU)                        | person 0.15 · bicycle 0.3 · motorbike 0.3 · car / bus / truck: mặc định 0.3 | Ngưỡng từ `threshold_sweep` (xem `qc/threshold_sweep.md`). Tour: r01, r02 trong `model_log.csv`. |
| three_wheeler                                    | YOLOE-s (`yoloe-26s-seg.pt`), prompt: `three_wheeler` | 0.2                                                                         | Tour: r03 trong `model_log.csv`.                                                                 |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.15,bicycle=0.3,motorbike=0.3,three_wheeler=0.2" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` → 630 box (person 366, motorbike 138, car 59, three_wheeler 44, bicycle 12, truck 7, bus 4) trên 28 ảnh (24 batch + 4 gold).
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 (`sam2.1_b.pt`), đã chạy ở r04 trên CPU với 10,4 giây/ảnh. Lý do khác: bản `b` cân bằng chất lượng / tốc độ trên máy không GPU; mask trọn xe ở img_032 và tuk-tuk img_002 (Part 5) đủ dùng. SAM 3 cần GPU + quyền truy cập nên không chọn; `mobile_sam.pt` nhanh hơn nhưng không cần vì 10 giây/ảnh vẫn chấp nhận được cho vài vật.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? YOLO26 không có lớp `three_wheeler`: tuk-tuk xanh img_002 không có box nào ở cả 0.25 lẫn 0.5 (r01, r02); xích lô img_007 bị gọi `motorbike` (r01). Sửa bằng model open-vocab YOLOE cho riêng `three_wheeler` (r03) rồi gộp bằng `merge_prelabels.py`, và người duyệt đổi nhãn xích lô từ `motorbike` sang `three_wheeler`.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): điểm thấp mà đúng: person 0.25 / 0.31 / 0.33 (người bị che sau tuk-tuk, img_002) mất khi lên 0.5 (r02). Điểm khá mà sai: YOLOE `three_wheeler` 0.40 và 0.37 x2 ở img_002 là quầy hàng / mặt tiền / pickup (r03). Box đúng lớp nhưng lỏng: tuk-tuk 0.93 img_002 lấn sang người lái xe máy.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng chọn và chấm trên cùng 66 đối tượng gold. Lớp ít vật (bus 1, truck 2) không chọn được ngưỡng riêng; car không ngưỡng nào ≤ 0.5 đạt recall 90% (tối đa 89%, 9 vật) nên dùng mặc định 0.3. Ngưỡng person 0.15 là ước lượng thô từ 33 vật.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số  | Tỉ lệ | Mẫu số                  |
| ---------- | --- | ----- | ----------------------- |
| accepted   | 480 | 90,4% | 531 box máy             |
| edited     | 4   | 0,8%  | 531 box máy             |
| deleted    | 47  | 8,9%  | 531 box máy             |
| added      | 3   | 0,6%  | 487 đối tượng sau duyệt |

Nguồn: `qc/review_stats.md` (đã loại 23 đối tượng quanh bẫy). Theo lớp: person bị xoá 30/313 box, three_wheeler bị xoá 12/35 box (tỉ lệ xoá cao nhất, 34%).

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? `three_wheeler`: 12/35 box bị xoá (34%), cao nhất mọi lớp. YOLOE open-vocab gán `three_wheeler` cho quầy hàng có mái, mặt tiền, pickup (img_002: 0.40, 0.37, 0.37) và gán trùng hai box lên cùng một xích lô (img_035, QC). Ngưỡng 0.2 cho lớp này thấp để giữ recall 100% trên gold, nên đổi lại nhiều box thừa. Thứ hai là `person`: 30/313 box bị xoá do ngưỡng 0.15 (ví dụ ban công bị gọi person ở img_006, QC).
- So với gold: precision / recall của **máy** 72,3% (KTC 95%: 62–81%) / 90,9% (KTC 95%: 82–96%); TP 60, FP 23, FN 6 (mẫu số: 66 đối tượng gold trên 4 ảnh). Lạc quan vì ngưỡng chọn trên chính 4 ảnh này.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 2/12 = 16,7% (KTC 95%: 5–45%; ngưỡng đạt 80%); loại nào lọt: missing 3/3 lọt, loose 3/3 lọt, wrong_class 2/3 lọt, extra 2/3 lọt (10 bẫy lọt: img_005, 006, 013, 017, 018, 021, 026, 029, 034, 035); vì sao (bạn đã bỏ bước nào của checklist)? Duyệt 0,8 phút/ảnh (so với ~3 phút gợi ý) không đủ cho bước 1 "ẩn nhãn máy, quét ảnh trước": nhìn nhãn máy trước và chỉ xoá box sai lộ rõ. Vì vậy bẫy **missing** (cần tự đếm vật trên ảnh) và **loose** (cần kiểm 4 cạnh box) lọt hết 3/3; hai bẫy bắt được là loại thấy ngay trên nhãn máy (extra, wrong_class). Accepted 90,4% (480/531) với bắt bẫy 2/12 cho thấy phần lớn "accepted" là bấm chấp nhận, không phải đã kiểm.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? giữ lại nhiều box nhỏ hơn 20px

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 30 phút / 4 ảnh gold = 7,5 phút/ảnh; duyệt 20 phút / 24 ảnh batch ≈ 0,8 phút/ảnh (guideline gợi ý ~3 phút/ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): gold là 4 ảnh khác batch nhưng cùng loại cảnh, nên lúc duyệt đã quen guideline và lớp; ngoài ra 0,8 phút/ảnh là duyệt hời hợt (bắt bẫy 2/12), nên so sánh 7,5 vs 0,8 phút/ảnh **phóng đại** lợi ích của máy.
- Kết luận: **có, cho person / motorbike / car** (máy so với gold: recall 90,9%, precision 72,3% trên 66 vật; deleted thấp ở motorbike 3/122, car 2/49); **chưa đáng tin cho three_wheeler** (xoá 12/35). Điều kiện: phải duyệt đúng checklist, vì ở tốc độ hiện tại người duyệt không bắt được lỗi máy (2/12 bẫy).
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Bỏ những box nhỏ hơn 20px
  2. Chú ý những box về three_wheel
  3. Chú ý những ảnh có đông người

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh                                    | SAM (phiên bản)       | Số click                                               | Mask gồm người lái / hàng hoá?                                                             | Lỗi quan sát                                                                                                                                    | File trong `masks/`                          |
| -------------------------------------- | --------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| img_002 (tuk-tuk xanh)                 | SAM 2.1 `sam2.1_b.pt` | 3 (2 dương thân + mui, 1 âm người lái xe máy bên phải) | Không gồm tài xế tuk-tuk (nằm ngoài mask); khoang khách lẫn người đi bộ phía sau qua khung | Mask bám thân, mui, bánh tốt; vùng khoang hở lẫn nền phía sau                                                                                   | `tuktuk_img002.jpg`                          |
| img_032 (xích lô đỗ, tiền cảnh)        | SAM 2.1 `sam2.1_b.pt` | 4 (3 dương: đệm, gầm, bánh; 1 âm người đứng cạnh)      | Không có người lái; gồm đệm / vải trên ghế                                                 | Mask trọn khung, 3 bánh, đệm; không lan sang người đứng cạnh nhờ click âm                                                                       | `xichlo_img032.jpg`                          |
| img_007 (xích lô đỗ bên trái, xa, nhỏ) | SAM 2.1 `sam2.1_b.pt` | 2 dương (lần 1) → 3 dương + box (lần 2)                | Không                                                                                      | Lần 1 chỉ ra **đệm ghế đỏ**, mất khung và bánh (0,7% ảnh). Lần 2 thêm box: phủ đủ hơn nhưng **lan sang xích lô đỗ sát bên** (2 xe thành 1 mask) | `xichlo_img007.jpg`, `xichlo_img007_box.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?
- _(Tuỳ chọn)_ Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline:

## 6. Calibration (Part 6)

- Số quyết định: 5 (`calibration_decisions.csv`: 2 `loi_may`, 2 `cau_hoi_mo`, 1 `bay_lot`); quan trọng nhất: **CAL-02** (box `three_wheeler` chỉ giữ khi thấy bánh xe / ghế khách), vì lớp này bị xoá 34% (12/35) và là lý do lab tồn tại (model COCO không biết xe ba bánh).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: xích lô đỗ không người vẫn là `three_wheeler` (CAL-01); quầy hàng có mái không phải `three_wheeler` (CAL-02); khối xe máy gần camera: từng xe khi ≥ 20 px và tách được biên, còn lại `ignore_region` (CAL-03, câu mở 2); không dùng `conf` để bỏ qua kiểm (CAL-04, câu mở 7). Câu mở 1, 3, 4, 5, 6 chưa chốt.
- Nguyên nhân gốc: **máy** (YOLO26 không có lớp xe ba bánh; YOLOE nhầm vật có mái) + **guideline mơ hồ** (câu 2, 7) + **người duyệt** bỏ bước quét ảnh (CAL-05, không sửa guideline).

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên: img_003, 006, 014, 031, 035; seed mặc định của `qc_sample.py`), 99 đối tượng; người kiểm: **Claude (AI)** đọc ảnh `work/qc/review/*.jpg`, không phải người duyệt lô (**hạn chế**: AI kiểm bằng mắt trên ảnh thu nhỏ, có thể sót vật nhỏ; không có người kiểm thứ hai).
- Lỗi theo **đối tượng**: 6/99 = 6,1% · theo **ảnh**: 5/5 = 100% (ngưỡng Release ≤ 2%)
- Top lỗi: **thừa** 4 (img_006 ban công gọi `person`; img_035 mặt tiền gọi `motorbike` và xích lô gán 2 box; img_003 `three_wheeler` bao vùng người + xe máy) · **sai vị trí / kích thước** 2 (img_014 box person ôm cả quang gánh; img_031 box person cụt ở thắt lưng). Không thấy sai lớp, bỏ sót trong mẫu (nhưng bỏ sót khó thấy trên ảnh có box sẵn).
- **Giới hạn mẫu nhỏ**: 5/5 ảnh lỗi, cận trên 95% (Wilson) = 100%; với 5 ảnh không thể chứng minh ngưỡng 2%, nhưng 5/5 đủ để bác bỏ Release.
- Script kiểm được / không kiểm được: script kiểm được định dạng, lớp lạ, box < 20 px (4 box: img_017, 018, 027, 034), đếm bẫy và tỉ lệ; **không** kiểm được box ôm sát hay không, vật mà cả máy lẫn người cùng bỏ sót, và người duyệt có làm theo checklist thật không (chỉ suy ra gián tiếp từ thời gian 0,8 phút/ảnh và bẫy 2/12).

**Quyết định: Hold**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release): Hold vì guideline còn 5 câu mở chưa chốt (1, 3, 4, 5, 6) nên chưa có chuẩn để sửa lô. Kể cả khi chốt xong, lô vẫn **không đạt Release**: ảnh lỗi 5/5 (ngưỡng ≤ 2%), lỗi `thua` lặp lại (lỗi hệ thống), bắt bẫy 2/12 (ngưỡng ≥ 10/12) → bước sau sẽ là Rework.
Nếu Rework / Hold, việc cần làm tiếp: (1) chốt các câu mở còn lại trong v1.1; (2) duyệt lại cả 24 ảnh theo checklist, ẩn nhãn máy và đếm vật trước, ≥ 2 phút/ảnh; ưu tiên xoá box `three_wheeler` không thấy bánh và box < 20 px; (3) cài bẫy mới (seed khác) để đo lại, đạt ≥ 10/12; (4) QC lại với mẫu lớn hơn (≥ 10 ảnh).

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): claude, dùng để tham khảo và đọc tài liệu, viết tài liệu. Cụ thể: Claude (Claude Code) đọc tài liệu và đưa lệnh; chạy các script (`threshold_sweep`, `merge_prelabels`, `inject_traps`, `review_stats`, `build_release`, `qc_sample`, `qc_report`); viết nhận xét `model_log.csv` từ ảnh `viz/`; chọn điểm click và chạy SAM ở Part 5; **kiểm QC 5 ảnh** (`qc_review.csv`, reviewer = claude); soạn `calibration_decisions.csv`, `guideline_v1.1_calibrated.md` và phần lớn REPORT từ số liệu trong `qc/`. Tôi tự làm: gán tay gold, duyệt 24 ảnh batch trên CVAT, ba quy tắc ở mục 4. Kiểm tra: đối chiếu số trong REPORT với `qc/*.md`; `check_submission.py`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không: gán gold trước khi chạy model nào lên ảnh; lần thử `cvat-cli auto-annotate` trên Task 21 trước đó bị lỗi nhãn nên không tạo ra nhãn máy nào.
- Máy / GPU / CVAT dùng: Windows 11, CPU (không GPU), Python 3.12, CVAT Docker (server 2.74.1). Vướng mắc: PowerShell chặn script (đã đổi `RemoteSigned`); `cvat-cli` báo lệch phiên bản SDK 2.83 / server 2.74.1; hàm auto-annotate báo `motorbike` không có trong Task 21 (Project thiếu nhãn). Xử lý: gọi trực tiếp `.venv\Scripts\python.exe` / đổi execution policy; bỏ đường `cvat-cli`, dùng cách A (chạy `prelabel.py` trên máy rồi Upload annotations COCO 1.0); Project dùng bộ nhãn `cvat/labels_generated.json` (`validate_labels.py` OK cho gold và batch).
- Dùng cache `data/cache/` thay vì tự chạy? không (đã tự chạy các lệnh model trên máy).
- Lỗi / bug script đã gặp (nếu có): không gặp bug script; chỉ có cảnh báo lệch phiên bản `cvat-sdk` 2.83 với server CVAT 2.74.1 (lab hướng dẫn 2.81).
- Một điều bạn làm khác lần sau: duyệt chậm lại theo đúng checklist (ẩn nhãn máy, đếm vật trước, ≥ 2 phút/ảnh) và bấm giờ từng ảnh, vì 0,8 phút/ảnh cho bắt bẫy 2/12.
