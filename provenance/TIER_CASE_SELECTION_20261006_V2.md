# Tier-specific qualitative case selection — 2026-10-06

Status: PASS WITH LIMITATIONS. No inference; benchmark values unchanged.

หนึ่ง shared anchor (Case 2: MOTS20-09 / 000263) และอีกสาม diagnostic cases ตามพฤติกรรมแต่ละ tier ไม่บังคับภาพทั้งหมดตรงกันข้าม tier ภายใน case ทุกโมเดลใช้ source frame/region/scale/policy เดียวกัน

| Tier | Case 1 | Case 2 (anchor) | Case 3 | Case 4 | Changed |
|---|---|---|---|---|---|
| Large | 09/000525 | 09/000263 | 02/000600 | 11/000450 | 1,4 |
| Second_Largest | 05/000419 | 09/000263 | 02/000001 | 11/000001 | 4 |
| Medium | 05/000419 | 09/000263 | 02/000001 | 11/000450 | 4 |
| Small | 02/000300 | 09/000263 | 02/000600 | 11/000900 | 4 |
| Nano | 11/000001 | 09/000263 | 02/000300 | 09/000001 | 1 |

20 slots ใช้ original frames ต่างกัน 10 เฟรม จากเดิม 6; เปลี่ยน 6 case slots สร้างเฉพาะ 6 comparison + 6 ROI ใหม่จาก saved predictions และ reuse อีก 14 cases เพิ่ม MOTS20-11 เพื่อแสดง GT หลังราวและ extra masks แต่ยังเลือกจาก pool 12 เฟรม ไม่อ้าง full-dataset extrema หรือ representative sample

Shared diagnostics ยังใช้ได้เมื่อ error เดียวกันตรวจคนละโมเดล: 11/1 ของ L ตรวจ GT 2016 ขณะที่ N ตรวจ GT 2028/extra GT 2006; 11/450 ของ Largest/M ตรวจ coverage เท่ากันกับ extra masks ที่ต่างกัน ข้อมูลซ้ำไม่เพิ่ม independent sample count เหตุผลทั้งการเลือก/แทน case อยู่ใน CASE_SELECTION.md ของแต่ละ repo

## สิ่งที่ตรวจจากภาพจริง

- Largest: GT 2003 กับ 2004 match คนละโมเดล แม้ TP เท่ากัน; near-tied 11x/v9e เก็บครบเหมือนกันแต่ FP ต่างกัน
- Second-largest: v9c เก็บ GT 2016 ที่ v8l พลาด แม้ Recall รวม v9c ต่ำกว่า; คง FN ร่วมและ counterexample ไว้
- Medium: TP 9 / FN 0 เท่ากัน แต่ extra masks ทับ GT 2010 ที่ได้คู่แล้วต่างกัน
- Small: near-tied 11s/v8s พลาด GT 2065 ต่างกันและ FP ต่างกัน; 26s เก็บเพิ่มแต่ยังมี FP/common FN
- Nano: 26n/11n match GT ชุดเดียวกัน แต่ 11n เพิ่ม mask บน GT 2006; v8n มี candidate GT 2028 ที่ IoU 0.476 ไม่ผ่านเกณฑ์

Full-frame images retained; ROI coordinates/GT matching/FP overlap inspected. FP is unmatched under one-to-one and ignore policy, not automatically a nonexistent person. Cases do not explain latency/VRAM or establish statistical/CCTV superiority. Old artifacts/summary archives preserved; active pointers identify selection v2.

[Provenance and protected hashes](TIER_CASE_SELECTION_20261006_V2.json) · [Prior review](CASE_DECISION_REVIEW_20261006.md)
