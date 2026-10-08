# Full 3D-ADAM Dataset — Paper Protocol Feasibility (read-only)

Retrieved: 2026-10-08  
Dataset: https://huggingface.co/datasets/pmchard/3D-ADAM  
License: CC BY-NC-SA 4.0  
Anomalib subset (separate): https://huggingface.co/datasets/pmchard/3D-ADAM_anomalib  

**No full download was performed.** Inspection used Hugging Face dataset tree API + README only.

## Repository tree (top-level)

- ~216 part-instance directories (e.g. `1M1_A_Good`, `1M1_A_Bulge`, `RackGear_…`)
- Files at root: `README.md`, `.gitattributes` only
- Unique part-family prefixes observed: **28** (matches paper category count narrative)

## Directory structure (sample)

```
{PartInstance}_{View?}_{Label}/
  MechMind-LSRL/
  MechMind-Nano/
  RealSense/
  Zed/
    images/
    xyz/
    masks/
    binary_masks/
    defect_masks/          # defective instances
    bounding_box/
    machine_elements/…     # fine-grained annotations
```

File naming example: `1M1_A_Bulge_Nano_000_image.png` under sensor-specific folders.

## Sensors

Four sensors per part-instance folder:

| Sensor | Notes |
|--------|-------|
| MechMind-Nano | Used by public anomalib subset (README) |
| MechMind-LSRL | Present in full tree |
| RealSense | Present |
| Zed | Present |

**Table V / anomalib subset sensor:** MechMind-Nano (subset README). Paper Table V may aggregate or select sensors — **not confirmed** from public README alone.

## Correspondence to anomalib subset

| Full HF | Anomalib subset |
|---------|-----------------|
| Part-instance folders with readable IDs | Flattened `datasets/ADAM3D/<cat>/{train,test,validation}/…` with many numeric stems |
| Multi-sensor | Single-sensor (Nano) packaging |
| Defect masks under `defect_masks` | `ground_truth` PNGs; 28 empty abnormal masks audited |

Mapping Nano scans → subset stems is **not published as a manifest** in the API listing; reconstructing exact anomalib hold-out would require additional author scripts or hashing file contents after download.

## Paper protocol reconstructability

Paper (Sec. IV-A): all defect-free scans in train; defective part-instances split test:validation = 60:40 without view leakage.

| Requirement | Full HF evidence |
|-------------|------------------|
| Part-instance identity | **Yes** — folder names encode instance |
| Defect vs good | **Yes** — `_Good` vs defect-type suffixes |
| View / pose separation | **Likely** — `A/B/C/D` and `A1/B1…` tokens in folder names (interpretation still assumed) |
| Official 60:40 assignment file | **Not found** in public tree (no split CSV/JSON at root) |
| Sensor filter for Table V | **Partial** — Nano identifiable; paper table sensor scope unclear |

### Verdict

**`paper protocol may be reconstructed but metadata is missing`**

Rationale: part-instance structure enables a *defensible* 60:40 split implementation, but no author-published split manifest / seed / exact membership list was found in the public tree. Any project-built split must be labeled **project reconstruction**, not paper-official.

## Capacity

- README / HF card: raw complete dataset is large (project notes historically ~669 GB — **reconfirm before download**)
- Do not auto-download in this phase

## Next steps (not executed)

1. Author confirmation of Table V sensor + split seed/file
2. Optional: download **MechMind-Nano only** if HF supports filtered fetch
3. Build `configs/splits/paper_reconstructed_manifest.csv` with `paper_protocol_equivalent=false` until author sign-off
