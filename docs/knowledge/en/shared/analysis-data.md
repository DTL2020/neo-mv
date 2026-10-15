# Shared analysis data: vectors, errors, and availability

## Properties carry calculation data

Analysis nodes carry geometry, vectors, and errors in frame properties. The visible image's dimensions or format do not automatically define those data. Other producers can supply valid data too; consumers do not depend on original node identity.

Property names concatenate `prefix` and suffix directly, for example `MVUtensilsAnalysisPel`. Scalars are host int64 elements. Vectors and errors are int64 arrays, not serialized native structures.

## Descriptor fields

| Suffix after `Analysis` | Meaning |
| --- | --- |
| `Width,Height` | Working dimensions without padding; positive and no smaller than real dimensions |
| `RealWidth,RealHeight` | Positive real-image dimensions |
| `HPad,VPad` | Nonnegative luma padding on each side |
| `Pel` | Vector units per pixel: 1, 2, 4 |
| `Levels` | At least 1; analysis level count, not multiple exported grids |
| `Chroma` | Whether chroma contributes to error; zero/nonzero |
| `XRatioUV,YRatioUV` | Each 1 or 2; GRAY uses 1, 1 |
| `BlkSizeX,BlkSizeY` | Each at least 2; functions add restrictions |
| `OverlapX,OverlapY` | Each 0 through half a block |
| `NBlkX,NBlkY` | Positive grid dimensions |
| `DeltaFrame` | d in reference frame `n+d`; shared format permits 0, though Analyse does not generate it |
| `BitsPerSample` | Integer 8–16, or 32 for floating analysis |

Read scalar element 0; missing, empty, or wrongly typed scalars initially become zero. Ignore later elements. Except for `Chroma`, which uses full int64 zero/nonzero truth, saturate to int32 before validation. Missing `Chroma` can give valid false; missing `Pel` gives invalid zero.

## Decoding arrays

`AnalysisSAD` stores the selected luma metric (SAD, SATD, DCT, or a mixture), plus chroma SAD when enabled. There is no metric identifier in the descriptor. Consumers use these stored errors directly for thresholds, weights, and masks; they do not recompute pixel SAD or convert the errors to a common metric. Recalculate chooses its own metric for fresh measurements.

Count `N=NBlkX·NBlkY`, row-major index `i=by·NBlkX+bx`. `AnalysisVectors` stores signed X in the low 32 bits and signed Y in the high 32 bits. `AnalysisSAD` stores nonnegative raw error. See [Analyse](../analyse.md) for packing and units.

For padded block origin `x0=hp+bx(Bx-Ox),y0=vp+by(By-Oy)`, vectors must satisfy:

$$-px_0\le v_x<p(W+2h_p-x_0-B_x),\quad
-py_0\le v_y<p(H+2v_p-y_0-B_y).$$

Upper bounds are exclusive. Shared range validation does not replace renderer-specific footprint checks.

## Raw error and user threshold units

Search comparison cost may include predictor-distance and candidate penalties. `AnalysisSAD` contains the winning candidate's freshly measured raw error, without those penalties. Recalculate remeasures even when its search threshold prevents refinement; it neither copies the old error nor inherits the producer's metric.

For integer SAD, a constant difference d over area A contributes A·d in native sample code values. There is no automatic division by bit depth. Enabled chroma adds two separate SADs at their actual subsampled sizes. An 8×8 block differing by 1 on every plane therefore gives 64 for GRAY, 96 for YUV420, 128 for YUV422, and 192 for YUV444. These values are the same for a difference of one code value at 8, 10, or 16 bits; equal normalized brightness changes need larger code-value differences at higher precision.

Integer SATD sums each 4×4 Hadamard cell's absolute coefficients divided by 2. Constant difference 1 gives 32 over an 8×8 luma block; a single differing sample gives SAD=1 and SATD=8. Chroma still contributes pixel SAD. Float SAD/SATD encode each complete plane error separately on the 65535 scale before adding planes; see the float error section in [Analyse](../analyse.md).

These exact GRAY8 8×8 examples distinguish selected metrics:

| Constant source/reference | SAD | SATD | DCT (DC weight 4) | Local SAD/DCT (DC weight 1) | Local SAD/SATD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 1 | 64 | 32 | 0 | 32 | 48 |
| 0 / 4 | 256 | 128 | 32 | 132 | 192 |

Both local examples use weight=0.5 and threshold=0. DCT quantizes each block independently: DC uses integer `sum/(2A)`, so 0 and 1 have the same quantized DC. Global Analyse with only one level uses SAD. On the 0/4 example, Recalculate's fixed global weight gives 144 for SAD/DCT, 192 for SAD/SATD, and 224 for the half-global SATD mixture. Neither a mode name nor a smaller raw error implies a better motion estimate.

For Compensate/Degrain `thsad` and scene `thscd1`, let A be luma block area, C be 1 without analysis chroma or `1+2/(rx·ry)` with it, and D be `(2^min(bits,16)-1)/255`. A nonnegative user threshold t scales to:

```text
T = floor(t · (A/64) · C · D + 0.5)
```

The analysis descriptor supplies bits and chroma, even when the renderer uses a clip of another precision. For GRAY 8×8, t=400 yields T=400 at 8 bits, 1605 at 10 bits, and 102800 for float analysis. This scales the user threshold's units; it does not convert SATD, DCT, or mixed errors into SAD.

Recalculate preserves a different rounding order: first round precision as `trunc(t·D+0.5)`, then multiply by A and truncate division by 64, then, if chroma is enabled, add `2·trunc(T/(rx·ry))`. For a 4×4 YUV420 block at 10 bits and t=1, its threshold is 1 while the render/scene threshold is 2. Do not consolidate these formulas as an arithmetic cleanup.

| Consumer or decision | At error E equal to threshold T |
| --- | --- |
| Compensate block | Uses the current block; reference requires E<T |
| Degrain reliability | Zero; a reference contributes only below T |
| Scene bad-block count, also used by Flow | Block is not bad; bad requires E>T |
| Recalculate refinement | Keeps the mapped vector and newly measured error; search requires E>T |

A scene cut additionally requires bad-block count to be strictly greater than the percentage-derived count threshold. These equality rules are independent of the selected metric.

## Three states and malformed data

1. Invalid scalar values give **InvalidMetadata**, without reading arrays.
2. Valid metadata with either array absent or not of count N gives **MetadataOnly**. Counts are checked before types or elements.
3. With both counts correct, validate integer types, nonnegative SAD, and vector bounds. Success gives **Complete**. Malformed elements here are errors, not missing data.

Metadata-only operations stop after scalars and do not certify arrays. Complete zero vectors and errors are valid measurements, distinct from missing arrays.

For N=4, vector count 4 and error count 3 give MetadataOnly even if the short array is floating-point. Both counts 4 with floating-point errors instead fail.

## Scene detection and reference availability

[SCDetection](../sc-detection.md) explains scaling error thresholds by area, chroma, and depth, then counting bad blocks. Many consumers reuse it: invalid metadata, missing arrays, or a scene change can make a reference unavailable; complete malformed data remain errors.

Functions respond differently. Analyse omits arrays for out-of-range references; Recalculate may start from zero vectors when old arrays are absent and measure again; renderers may fall back to the center image. The shared decoder does not impose universal zero filling.

[Back to the English index](../README.md)
