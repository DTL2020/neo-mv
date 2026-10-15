"""Independent raw-error examples and consumer threshold boundaries."""
import vapoursynth as vs

core = vs.core
core.num_threads = 4
core.std.LoadPlugin(path=plugin)
mv = core.neo_mv
P = 'MVUtensils'


def nodes(value):
    return [value] if isinstance(value, vs.VideoNode) else list(value)


def pixels(frame):
    return [bytes(frame[p]) for p in range(len(frame))]


def constant(frame, value):
    plane = frame[0]
    return all(plane[y, x] == value for y in range(plane.shape[0]) for x in range(plane.shape[1]))


def errors(node, expected, n=1):
    f = node.get_frame(n)
    actual = f.props[P + 'AnalysisSAD']
    actual = list(actual) if isinstance(actual, (list, tuple)) else [actual]
    assert all(x == expected for x in actual), (expected, actual)


def constant_pair(fmt, width=32, height=32, difference=1):
    info = core.get_video_format(fmt)
    delta = difference / 65536 if info.sample_type == vs.FLOAT else difference
    zero = core.std.BlankClip(width=width, height=height, length=1, format=fmt,
                              color=[0] * info.num_planes)
    one = core.std.BlankClip(zero, color=[delta] * info.num_planes)
    return core.std.Splice([one, zero, one]), info


# One integer code-value difference stays one code value at every bit depth.
# Float 1/65536 is exact; these small plane totals encode to the same integers.
for fmt in (vs.GRAY8, vs.GRAY10, vs.GRAY16, vs.GRAYS,
            vs.YUV420P8, vs.YUV422P10, vs.YUV444P16, vs.YUV444PS):
    c, info = constant_pair(fmt)
    for bw, bh in ((4, 4), (8, 8), (16, 8)):
        area = bw * bh
        chroma_error = 0 if info.num_planes == 1 else 2 * (area >> (info.subsampling_w + info.subsampling_h))
        s = mv.Super(c, blksize=[bw, bh], overlap=0, pad=16, pel=1, onelevel=True)
        for metric, luma in [('sad', area), ('satd', area // 2)]:
            v = nodes(mv.AnalyseMany(s, radius=1, metric=metric, badrange=0))
            r = nodes(mv.Recalculate(s, v, metric=metric, thsad=0))
            for node in v + r:
                errors(node, luma + chroma_error)
            if info.num_planes == 3:
                errors(mv.Analyse(s, metric=metric, chroma=False, badrange=0), luma)

# At 8-bit GRAY, area 8x8, user render thresholds equal raw-error integers.
# One-level global Analyse uses SAD; Recalculate has fixed global weight 8/16.
unit, _ = constant_pair(vs.GRAY8)
unit_super = mv.Super(unit, blksize=8, overlap=0, pad=16, pel=1, onelevel=True)
errors(mv.Analyse(unit_super, metric='dct', badrange=0), 0)
for name, expected in [('sad_dct_local', 32), ('sad_satd_local', 48)]:
    errors(mv.Analyse(unit_super, metric=name, metric_weight=0.5, metric_threshold=0, badrange=0), expected)
c, _ = constant_pair(vs.GRAY8, difference=4)
s = mv.Super(c, blksize=8, overlap=0, pad=16, pel=1, onelevel=True)
for metric, analysis_error, recalc_error in [
    ('sad', 256, 256), ('satd', 128, 128), ('dct', 32, 32),
    ('sad_dct_local', 132, 132), ('sad_satd_local', 192, 192),
    ('sad_dct_global', 256, 144), ('sad_satd_global', 256, 192),
    ('sad_satd_global_half', 256, 224),
]:
    kw = dict(metric=metric)
    if 'local' in metric:
        kw.update(metric_weight=0.5, metric_threshold=0)
    v = nodes(mv.AnalyseMany(s, radius=1, badrange=0, **kw))
    r = nodes(mv.Recalculate(s, v, thsad=0, **kw))
    for group, expected in ((v, analysis_error), (r, recalc_error)):
        for node in group:
            errors(node, expected)
        for threshold in (expected - 1, expected, expected + 1):
            comp = mv.Compensate(c, s, group[0], thsad=threshold, thscd1=16320)
            selected = 0 if threshold <= expected else 4
            assert constant(comp.get_frame(1), selected), (metric, threshold)
            # Flow uses scene eligibility, whose equality rule is the opposite.
            flow = mv.Flow(c, s, group[0], thscd1=threshold, thscd2=0)
            selected = 0 if threshold < expected else 4
            assert constant(flow.get_frame(1), selected), (metric, threshold)
        # Reliability is zero at equality; a high threshold admits both references.
        assert pixels(mv.Degrain(c, s, group, thsad=expected, thscd1=16320).get_frame(1)) == pixels(c.get_frame(1))
        high = mv.Degrain(c, s, group, thsad=expected * 100, thscd1=16320).get_frame(1)
        assert constant(high, 3), metric
    # Old error is deliberately wrong: Recalculate must measure its own metric.
    forged = core.std.SetFrameProps(v[0], **{P + 'AnalysisSAD': [0] * 16})
    errors(mv.Recalculate(s, forged, thsad=16320, **kw), recalc_error)

print('metric contract passed: raw errors, depth/chroma/area, mixed modes and threshold boundaries')
c.set_output()
