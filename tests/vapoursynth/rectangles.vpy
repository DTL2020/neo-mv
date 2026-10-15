"""Rectangular geometry, partial edges, metric restrictions and all consumers."""
import vapoursynth as vs

core = vs.core
core.num_threads = 4
core.std.LoadPlugin(path=plugin)
mv = core.neo_mv
SHAPES = [(4, 8), (8, 16)]


def nodes(value):
    return [value] if isinstance(value, vs.VideoNode) else list(value)


def check_vectors(node, shape):
    f = node.get_frame(1)
    assert f.props['MVUtensilsAnalysisBlkSizeX'] == shape[0]
    assert f.props['MVUtensilsAnalysisBlkSizeY'] == shape[1]
    for name in ('Vectors', 'SAD'):
        values = f.props['MVUtensilsAnalysis' + name]
        values = values if isinstance(values, (list, tuple)) else [values]
        assert all(x == 0 for x in values), (shape, name, values)


def same_pixels(node, source):
    with node.get_frame(1) as actual, source.get_frame(1) as expected:
        for p in range(len(expected)):
            a, b = actual[p], expected[p]
            assert a.shape == b.shape
            # At most four overlap weights, each rounded to 1/2048.
            # Their combined error is at most 2/2048 for samples in [0, 1].
            tolerance = 2/2048 + 1e-6 if source.format.sample_type == vs.FLOAT else 0
            assert all(abs(a[y, x] - b[y, x]) <= tolerance
                       for y in range(b.shape[0]) for x in range(b.shape[1]))


for bw, bh in SHAPES:
    for fmt in (vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV422P16, vs.YUV444PS):
        info = core.get_video_format(fmt)
        rx, ry = 1 << info.subsampling_w, 1 << info.subsampling_h
        c = core.std.BlankClip(width=4*bw+2, height=4*bh+2, format=fmt, length=3)
        def fill(n, f):
            out = f.copy()
            for p in range(len(out)):
                a = out[p]
                for y in range(a.shape[0]):
                    for x in range(a.shape[1]):
                        value = (17*x + 29*y + x*y + 13*p) % 251
                        a[y, x] = value / 256 if info.sample_type == vs.FLOAT else value
            return out
        c = core.std.ModifyFrame(c, c, fill)
        for pel in (1, 2, 4):
            for overlap in ([0, 0], [bw//(2*rx)*rx, bh//(2*ry)*ry]):
                s = mv.Super(c, blksize=[bw, bh], overlap=overlap, pel=pel)
                for metric in ('sad', 'satd', 'dct'):
                    if metric == 'dct' and info.sample_type == vs.FLOAT:
                        continue
                    if metric == 'satd' and (bw % 4 or bh % 4):
                        try:
                            mv.Analyse(s, metric=metric)
                        except vs.Error as error:
                            assert 'divisible by 4' in str(error)
                        else:
                            raise AssertionError('invalid SATD geometry accepted')
                        continue
                    v = nodes(mv.AnalyseMany(s, radius=1, metric=metric, badrange=0))
                    r = nodes(mv.Recalculate(s, v, metric=metric, thsad=0))
                    for node in v+r:
                        check_vectors(node, (bw, bh))
                    for index, node in enumerate((mv.Compensate(c, s, r[0]), mv.Degrain(c, s, r),
                                 mv.Flow(c, s, r[0]), mv.FlowInter(c, s, r),
                                 mv.FlowFPS(c, s, r, num=48, den=1), mv.FlowBlur(c, s, r))):
                        try:
                            same_pixels(node, c)
                        except AssertionError as error:
                            raise AssertionError((bw, bh, fmt, pel, overlap, metric, index)) from error
                    for node in (mv.VectorLengthMask(r[0]), mv.SADMask(r[0]), mv.OcclusionMask(r[0])):
                        node.get_frame(1)
                if (bw//2, bh//2) in SHAPES:
                    refined = nodes(mv.Recalculate(s, v, blksize=[bw//2, bh//2], overlap=0, thsad=0))
                    for node in refined:
                        check_vectors(node, (bw//2, bh//2))
                        same_pixels(mv.Compensate(c, s, node), c)

# An asymmetric one-pixel translation detects swapped axes and inert analysis.
for bw, bh in SHAPES:
    c = core.std.BlankClip(width=4*bw+2, height=4*bh+2, format=vs.GRAY8, length=3)
    def shift(n, f):
        out = f.copy()
        a = out[0]
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                xx = max(0, x-n)
                a[y, x] = (17*xx + 29*y + xx*y) % 251
        return out
    c = core.std.ModifyFrame(c, c, shift)
    s = mv.Super(c, blksize=[bw, bh], overlap=0, pel=1, onelevel=True)
    v = mv.Analyse(s, search=4, searchparam=2, mvlambda=0, pnew=0, pzero=0, badrange=0)
    f = v.get_frame(0)
    assert f.props['MVUtensilsAnalysisVectors'][0] == 1
    assert f.props['MVUtensilsAnalysisSAD'][0] == 0

print('Rectangular block checks passed')
c.set_output()
