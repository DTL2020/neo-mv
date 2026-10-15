"""Exercise the C++ AviSynth plugin using DS2's pinned host runner."""
import argparse
import ctypes
import os
import pathlib
import subprocess

parser = argparse.ArgumentParser()
for option in ("runner", "plugin", "runtime", "backend", "work"):
    parser.add_argument("--" + option, required=True)
parser.add_argument("--host-api", choices=("c", "cpp"), default="c")
a = parser.parse_args()
work = pathlib.Path(a.work).resolve()
work.mkdir(parents=True, exist_ok=True)
plugin = pathlib.Path(a.plugin).resolve().as_posix()
if os.name == "nt":
    # LoadPlugin must exercise the C++ entry, never silently fall back to C.
    module = ctypes.WinDLL(plugin)
    assert hasattr(module, "AvisynthPluginInit3") or hasattr(module, "_AvisynthPluginInit3@8")
    assert not hasattr(module, "avisynth_c_plugin_init2")
header = f'LoadPlugin("{plugin}")\n'
fixture = """
c = BlankClip(width=64,height=48,length=12,fps=24,pixel_type="YV12",color_yuv=$204080)
s = neo_mv_Super(c,blksize=8,overlap=4,pad=32,pel=2)
v = neo_mv_AnalyseMany(s,radius=2,badrange=0)
b = v[0]
f = v[1]
d = neo_mv_DepanEstimate(c,winx=32,winy=32)
"""
count = 0

def run(name, body, frame=4, error=None, prefix=fixture, mode="--video", extra=()):
    global count
    path = work / (name + ".avs")
    path.write_text(header + prefix + body, encoding="utf-8")
    result = subprocess.run([a.runner, mode, str(path), "--backend", a.host_api,
                             "--runtime", a.runtime, "--frame", str(frame), *extra],
                            capture_output=True, text=True, timeout=40)
    text = result.stdout + result.stderr
    if error:
        assert result.returncode != 0 and error in text, (name, text)
    else:
        assert result.returncode == 0, (name, text)
    count += 1
    return result.stdout

names = ["Super", "Analyse", "AnalyseMany", "Recalculate", "SCDetection", "Compensate",
         "Degrain", "VectorLengthMask", "SADMask", "OcclusionMask", "Flow", "FlowInter",
         "FlowFPS", "FlowBlur", "DepanAnalyse", "DepanCompensate", "DepanEstimate",
         "DepanStabilise", "KernelInfo"] + [f"Degrain{i}" for i in range(1, 26)]
run("registration", "\n".join(f'Assert(FunctionExists("neo_mv_{n}"),"missing {n}")' for n in names)
    + f'\nk=neo_mv_KernelInfo()\nAssert(k[0]=="{a.backend}")\nAssert(k[3]>=1)\nreturn c')

paths = {
    "super": "s", "analyse": "neo_mv_Analyse(s,delta=-1,badrange=0)",
    "many": "v[3]", "recalculate": "neo_mv_Recalculate(s,v)[2]",
    "recalculate-single": "neo_mv_Recalculate(s,b)",
    "scene": "neo_mv_SCDetection(c,b)", "compensate": "neo_mv_Compensate(c,s,b)",
    "degrain": "neo_mv_Degrain(c,s,v,thsad=[400,400],limit=255.0,planes=[0,1,2])",
    "vector-mask": "neo_mv_VectorLengthMask(b)", "sad-mask": "neo_mv_SADMask(b)",
    "occlusion-mask": "neo_mv_OcclusionMask(b)", "flow": "neo_mv_Flow(c,s,b)",
    "inter": "neo_mv_FlowInter(c,s,[b,f])", "fps": "neo_mv_FlowFPS(c,s,[b,f],num=48,den=1)",
    "blur": "neo_mv_FlowBlur(c,s,[b,f])", "depan-analyse": "neo_mv_DepanAnalyse(c,b)",
    "depan-estimate": "d", "depan-compensate": "neo_mv_DepanCompensate(c,d,offset=1.5)",
    "depan-stabilise": "neo_mv_DepanStabilise(c,d)",
    "depan-window": "neo_mv_DepanStabilise(c,d,method=1)",
}
for name, expression in paths.items():
    run(name, "return " + expression)
    run(name + "-prefetch", "return (" + expression + ").Prefetch(4)")

# A later-frame metadata failure must cross the bridge and Prefetch.
for prefetch in (False, True):
    run("runtime-error-" + str(prefetch),
        'bad=b.ScriptClip("""current_frame == 4 ? propSet(last, "MVUtensilsAnalysisPel", 1) : last""")\n'
        'return neo_mv_Compensate(c,s,bad)' + (".Prefetch(4)" if prefetch else ""),
        error="render analysis metadata changed")
for radius in range(1, 26):
    run(f"degrain{radius}", f"vv=neo_mv_AnalyseMany(s,radius={radius},badrange=0)\n"
        f"return neo_mv_Degrain{radius}(c,s,vv)")
for name in ("compensate", "degrain", "inter", "blur", "depan-stabilise"):
    for frame in (0, 11):
        run(f"{name}-edge{frame}", "return " + paths[name], frame=frame)

# Runtime property assertions check bundle order and custom-prefix propagation.
run("properties", '''
s2=neo_mv_Super(c,blksize=[8,8],overlap=[],pad=[32,32],prefix="Test")
v2=neo_mv_AnalyseMany(s2,radius=2,delta=2,prefix="Test",badrange=0)
return v2[2].ScriptClip("""Assert(propGetInt(last, "TestAnalysisDelta") == 4)
last""")
''')
run("recalculate-order", '''
r=neo_mv_Recalculate(s,v)
Assert(IsArray(r) && ArraySize(r)==4)
return r[3].ScriptClip("""Assert(propGetInt(last, "MVUtensilsAnalysisDelta") == -2)
last""")
''')
# Preserve the caller's container, including explicit singleton arrays.
for name, vectors, check, selected in (
    ("clip", "b", "IsClip(r)", "r"),
    ("singleton", "[b]", "IsArray(r) && ArraySize(r)==1", "r[0]"),
    ("array", "[b,f]", "IsArray(r) && ArraySize(r)==2", "r[0]"),
):
    for named in (False, True):
        arguments = f"s,vectors={vectors}" if named else f"s,{vectors}"
        run(f"recalculate-shape-{name}-{named}",
            f"r=neo_mv_Recalculate({arguments})\nAssert({check})\n"
            f"r=neo_mv_Recalculate(s,r)\nAssert({check})\n"
            f'return {selected}.ScriptClip("""Assert(propGetInt(last, "MVUtensilsAnalysisDelta") == 1)\n'
            'last""").Prefetch(4)')
for name, expression in {
    "analyse": "neo_mv_DepanAnalyse(c,b,info=true)",
    "compensate": "neo_mv_DepanCompensate(c,d,info=true)",
    "estimate": "neo_mv_DepanEstimate(c,winx=32,winy=32,info=true)",
    "stabilise": "neo_mv_DepanStabilise(c,d,info=true)",
}.items():
    run("info-" + name, "return " + expression)

# Audio and parity come from the first clip, independent of vector inputs.
audio = """
c=AudioDub(c,Tone(length=0.5,samplerate=48000,channels=2)).AssumeTFF()
o=neo_mv_Compensate(c,s,b)
Assert(AudioRate(o)==AudioRate(c))
Assert(AudioChannels(o)==AudioChannels(c))
Assert(GetParity(o,4)==GetParity(c,4))
"""
original = run("audio-source", audio + "return c", mode="--audio")
forwarded = run("audio-output", audio + "return o.Prefetch(4)", mode="--audio")
assert original == forwarded

# Distinct temporal samples catch an adapter that returns the current frame or
# requests the wrong reference. Check pixels, including end-of-clip fallback.
temporal = """
c=BlankClip(width=32,height=32,length=1,pixel_type="Y8",color_yuv=$0A8080) \\
 + BlankClip(width=32,height=32,length=1,pixel_type="Y8",color_yuv=$1E8080) \\
 + BlankClip(width=32,height=32,length=1,pixel_type="Y8",color_yuv=$5A8080)
s=neo_mv_Super(c,blksize=8,overlap=4,pad=32,pel=1)
b=neo_mv_Analyse(s,delta=1,badrange=0)
"""
for frame, value in enumerate((30, 90, 90)):
    run(f"reference-pixels-{frame}",
        "return neo_mv_Compensate(c,s,b,thsad=16320,thscd1=16320).Prefetch(4)",
        frame=frame, prefix=temporal, extra=("--expect-y8-sum", str(32 * 32 * value)))
    for refine in (False, True):
        run(f"recalculate-consumer-{frame}-{refine}",
            f"v={'neo_mv_Recalculate(s,b,thsad=0)' if refine else 'b'}\n"
            "Assert(IsClip(v))\n"
            "return neo_mv_Compensate(c,s,v,thsad=16320,thscd1=16320).Prefetch(4)",
            frame=frame, prefix=temporal, extra=("--expect-y8-sum", str(32 * 32 * value)))

for name, body, error in [
    ("missing-block", "return neo_mv_Super(c,overlap=4)", "blksize"),
    ("bad-array", 'return neo_mv_Super(c,blksize=[8,"bad"],overlap=4)', "expected integer"),
    ("bad-radius", "return neo_mv_AnalyseMany(s,radius=0)", "radius"),
    ("large-bundle", "return neo_mv_AnalyseMany(s,radius=16384)", "radius"),
    ("empty-recalculate", "return neo_mv_Recalculate(s,[])", "nonempty"),
    ("single-recalculate-invalid", "return neo_mv_Recalculate(s,c)", "member[0]"),
    ("singleton-recalculate-invalid", "return neo_mv_Recalculate(s,[c])", "member[0]"),
    ("bundle-rollback", "return neo_mv_Recalculate(s,[b,c])", "member[1]"),
    ("bad-degrain", "return neo_mv_Degrain1(c,s,v)", "exactly 2R"),
    ("bad-format", 'return neo_mv_Super(BlankClip(c,pixel_type="RGB32"),8,4)', "format"),
]:
    run(name, body, error=error)

# Exercise integer high depth and floating-point planes through the same bridge.
for pixel in ("Y8", "YUV420P10", "YUV420P16", "Y32", "YUV444PS"):
    setup = fixture.replace('pixel_type="YV12"', f'pixel_type="{pixel}"')
    # Global compensation is integer-only; it is not used in these graphs.
    setup = setup[:setup.index("d = ")]
    run("format-" + pixel, "return neo_mv_Compensate(c,s,b)", prefix=setup)
# Non-power-of-two square blocks through the public analysis/recalculation chain.
for block in (12, 24, 48):
    for overlap in (0, block // 2):
        for metric in ("sad", "satd"):
            setup = temporal[:temporal.index("s=neo_mv_Super")].replace("width=32,height=32", "width=192,height=192")
            setup += (f"s=neo_mv_Super(c,blksize={block},overlap={overlap},pad=32,pel=1)\n"
                      f"v=neo_mv_AnalyseMany(s,radius=1,badrange=0,metric=\"{metric}\")\n"
                      f"r=neo_mv_Recalculate(s,v,thsad=0,metric=\"{metric}\")\n")
            run(f"square-{block}-{overlap}-{metric}",
                "return neo_mv_Compensate(c,s,r[0],thsad=16320,thscd1=16320)",
                frame=0, prefix=setup, extra=("--expect-y8-sum", str(192 * 192 * 30)))
# 6x6 SAD: exercise 3x3 chroma and overlap normalization through the AVS bridge.
for pixel in ("Y8", "Y16", "Y32", "YV12", "YUV420P16", "YUV420PS", "YV16", "YV24"):
    for overlap in ((0, 1, 2, 3) if pixel in ("Y8", "Y16", "Y32", "YV24") else (0, 2)):
        setup = (f'c=BlankClip(width=32,height=26,length=5,pixel_type="{pixel}")\n'
                 f's=neo_mv_Super(c,blksize=6,overlap={overlap},pad=16,pel=2)\n'
                 'v=neo_mv_AnalyseMany(s,radius=1,badrange=0)\n'
                 'r=neo_mv_Recalculate(s,v,thsad=0)\n'
                 'b=r[0]\nf=r[1]\n')
        for name in ("compensate", "degrain", "flow", "inter", "fps", "blur",
                     "vector-mask", "sad-mask", "occlusion-mask"):
            run(f"six-{pixel}-{overlap}-{name}", "return " + paths[name], frame=2, prefix=setup)
for expression in ('neo_mv_Analyse(s,metric="satd")', 'neo_mv_AnalyseMany(s,metric="satd")[0]',
                   'neo_mv_Recalculate(s,v,metric="satd")[0]'):
    run("six-satd-" + str(count), "return " + expression, prefix=setup, error="divisible by 4")
for overlap in (1, 3):
    run(f"six-420-overlap-{overlap}",
        f"return neo_mv_Super(c,blksize=6,overlap={overlap})", error="chroma aligned")
# A distinct reference must actually be read, rather than simply copying the input.
for overlap in range(4):
    setup = temporal.replace('blksize=8,overlap=4', f'blksize=6,overlap={overlap}')
    run(f"six-reference-{overlap}",
        "return neo_mv_Compensate(c,s,b,thsad=16320,thscd1=16320).Prefetch(4)",
        frame=0, prefix=setup, extra=("--expect-y8-sum", str(32 * 32 * 30)))
# Exercise DCT through the C++ entry, including Prefetch and non-power-of-two blocks.
for pixel in ("Y8", "Y16", "YUV420P10", "YUV420P16"):
    for block in (6, 8, 12, 24, 48):
        setup = (f'c=BlankClip(width=96,height=96,length=6,pixel_type="{pixel}")\n'
                 f's=neo_mv_Super(c,blksize={block},overlap=0,pad=16,pel=2)\n'
                 'v=neo_mv_AnalyseMany(s,radius=1,metric="dct",badrange=0)\n'
                 'r=neo_mv_Recalculate(s,v,metric="dct",thsad=0)\n')
        run(f"dct-{pixel}-{block}", "return neo_mv_Compensate(c,s,r[0]).Prefetch(4)", frame=2, prefix=setup)
setup = ('c=BlankClip(width=32,height=32,length=6,pixel_type="Y32")\n'
         's=neo_mv_Super(c,blksize=8,overlap=0)\n'
         'v=neo_mv_Analyse(s)\n')
for expression in ('neo_mv_Analyse(s,metric="dct")', 'neo_mv_AnalyseMany(s,metric="dct")[0]',
                   'neo_mv_Recalculate(s,v,metric="dct")'):
    run("float-dct-"+str(count), "return "+expression, prefix=setup, error="integer samples")
print(f"AviSynth {a.backend}: {count} cases passed; all 44 functions registered.")
# Mixed policy registration and configuration through the native C++ AVS adapter.
for name in ("sad_dct_global", "sad_dct_local", "sad_satd_global", "sad_satd_local", "sad_satd_global_half"):
    setup = ('a=BlankClip(width=32,height=32,length=3,pixel_type="Y8",color_yuv=$280000)\n'
             'b=BlankClip(a,color_yuv=$320000)\n'
             'c=Interleave(a,b)\n'
             's=neo_mv_Super(c,blksize=8,overlap=0,pad=16,pel=2)\n'
             f'v=neo_mv_AnalyseMany(s,radius=1,metric="{name}",badrange=0)\n'
             f'r=neo_mv_Recalculate(s,v,metric="{name}",thsad=0)\n')
    run("mixed-"+name,"return neo_mv_Compensate(c,s,r[0]).Prefetch(4)",frame=2,prefix=setup)
    extra=',metric_weight=0.6,metric_threshold=0.04' if 'local' in name else ',metric_weight=0.5'
    for expression in (f'neo_mv_Analyse(s,metric="{name}"{extra})',
                       f'neo_mv_AnalyseMany(s,metric="{name}"{extra})[0]',
                       f'neo_mv_Recalculate(s,v,metric="{name}"{extra})[0]'):
        run("mixed-params-"+str(count),"return "+expression,prefix=setup,
            error=None if 'local' in name else "local metric")
for key in ("metric_weight", "metric_threshold"):
    for value in (-0.1, 1.1):
        run("mixed-invalid-"+str(count),
            f'return neo_mv_Analyse(s,metric="sad_satd_local",{key}={value})',prefix=setup,error="[0,1]")
print(f"AviSynth {a.backend}: {count} cases including mixed metrics passed.")

# Constant temporal blocks provide an independent error and consumer oracle.
setup = temporal[:temporal.index('b=neo_mv_Analyse')]
setup += 'b=neo_mv_Analyse(s,badrange=0)\n'
for fn, args, select in [('Analyse','s','r'), ('AnalyseMany','s','r[0]'),
                         ('Recalculate','s,b','r'), ('Recalculate','s,[b]','r[0]')]:
    for value, metric, error_value in [('false','sad',1280), ('true','satd',640)]:
        for suffix in [f',satd={value}', f',satd={value},metric="{metric}"'] + ([',metric="sad"',''] if value=='false' else []):
            kind='clip' if select=='r' else 'array'
            run('satd-alias-'+str(count),
                f'r=neo_mv_{fn}({args}{suffix})\nAssert(Is{kind.title()}(r))\nx={select}\n'
                f'x=x.ScriptClip("""Assert(propGetInt(last, "MVUtensilsAnalysisSAD", 0)=={error_value})\nlast""")\n'
                'return neo_mv_Compensate(c,s,x,thsad=16320,thscd1=16320)',
                frame=0, prefix=setup, extra=('--expect-y8-sum',str(32*32*30)))
        for conflict in (('satd','dct','sad_satd_local') if value=='false' else ('sad','dct','sad_dct_local')):
            run('satd-conflict-'+str(count),f'return neo_mv_{fn}({args},satd={value},metric="{conflict}")',
                prefix=setup,error='satd conflicts with metric')
    run('satd-type-'+str(count),f'return neo_mv_{fn}({args},satd="true")',prefix=setup,error='wrong type')
for fn, args, skipped, tail, select in [('Analyse','s',21,3,'r'), ('AnalyseMany','s',21,4,'r[0]'),
                                       ('Recalculate','s,b',12,3,'r')]:
    for params in ([*(['Undefined()']*skipped),'"satd"'], [*(['Undefined()']*(skipped+1+tail)),'true']):
        run('satd-positional-'+str(count), f'r=neo_mv_{fn}({args},'+','.join(params)+f')\nx={select}\n'
            'return x.ScriptClip("""Assert(propGetInt(last, "MVUtensilsAnalysisSAD", 0)==640)\nlast""")',
            prefix=setup,frame=0)
print(f'AviSynth {a.backend}: {count} cases including satd aliases passed.')

# Analyse -> Recalculate -> consumers retain the chosen error units.
contract_setup = '''
c=BlankClip(width=32,height=32,length=1,pixel_type="Y8",color_yuv=$008080) \\
 + BlankClip(width=32,height=32,length=1,pixel_type="Y8",color_yuv=$048080) \\
 + BlankClip(width=32,height=32,length=1,pixel_type="Y8",color_yuv=$008080)
s=neo_mv_Super(c,blksize=8,overlap=0,pad=16,pel=1,onelevel=true)
'''
for metric, ae, re in [('sad',256,256), ('satd',128,128), ('dct',32,32),
                       ('sad_dct_local',132,132), ('sad_satd_local',192,192),
                       ('sad_dct_global',256,144), ('sad_satd_global',256,192),
                       ('sad_satd_global_half',256,224)]:
    options=f'metric="{metric}"' + (',metric_weight=0.5,metric_threshold=0' if 'local' in metric else '')
    setup=contract_setup+f'v=neo_mv_AnalyseMany(s,radius=1,badrange=0,{options})\nr=neo_mv_Recalculate(s,v,thsad=0,{options})\n'
    for group, error_value in [('v',ae),('r',re)]:
        checked=setup+f'x={group}[0].ScriptClip("""Assert(propGetInt(last, "MVUtensilsAnalysisSAD", 0)=={error_value})\nlast""")\n'
        for threshold in (error_value-1,error_value,error_value+1):
            run('metric-comp-'+str(count),f'return neo_mv_Compensate(c,s,x,thsad={threshold},thscd1=16320)',
                prefix=checked,frame=1,extra=('--expect-y8-sum',str(32*32*(4 if threshold<=error_value else 0))))
            run('metric-flow-'+str(count),f'return neo_mv_Flow(c,s,x,thscd1={threshold},thscd2=0)',
                prefix=checked,frame=1,extra=('--expect-y8-sum',str(32*32*(4 if threshold<error_value else 0))))
        for threshold, pixel in [(error_value,4),(error_value*100,1)]:
            run('metric-degrain-'+str(count),f'return neo_mv_Degrain(c,s,{group},thsad={threshold},thscd1=16320)',
                prefix=setup,frame=1,extra=('--expect-y8-sum',str(32*32*pixel)))
print(f'AviSynth {a.backend}: {count} cases including metric contracts passed.')

# Rectangular blocks: non-square metadata, partial edges and distinct references.
for bw, bh in [(4,8), (8,16)]:
    w,h=4*bw+2,4*bh+2
    for pel in (1,2,4):
        for metric in ('sad','satd','dct'):
            if metric=='satd' and (bw%4 or bh%4): continue
            setup=(f'c=BlankClip(width={w},height={h},length=1,pixel_type="Y8",color_yuv=$008080) + '
                   f'BlankClip(width={w},height={h},length=2,pixel_type="Y8",color_yuv=$048080)\n'
                   f's=neo_mv_Super(c,blksize=[{bw},{bh}],overlap=0,pel={pel})\n'
                   f'v=neo_mv_AnalyseMany(s,radius=1,badrange=0,metric="{metric}")\n'
                   f'r=neo_mv_Recalculate(s,v,thsad=0,metric="{metric}")\n')
            run('rectangle-'+str(count),'return neo_mv_Compensate(c,s,r[0],thsad=16320,thscd1=16320)',
                frame=0,prefix=setup,extra=('--expect-y8-sum',str(w*h*4)))
print(f'AviSynth {a.backend}: {count} cases including rectangles passed.')
