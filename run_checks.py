"""Run actual unit tests and functional negative variants; save concise evidence."""
import io,json,unittest
from pathlib import Path
import experiment

suite=unittest.defaultTestLoader.discover('.',pattern='test_experiment.py')
stream=io.StringIO()
r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
print(stream.getvalue())
Path('artifacts').mkdir(exist_ok=True)
result={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'passed':r.wasSuccessful(),
        'composition_checks':8*8*4,
        'mutations':{m:experiment.invariant_failures(experiment.run(m))
                     for m in ('special_only','drop_preprocessing')}}
Path('artifacts/checks.json').write_text(json.dumps(result,indent=2)+'\n')
if not r.wasSuccessful() or not all(result['mutations'].values()): raise SystemExit(1)
