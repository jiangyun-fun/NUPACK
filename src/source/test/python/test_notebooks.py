
import os, json
from tempfile import TemporaryDirectory
from pathlib import Path

def run_notebook(path):
    try:
        with path.open('r') as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1800, kernel_name='python3')
        ep.preprocess(nb)
    except Exception as e:
        if 'This event loop is already running' not in str(e): # happens in macOS x86 3.11 for some reason
            raise RuntimeError('Notebook %s failed' % path) from e


env = os.environ.get('NUPACK_NOTEBOOK_TEST_DIR')

if env is not None:
    import nbconvert, nbformat, matplotlib, ipykernel # These packages should be installed
    from nbconvert.preprocessors import ExecutePreprocessor

    paths = [Path(p).expanduser() for p in env.split(':')]
    for p in paths:
        assert p.exists(), p
    for p in paths:
        for notebook in p.glob('**/*.ipynb'):
            if '.ipynb_checkpoints' in str(notebook):
                continue
            globals()['test_notebook_' + notebook.stem] = lambda x=notebook: run_notebook(x)



