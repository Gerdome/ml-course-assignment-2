import os
import nbformat
from nbconvert import PythonExporter, HTMLExporter
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from traitlets.config import Config
import warnings
warnings.filterwarnings('ignore') 

# Convert notebook to python
assignment_path = '../Assignment 2.ipynb'
export_path = 'solution.py'
template_path = 'Template.ipynb'
submission_path = '../Submission.html'

print("Converting: {0}".format(assignment_path))
exporter = PythonExporter()
# source is a tuple of python source code, meta contains metadata
(source, meta) = exporter.from_filename(assignment_path)
with open(export_path, 'w+') as fh:
    fh.writelines(source)
    fh.writelines("last_edit = '{}'".format(meta['metadata']['modified_date']))
    
# Run solution notebook
with open(template_path) as f:
    snb = nbformat.read(f, as_version=4)
ep = ExecutePreprocessor(timeout=2000, kernel_name='python3')

try:
    print("Running notebook... (may take a while)")
    out = ep.preprocess(snb, {'metadata': {'path': './'}})
except CellExecutionError:
    out = None
    msg = 'Error executing the notebook "%s".\n\n' % template_path
    msg += 'See notebook "%s" for the traceback.' % template_path
    print(msg)
    raise
finally:
    # Save notebook
    with open(template_path, mode='w', encoding='utf-8') as f:
        nbformat.write(snb, f)

# Export as HTML (PDF is too much hassle)
print("All good. Building report.")
c = Config()
c.TagRemovePreprocessor.enabled=True
c.TagRemovePreprocessor.remove_input_tags = set(["hide_input"])
c.preprocessors = ["TagRemovePreprocessor"]

html_exporter = HTMLExporter(config=c)
html_data, resources = html_exporter.from_notebook_node(snb)
html_data = html_data.replace('</head>', '<style>pre{font-family: "Times New Roman", Times, serif;}</style></head>')

with open(submission_path, "wb") as f:
    f.write(html_data.encode('utf8'))
    f.close()
#Cleanup
#os.remove("solution.py")
print("Done.")