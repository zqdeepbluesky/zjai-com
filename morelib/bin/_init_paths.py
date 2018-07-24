import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
morelib_path = osp.join(this_dir, '..',"..", 'morelib')
add_path(morelib_path)

lib_path = osp.join(this_dir, '..',"..", 'lib')
add_path(lib_path)
