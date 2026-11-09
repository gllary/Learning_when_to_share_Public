import runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.argv=[str(root/"code/scripts/predict_external.py"),"core23"]+sys.argv[1:]
runpy.run_path(sys.argv[0],run_name="__main__")
