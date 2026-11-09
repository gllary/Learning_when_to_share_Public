"""Run the literal author training body, stopping before its test evaluator."""
import ast
import hashlib
import sys
from common import ROOT

def training_function():
    sys.path.insert(0,str(ROOT/'vendor_ext/gsmnet'))
    import train_prop as author
    source=ROOT/'vendor_ext/gsmnet/train_prop.py'
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_pyg')
    cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and ast.unparse(n)=="print('Testing!')")
    prefix=fn.body[:cut]
    assert any(isinstance(n,ast.If) and ast.unparse(n.test)=='not testing' and 'trainer.run' in ast.unparse(n) for n in prefix)
    scientific_body_sha256=hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    fn.body=prefix+[ast.Return(ast.Tuple(elts=[ast.Name('net',ast.Load()),ast.Name('history',ast.Load())],ctx=ast.Load()))]
    namespace=dict(vars(author))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),namespace)
    return namespace['train_pyg'],namespace,scientific_body_sha256
