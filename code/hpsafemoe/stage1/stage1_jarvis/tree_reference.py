"""Execute only the pinned official estimator factory, not its top-level jobs."""
import ast
from common import ROOT

def official_pipeline(model_id,classification):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
    import xgboost as xgb
    assert model_id in ['matminer_rf','matminer_xgboost']
    source=ROOT/'official'/(model_id+'.py')
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_model')
    namespace=dict(StandardScaler=StandardScaler,SimpleImputer=SimpleImputer,Pipeline=Pipeline,
                   RandomForestRegressor=RandomForestRegressor,RandomForestClassifier=RandomForestClassifier,xgb=xgb)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['get_model']('SinglePropertyClass' if classification else 'SinglePropertyPrediction')
