"""The single source of truth for 34 full + 170 OOF jobs."""
FOLDS=(-1,0,1,2,3,4)
def fit_name(fold):
    assert fold in FOLDS
    return 'full' if fold<0 else f'fold{fold}'
def jobs(rows):
    return [(row,fold) for fold in FOLDS for row in rows]
def key(row,fold):return row['id']+'/'+fit_name(fold)
