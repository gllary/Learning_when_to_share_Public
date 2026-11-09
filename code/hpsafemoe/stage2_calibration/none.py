"""No external safety control. Reuse the frozen raw proposal without fitting."""
import numpy as np

def apply(anchor,proposal):
    anchor=np.asarray(anchor);proposal=np.asarray(proposal)
    if anchor.shape!=proposal.shape or not np.isfinite(proposal).all():
        raise ValueError('Invalid frozen proposal')
    return proposal.copy()
