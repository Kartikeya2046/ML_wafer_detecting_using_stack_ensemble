import sys
import numpy as np
from scipy import stats
sys.path.append(r"d:\assignment college\VLSI_PROJECT\WMPC_Stacking_TF2\run_code")
from extract_manual_features import fea_geom

def test():
    img = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    try:
        res = fea_geom(img)
        print("Success!", res)
    except Exception as e:
        print("Failed!", e)
test()
