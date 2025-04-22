# import pandas as pd
# import numpy as np
# from utils import percent_pointing_inside_com

# from recbole.quick_start import run_recbole
#
# run_recbole(model= 'GRU4Rec', dataset='ml-100k')

import numpy as np
from utils_functions import get_community_labels
import scipy.sparse as sp
from scipy.sparse import csr_matrix, dok_matrix

# Create a dok_matrix
matrix = dok_matrix((5, 5), dtype=np.float32)
# get community labels
# Correctly use the update method
matrix._update({(0, 0): 1, (1, 2): 2})
#
print(matrix)
