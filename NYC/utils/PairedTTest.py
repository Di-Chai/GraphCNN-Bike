import numpy as np
from scipy import stats

def pairedTTest(valueList1, valueList2):
    if len(valueList1) != len(valueList2):
        print('Length error in paired t test')
        return None
    d = []
    for i in range(len(valueList1)):
        d.append(valueList1[i] - valueList2[i])

    d_bar = np.mean(d)
    sd = np.std(d)

    SE = sd / np.sqrt(len(valueList1))
    degreeFreedom = len(valueList1) - 1

    # t statics
    T = (d_bar + 0.00001) / (SE + 0.00001)
    # p-value
    p_value = stats.t.sf(abs(T), degreeFreedom)

    return T, p_value