from scipy.io import loadmat
import sys
import numpy as np

group_1 = [
    'Frontal_Inf_Tri', 
    'Frontal_Inf_Oper', 
    'Frontal_Inf_Orb', 
    'Temporal_Mid', 
    'Temporal_Sup'
]
group_2 = [
    'Frontal_Sup', 
    'Frontal_Mid', 
    'Angular', 
    'Precuneus', 
    'Frontal_Sup_Med', 
    'Frontal_Med_Orb'
]

regs_1 = [i + '_L' for i in group_1] + [i + '_R' for i in group_1]
regs_2 = [i + '_L' for i in group_2] + [i + '_R' for i in group_2]

x = loadmat(sys.argv[1])

ROI_num_to_name = [x['meta']['ROInumToName'][0][0][0][i][0] for i in range(117)]

group_id = list()
for i in x['meta']['colToROInum'][0][0][0]:
    if 'Cerebellum' not in ROI_num_to_name[i - 1] and 'Vermis' not in ROI_num_to_name[i - 1]: 
        if ROI_num_to_name[i - 1] in regs_1:
            group_id.append(1)
        elif ROI_num_to_name[i - 1] in regs_2:
            group_id.append(2)
        else:
            group_id.append(0)

group_array = np.array(group_id)
group_array.dump(sys.argv[2])