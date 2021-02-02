import numpy as np
import grasp_data_toolkit as gdt

# test if setting properties of GraspSet performs deep copies or not
# to check this, i will initialise some grasp set
# i then need to set the translations with some other np array
# then change the other np array
# then check if the grasp set is affected


initial_grasp_set = np.zeros((3, gdt.core_types.Grasp.ARRAY_LEN), dtype=np.float32)
gs = gdt.core_types.GraspSet(initial_grasp_set)
print(gs.translations)  # should be zeros

translations = np.ones((3, 3), dtype=np.float32)
gs.translations = translations
print(gs.translations)  # should be ones

translations[0] = np.asarray([3, 3, 3])
translations[1] = 3
print(gs.translations)  # should be ones if deep copied, or contains threes if shallow copied
# result: does contain only ones, so there are no unwanted side effects

print(translations)

