import pickle
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import numpy as np

train_path = "./train.txt"
trnMat_save_path = "./trnMat.pkl"

test_path = "./test.txt"
tstMat_save_path = "./tstMat.pkl"

f = open(test_path, "r")
row_list = []
col_list = []
data_list = []

for line in f:
    line = line.strip().split()
    src = line[0]
    dsts = line[1:]
    for dst in dsts:
        row_list.append(int(src))
        col_list.append(int(dst))
        data_list.append(1)
f.close()

row = np.array(row_list)
col = np.array(col_list)
data = np.array(data_list)
# print(row)
coo_m = coo_matrix((data, (row, col)))

save_f = open(tstMat_save_path, 'wb')
pickle.dump(coo_m, save_f)
save_f.close()


