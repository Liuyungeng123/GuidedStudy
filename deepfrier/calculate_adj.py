from Bio.PDB import PDBParser, DSSP
import numpy as np

def calculate_min_ca_distance(residue1, residue2):
    # 获取 Cα 原子坐标
    ca1 = residue1['CA'].get_coord()
    ca2 = residue2['CA'].get_coord()

    # 计算 Cα–Cα 的距离
    distance = np.linalg.norm(ca1 - ca2)
    return distance

def calculate_block_adjacency_matrix(structure, filename,chain_id = 'A',chain_length=107):
    model = structure[0]
    chain = model[chain_id]
    residues = list(chain.get_residues())
    dssp = DSSP(model, filename)

    # 初始化 block-adjacency matrix
    num_residues = chain_length
    adjacency_matrix = np.zeros((num_residues, num_residues))
    non_adjacency_matrix = np.zeros((num_residues, num_residues))
    helix_matrix = {}
    # print(dssp.property_dict.values())
    # print(dssp.keys())
    list1 = list(dssp.property_dict.values())
    # print(list1[0],list1[1],list1[2],list1[3],list1[4],list1[5],list1[6])
    for i in range(num_residues):
        for j in range(i+1, num_residues):
            # print(dssp[i])
            # print(list1[i],list1[j])
            _, _, ss1,_, _, _, _, _ , _, _, _, _, _, _= list1[i]
            _, _, ss2,_, _, _, _, _, _, _, _, _, _, _= list1[j]    
            # print(ss1,ss2)
            # helix_matrix[i] = str(ss1)

            # 判断是否满足条件
            # 使用DSSP [75]计算了训练集中每个残基的二级结构注释。DSSP是一种基于结构的算法，为每个残基分配二级结构类型的分类。
            # 从DSSP返回的二级结构字符串计算了训练集中每个结构的块邻接矩阵。
            # 如果（1）两个块都不是环类型，并且
            # （2）任意一对块之间的Cα-Cα最小距离在8˚A内，
            # 则在块邻接矩阵中标记块为“邻接”。
            if (ss1 != ' ' and ss2 != ' ') and (ss1 != 'C' and ss2 != 'C'):
                min_ca_distance = calculate_min_ca_distance(residues[i], residues[j])
                if min_ca_distance <= 8.0:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                # else:
                #     non_adjacency_matrix[i, j] = 1
                #     non_adjacency_matrix[j, i] = 1
        helix_matrix[i] = str(ss1)
    non_adjacency_matrix = np.logical_not(adjacency_matrix).astype(int)
    np.fill_diagonal(non_adjacency_matrix, 0)
    return adjacency_matrix, non_adjacency_matrix, helix_matrix

# 读取 PDB 文件
parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', '/data/1a2y.pdb')

# 计算 block-adjacency matrix
adjacency_matrix, non_adjacency_matrix, helix= calculate_block_adjacency_matrix(structure,'/data/1a2y.pdb')

# 打印结果
print("Block-Adjacency Matrix:")
print(adjacency_matrix)
# print("Non-Adjacency Matrix:")
# print(non_adjacency_matrix)
print("Helix Matrix:")
print(helix)
