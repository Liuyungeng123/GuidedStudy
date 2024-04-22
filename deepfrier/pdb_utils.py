import numpy as np
import torch
# from contigs import ContigMap
from Bio import PDB

# ideal N, CA, C initial coordinates
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.full((27, 3), np.nan)
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0) # (3,3)

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    ]
aa2num= {x:i for i,x in enumerate(num2aa)}

aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",  None,  None,  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",  None,  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",  None,  None,  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
]

def calculate_ca_cb_distances(structure):
    model = structure[0]
    chain_ids = []
    for model in structure:
        for chain in model:
            chain_ids.append(chain.id)
    # print(chain_ids,chain_id_length)
    chain = model[chain_ids[0]]

    # 获取所有残基的信息
    residues = list(chain.get_residues())
    # print(residues)
    for residue in chain.get_list():
        residue_id = residue.get_id()
        hetfield = residue_id[0]
        if hetfield[0]=="H" or hetfield[0]=="W":
            # remove HETATM
            residues.remove(residue)
    # print(residues,len(residues))
    # 初始化邻接矩阵
    num_residues = len(residues)
    # print(num_residues)
    ca_distances = np.zeros((num_residues, num_residues))
    cb_distances = np.zeros((num_residues, num_residues))

    # print(ca_distances.shape,cb_distances.shape)
    # 计算 Cα–Cα 和 Cβ–Cβ 的距离
    for i in range(num_residues):
        for j in range(i+1, num_residues):
            # 检查是否存在Cα原子
            if 'CA' in residues[i]:
                ca1 = residues[i]['CA'].get_coord()
            else:
                ca1 = np.nan
            # print(i,j,ca1)

            if 'CA' in residues[j]:
                ca2 = residues[j]['CA'].get_coord()
            else:
                ca2 = np.nan

            # 检查是否存在Cβ原子
            if 'CB' in residues[i]:
                cb1 = residues[i]['CB'].get_coord()
            else:
                cb1 = np.nan

            if 'CB' in residues[j]:
                cb2 = residues[j]['CB'].get_coord()
            else:
                cb2 = np.nan

            # 检查Cα原子是否存在，如果不存在，将距离置为nan
            if not np.isnan(ca1).any() and not np.isnan(ca2).any():
                ca_distances[i, j] = ca_distances[j, i] = np.linalg.norm(ca1 - ca2)
            else:
                ca_distances[i, j] = ca_distances[j, i] = np.nan

            # 检查Cβ原子是否存在，如果不存在，将距离置为nan
            if not np.isnan(cb1).any() and not np.isnan(cb2).any():
                cb_distances[i, j] = cb_distances[j, i] = np.linalg.norm(cb1 - cb2)
            else:
                cb_distances[i, j] = cb_distances[j, i] = np.nan
    #
    coordinates = find_coordinates_both(cb_distances, ca_distances, 8, 8, 0.1, 'cb')
    # print('co',coordinates)
    return ca_distances, cb_distances, coordinates


def find_coordinates_both(matrix_cb, matrix_ca, threshold_cb, threshold_ca, min_percentage, distance_type='cb'):
    coordinates = []

    # da
    matrix = matrix_cb if distance_type == 'cb' else matrix_ca
    threshold = threshold_cb if distance_type == 'cb' else threshold_ca
    # print('matrix',matrix.shape,distance_type,matrix)
    coordinates = filter_distances(matrix, threshold, min_percentage)

    return coordinates

def filter_distances(adjacency_matrix, threshold, percent):
    num_residues, _ = adjacency_matrix.shape

    selected_residues = []

    for i in range(num_residues):
        # 计算小于阈值的距离数量
        num_below_threshold = np.sum(adjacency_matrix[i, :] < threshold)
        # print('num_below_threshold',num_below_threshold)
        
        # 计算小于阈值的比例
        percentage_below_threshold = num_below_threshold / num_residues

        # 如果比例大于percent，将该残基加入结果列表
        if percentage_below_threshold > percent:
            selected_residues.append(i)

    return selected_residues


def parse_pdb(filename, **kwargs):
    """extract xyz coords for all heavy atoms"""
    with open(filename,"r") as f:
        lines=f.readlines()
    return parse_pdb_lines(filename, lines, **kwargs)

def parse_pdb_lines(filename, lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res, pdb_idx = [],[]
    for l in lines:
        if l[:4] == "ATOM" and l[12:16].strip() == "CA":
            res.append((l[22:26], l[17:20]))
            # chain letter, res num
            pdb_idx.append((l[21:22].strip(), int(l[22:26].strip())))
    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [
        (l[21:22].strip(), int(l[22:26].strip()))
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22],
            int(l[22:26]),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        if (chain,resNo) in pdb_idx:
            idx = pdb_idx.index((chain, resNo))
            # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            for i_atm, tgtatm in enumerate(
                aa2long[aa2num[aa]][:14]
                ):
                if (
                    tgtatm is not None and tgtatm.strip() == atom.strip()
                    ):  # ignore whitespace
                    xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i, idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', filename)
    ca_distance, cb_distance, hotspot = calculate_ca_cb_distances(structure)
    print('hotspot',hotspot)
    print('length',len(new_idx))
    seq = np.array(seq)[i_unique]

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        "mask": mask,  # mask showing which atoms are present in the PDB file, [Lx14]
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        "pdb_idx": pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
        "hotspot": hotspot
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6] == "HETATM" and not (ignore_het_h and l[77] == "H"):
                info_het.append(
                    dict(
                        idx=int(l[7:11]),
                        atom_id=l[12:16],
                        atom_type=l[77],
                        name=l[16:20],
                    )
                )
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out["xyz_het"] = np.array(xyz_het)
        out["info_het"] = info_het

    return out

def process_target(pdb_path, parse_hetatom=False, center=True):
    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct["xyz"] - ca_center)
    seq_orig = torch.from_numpy(target_struct["seq"])
    atom_mask = torch.from_numpy(target_struct["mask"])
    seq_len = len(xyz)

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    xyz_27[:, :14, :] = xyz[:, :14, :]

    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :14] = atom_mask
    out = {
        "xyz_27": xyz_27,
        "mask_27": mask_27,
        "seq": seq_orig,
        "pdb_idx": target_struct["pdb_idx"],
        "hotspot": target_struct["hotspot"]
    }
    if parse_hetatom:
        out["xyz_het"] = target_struct["xyz_het"]
        out["info_het"] = target_struct["info_het"]
    return out


# def construct_contig(target_feats, contig_conf):
#     """
#     Construct contig class describing the protein to be generated
#     """
#     # self._log.info(f'Using contig: {self.contig_conf.contigs}')

#     return ContigMap(target_feats, **contig_conf)

def get_idx0_hotspots(mappings, ppi_conf, binderlen):
    """
    Take pdb-indexed hotspot resudes and the length of the binder, and makes the 0-indexed tensor of hotspots
    """

    hotspot_idx = None
    if binderlen > 0:
        # if ppi_conf.hotspot_res is not None:
        if ppi_conf['hotspot_res']:
            assert all(
                [i[0].isalpha() for i in ppi_conf.hotspot_res]
            ), "Hotspot residues need to be provided in pdb-indexed form. E.g. A100,A103"
            hotspots = [(i[0], int(i[1:])) for i in ppi_conf.hotspot_res]
            hotspot_idx = []
            for i, res in enumerate(mappings["receptor_con_ref_pdb_idx"]):
                if res in hotspots:
                    hotspot_idx.append(mappings["receptor_con_hal_idx0"][i])
    return hotspot_idx

def get_init_xyz(xyz_t):
    # input: xyz_t (B, T, L, 14, 3)
    # ouput: xyz (B, T, L, 14, 3)
    B, T, L = xyz_t.shape[:3]
    init = INIT_CRDS.to(xyz_t.device).reshape(1,1,1,27,3).repeat(B,T,L,1,1)
    if torch.isnan(xyz_t).all():
        return init

    mask = torch.isnan(xyz_t[:,:,:,:3]).any(dim=-1).any(dim=-1) # (B, T, L)
    #
    center_CA = ((~mask[:,:,:,None]) * torch.nan_to_num(xyz_t[:,:,:,1,:])).sum(dim=2) / ((~mask[:,:,:,None]).sum(dim=2)+1e-4) # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B,T,1,1,3)
    #
    idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = torch.where(~mask[i_b, i_T])[0] # (L_sub)
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap = torch.argmin(seqmap, dim=-1) # (L)
            idx = torch.gather(exist_in_templ, -1, seqmap) # (L)
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L,1).expand(-1,3))
            init[i_b,i_T] += offset_CA.reshape(L,1,3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz


parse_pdb('/data/1a2y.pdb')