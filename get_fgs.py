import itertools
import numpy as np
import pandas as pd

import networkx as nx
import torch
from rdkit import Chem
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.utils.convert import to_networkx
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.utils import dense_to_sparse
from collections import defaultdict

# smiles = 'NS(=O)(=O)c1ccc(CNC(=O)c2ccc(CN(Cc3ccc(F)cc3)S(=O)(=O)c3cc(Cl)cc(Cl)c3O)cc2)cc1'
# mol = Chem.MolFromSmiles(smiles)

# atomic_number = []
# for atom in mol.GetAtoms():
#     atom.SetProp("atomNote", str(atom.GetIdx()))
#     atomic_number.append(torch.tensor([atom.GetAtomicNum()]))

single_bond = list(torch.tensor([1., 0., 0., 0.]))
double_bond = list(torch.tensor([0., 1., 0., 0.]))
triple_bond = list(torch.tensor([0., 0., 1., 0.]))
aromatic_bond = list(torch.tensor([0., 0., 0., 1.]))
# bonds = {BT.SINGLE: single_bond, BT.DOUBLE: double_bond, BT.TRIPLE: triple_bond, BT.AROMATIC: aromatic_bond}
# bond_type = []   
# # for bond in mol.GetBonds():
# #     bond_type += 2 * [bonds[bond.GetBondType()]]

# num_atoms = mol.GetNumAtoms()
# for i in range(num_atoms):
#     for j in range(num_atoms):
#         if i != j:
#             bond = mol.GetBondBetweenAtoms(i, j)
#             if bond is not None:
                       
#                 bond_type.append(bonds[bond.GetBondType()])

# rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
# rows = torch.tensor(rows, dtype=torch.long)
# cols = torch.tensor(cols, dtype=torch.long)

# edge_index = torch.stack([rows, cols], dim = 0)
# data = Data(edge_index=edge_index)
# data.atomic_number = atomic_number
# data.bond_type = bond_type
# nx_g = to_networkx(data, node_attrs=['atomic_number'],edge_attrs=['bond_type'],to_undirected=True)
# nx_g = nx.Graph(nx_g)

"""
Separate atoms into element groups, cyclics and non cyclics
C, N, O, P or S
在现代元素周期表中，元素按原子序数递增的顺序排列。原子序数是原子核中的质子数。质子数决定了元素的特性
即，具有 6 个质子的元素是碳原子 
"""
carbon_atnum = torch.tensor([6])
nitrogen_atnum = torch.tensor([7])
oxygen_atnum = torch.tensor([8])
phosphorus_atnum = torch.tensor([15])
sulfur_atnum = torch.tensor([16])

# 返回中心原子在mol中的序号列表 G：nx图
def atom_groups(G):
    "get all atoms in atom groups"
    carbons = []
    nitrogens = []
    sulfurs = []
    oxygens = []
    phosphorus = []

    node_list = list(G.nodes())
    for node_i in node_list:
        if nx.get_node_attributes(G, 'atomic_number')[node_i] == carbon_atnum:
            carbons.append(node_i)
        elif nx.get_node_attributes(G, 'atomic_number')[node_i] == nitrogen_atnum:
            nitrogens.append(node_i)
        elif nx.get_node_attributes(G, 'atomic_number')[node_i] == sulfur_atnum:
            sulfurs.append(node_i)
        elif nx.get_node_attributes(G, 'atomic_number')[node_i] == oxygen_atnum:
            oxygens.append(node_i)
        elif nx.get_node_attributes(G, 'atomic_number')[node_i] == phosphorus_atnum:
            phosphorus.append(node_i)


    return carbons, nitrogens, oxygens, phosphorus, sulfurs


# 返回环中原子的序号列表
def cyclic_atoms(G, edge_name=None):
    edges_in_cycle = []  # edges between nodes in cycles
    nodes_in_cycle = []  # all nodes in cycles

    cycle_list = []  # nodes in cycle for each molecule
    for cycle_idx, nodes_in_cycle_list in enumerate(nx.cycle_basis(G)):
        "get nodes in cycles (cycle by cycle)"
        cycle_list.append(nodes_in_cycle_list)  # cycle list for each molecule

        "get edges between nodes in cycle"
        for j in range(len(cycle_list)):
            for k in range(len(cycle_list[j])):
                if G.has_edge(cycle_list[j][k], cycle_list[j][k - 1]) == True:
                    edge_in_cycle_j = [cycle_list[j][k], cycle_list[j][k - 1]]
                    edge_in_cycle_j.sort()
                    edges_in_cycle.append(edge_in_cycle_j)
                    edges_in_cycle.sort()
                    edges_in_cycle = list(
                        edges_in_cycle for edges_in_cycle, _ in itertools.groupby(edges_in_cycle))


        "all nodes in cycles"
        nodes_in_cycle = []  # nodes in cycle for each molecule
        for sublist in cycle_list:
            for item in sublist:
                nodes_in_cycle.append(item)
                nodes_in_cycle = list(dict.fromkeys(nodes_in_cycle))

    return cycle_list, edges_in_cycle, nodes_in_cycle


# 返回不在环中原子的序号列表
def noncyc_atoms(G):
    "get nodes not in cycles"
    (cycle_list, edges_in_cycle, nodes_in_cycle) = cyclic_atoms(G)
    noncyc_carbons = []
    noncyc_nitrogens = []
    noncyc_sulfurs = []
    noncyc_oxygens = []
    noncyc_phosphorus = []

    node_list = list(G.nodes())

    for node_i in node_list:
        if node_i not in nodes_in_cycle:
            if nx.get_node_attributes(G, 'atomic_number')[node_i] == carbon_atnum:
                noncyc_carbons.append(node_i)
            elif nx.get_node_attributes(G, 'atomic_number')[node_i] == nitrogen_atnum:
                noncyc_nitrogens.append(node_i)
            elif nx.get_node_attributes(G, 'atomic_number')[node_i] == sulfur_atnum:
                noncyc_sulfurs.append(node_i)
            elif nx.get_node_attributes(G, 'atomic_number')[node_i] == oxygen_atnum:
                noncyc_oxygens.append(node_i)
            elif nx.get_node_attributes(G, 'atomic_number')[node_i] == phosphorus_atnum:
                noncyc_phosphorus.append(node_i)

    return noncyc_carbons, noncyc_nitrogens, noncyc_sulfurs, noncyc_oxygens, noncyc_phosphorus

# 返回38种C, N, O, P, S作为中心原子的官能团中的原子列表
def func_groups(G):
    alkene = []
    alkyne = []
    allene = []
    carboxyl = []  # 1, 2, 9,fo
    ketene = []
    alcohol = []  # 2, 5, 12
    ketone = []  # 1, 35
    aldehyde = []
    ether = []  # 1, 2
    peroxide = []
    carbamate = []

    thioether = []
    disulfide = []  # 5, 11, 32
    sulfone = []
    thioamide = []
    thiourea = []
    thiol = []
    thione = []
    sulfoxide = []
    isothiocynate = []
    sulfonamide = []
    sulfonate = []

    amine = []  # 5, 11, 19
    amide = []  # 25, 35
    imine = []
    carbamide = []  # 23
    hydrazine = []
    nitrile = []
    hydrazone = []
    azo = []  # 23
    isocynate = []
    nitro = []  # 6, 20
    carbodiimide = []
    oxime = []
    c_nitroso = []
    hydroamine = []
    carboximidamide = []

    phosphorus_grps = []
    (carbons, nitrogens, oxygens, phosphorus, sulfurs) = atom_groups(G)
    (noncyc_carbons, noncyc_nitrogens, noncyc_sulfurs, noncyc_oxygens, noncyc_phosphorus) = noncyc_atoms(G)

    # 循环每个中心原子
    # C as the central atom
    for node_i in carbons:   
        neighbors_i = list(G.neighbors(node_i))  # 中心原子的邻居

        # for molecules with C=O
        if (True in (((list(G[node_i][neighbor_i]['bond_type']) == double_bond) and (
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum)) for neighbor_i in
                        neighbors_i)):
            "C double bonded to O"
            if len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 2:
                "C bonded to two Os"
                if (([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) and (
                        [neighbor_i for neighbor_i in neighbors_i if
                            list(G[node_i][neighbor_i]['bond_type']) == single_bond])):
                    "C single-bonded to C"
                    "-COO-"
                    for neighbor_i in neighbors_i:
                        if ((list(G[node_i][neighbor_i]['bond_type']) == single_bond) and (
                                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum)):
                            "get neighbors of single-bonded O"
                            all_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + all_neighbors_i))
                    carboxyl.append(nlist)  # carboxyl groups of each molecule

                elif (([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) and (
                                [neighbor_i for neighbor_i in neighbors_i if
                                list(G[node_i][neighbor_i]['bond_type']) == single_bond])):
                    "C single-bonded to N"
                    "meaning C bonded to one N and two Os"
                    "-NCOO-"
                    for neighbor_i in neighbors_i:
                        if ((list(G[node_i][neighbor_i]['bond_type']) == single_bond) and (
                                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum)):
                            all_n_neighbors = list(nx.all_neighbors(G, neighbor_i))
                    for neighbor_i in neighbors_i:
                        if ((list(G[node_i][neighbor_i]['bond_type']) == single_bond) and (
                                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum)):
                            all_o_neighbors = list(nx.all_neighbors(G, neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors + all_o_neighbors))
                    carbamate.append(nlist)
            elif len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 2:
                "C bonded to two Ns"
                "-NCON-"
                all_n_neighbors_i = []
                for neighbor_i in neighbors_i:
                    if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                        n_neighbor_i = list(nx.all_neighbors(G, neighbor_i))
                        for ni, n in enumerate(n_neighbor_i):
                            all_n_neighbors_i.append(n)
                nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                carbamide.append(nlist)

            elif (len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 1 and len(
                [neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1 and len(
                [neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 1):
                "C bonded to one C and one N and one O"
                "-NCO-"
                for neighbor_i in neighbors_i:
                    if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                        all_n_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                amide.append(nlist)

            elif len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 2:
                "C bonded to two Cs"
                "-CO-"
                nlist = list(dict.fromkeys([node_i] + neighbors_i))
                ketone.append(nlist)

            elif len(list(G.neighbors(node_i))) == 2:
                if (len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 1 and len(
                    [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1):
                    "C bonded to one O and one C"
                    if len([neighbor_i for neighbor_i in neighbors_i if
                            list(G[node_i][neighbor_i]['bond_type']) == single_bond]) == 1:
                        "-C=O"
                        nlist = list(dict.fromkeys([node_i] + neighbors_i))
                        aldehyde.append(nlist)

                    elif len([neighbor_i for neighbor_i in neighbors_i if
                                list(G[node_i][neighbor_i]['bond_type']) == double_bond]) == 2:
                        "-C=C=O"
                        if [neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]:
                            for neighbor_i in neighbors_i:
                                c_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                        nlist = list(dict.fromkeys([node_i] + neighbors_i + c_neighbors_i))
                        ketene.append(nlist)

                elif (len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[
                                neighbor_i] == nitrogen_atnum]) == 1 and len(
                    [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 1):
                    "C bonded to one N and one O"
                    "-N=C=O"
                    if [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]:
                        for neighbor_i in neighbors_i:
                            all_n_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                    isocynate.append(nlist)

        # for molecules with C=S
        elif (True in (((list(G[node_i][neighbor_i]['bond_type']) == double_bond) and (
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == sulfur_atnum)) for neighbor_i in
                        neighbors_i)):
            "C is double bonded to one S"
            "C=S"
            if len(list(G.neighbors(node_i))) == 2:
                if (len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == sulfur_atnum]) == 1 and len(
                    [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 1):
                    "C bonded to one S and one N"
                    if [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]:
                        for neighbor_i in neighbors_i:
                            all_n_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                    isothiocynate.append(nlist)

            elif len(list(G.neighbors(node_i))) == 3:
                if len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 2:
                    "C bonded to two Cs"
                    "-CS-"
                    nlist = list(dict.fromkeys([node_i] + neighbors_i))
                    thione.append(nlist)

                elif (len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[
                                neighbor_i] == nitrogen_atnum]) == 1 and len(
                    [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1):
                    "C bonded to one N and one C"
                    "-HN-CS-"
                    for neighbor_i in neighbors_i:
                        if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                            all_n_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                    thioamide.append(nlist)

                elif len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 2:
                    "meaning C bonded to two Ns"
                    all_n_neighbors_i = []
                    for neighbor_i in neighbors_i:
                        if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                            n_neighbor_i = list(nx.all_neighbors(G, neighbor_i))
                            for ni, n in enumerate(n_neighbor_i):
                                all_n_neighbors_i.append(n)
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                    thiourea.append(nlist)

        # for molecules with C=N
        elif (True in (((list(G[node_i][neighbor_i]['bond_type']) == double_bond) and (
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum)) for neighbor_i in
                        neighbors_i)):
            "meaning C is double bonded to N"
            if len([neighbor_i for neighbor_i in neighbors_i if
                    list(G[node_i][neighbor_i]['bond_type']) == double_bond]) == 1:
                "1 double bond"
                for neighbor_i in neighbors_i:
                    if len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 1:
                        "meaning C only bonded to one N"
                        if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                            "fishing N out"
                            all_n_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
                            for n_neighbor_i in all_n_neighbors_i:
                                "fishing neighbors of N"
                                if nx.get_node_attributes(G, 'atomic_number')[n_neighbor_i] == oxygen_atnum:
                                    "meaning N bonded to O"
                                    all_c_neighbors_i = list(nx.all_neighbors(G, node_i))
                                    all_o_neighbors_i = list(nx.all_neighbors(G, n_neighbor_i))
                                    nlist = list(
                                        dict.fromkeys(all_c_neighbors_i + all_n_neighbors_i + all_o_neighbors_i))
                                    oxime.append(nlist)
                                elif nx.get_node_attributes(G, 'atomic_number')[n_neighbor_i] == nitrogen_atnum:
                                    "meaning N bonded to N"
                                    all_c_neighbors_i = list(nx.all_neighbors(G, node_i))
                                    all_n2_neighbors_i = list(nx.all_neighbors(G, n_neighbor_i))
                                    nlist = list(
                                        dict.fromkeys(all_c_neighbors_i + all_n_neighbors_i + all_n2_neighbors_i))
                                    hydrazone.append(nlist)
                                else:
                                    all_c_neighbors_i = list(nx.all_neighbors(G, node_i))
                                    nlist = list(dict.fromkeys(all_c_neighbors_i + all_n_neighbors_i))
                                    imine.append(nlist)

                    elif len([neighbor_i for neighbor_i in neighbors_i if
                                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 3:
                        "meaning C bonded to three Ns"
                        n_neighbors_i = []
                        for j in range(len(neighbors_i)):
                            if nx.get_node_attributes(G, 'atomic_number')[neighbors_i[j]] == nitrogen_atnum:
                                n_neighbor_i = list(nx.all_neighbors(G, neighbors_i[j]))
                                n_neighbors_i.append(n_neighbor_i)
                        all_n_neighbors_i = []
                        for sublist in n_neighbors_i:
                            for n in sublist:
                                all_n_neighbors_i.append(n)
                        nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                        carboximidamide.append(nlist)
                        carboximidamide.sort()
                        carboximidamide = list(
                            carboximidamide for carboximidamide, _ in itertools.groupby(carboximidamide))

            elif len([neighbor_i for neighbor_i in neighbors_i if
                        list(G[node_i][neighbor_i]['bond_type']) == double_bond]) == 2:
                "2 double bonds"
                n_neighbors_i = []
                for j in range(len(neighbors_i)):
                    if nx.get_node_attributes(G, 'atomic_number')[neighbors_i[j]] == nitrogen_atnum:
                        n_neighbor_i = list(nx.all_neighbors(G, neighbors_i[j]))
                        n_neighbors_i.append(n_neighbor_i)
                all_n_neighbors_i = []
                for sublist in n_neighbors_i:
                    for n in sublist:
                        all_n_neighbors_i.append(n)
                nlist = list(dict.fromkeys([node_i] + neighbors_i + all_n_neighbors_i))
                carbodiimide.append(nlist)

        # C single bonded to O (aka OH)in
        elif (True in (((list(G[node_i][neighbor_i]['bond_type']) == single_bond) and (
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum)) for neighbor_i in
                        neighbors_i)):
            "meaning C single bonded to one O"
            all_c_neighbors_i = list(nx.all_neighbors(G, node_i))
            for c_neighbor_i in all_c_neighbors_i:
                "fishing neighbors of C"
                if (nx.get_node_attributes(G, 'atomic_number')[c_neighbor_i] == oxygen_atnum) and len(
                        list(G.neighbors(c_neighbor_i))) == 1:
                    "meaning C bonded to one O and the O only has one neighbor"
                    "-COH"
                    if len([neighbor_i for neighbor_i in neighbors_i if
                            list(G[node_i][c_neighbor_i]['bond_type']) == double_bond]) == 0:
                        "meaning no double bonds between node and neighbors"
                        alcohol.append([node_i] + all_c_neighbors_i)


        # C single bonded to S (aka SH)
        elif (True in (((list(G[node_i][neighbor_i]['bond_type']) == single_bond) and (
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == sulfur_atnum)) for neighbor_i in
                        neighbors_i)):
            "meaning C single bonded to one S"
            "-CS"
            all_c_neighbors_i = list(nx.all_neighbors(G, node_i))
            for c_neighbor_i in all_c_neighbors_i:
                "fishing neighbors of C"
                if (True in ((nx.get_node_attributes(G, 'atomic_number')[c_neighbor_i] == sulfur_atnum) and len(
                        list(G.neighbors(c_neighbor_i))) == 1 for neighbor_i in neighbors_i)):
                    thiol.append([node_i] + all_c_neighbors_i)

        # C double bonded to two Cs
        elif (len(neighbors_i) == 2) and (len([neighbor_i for neighbor_i in neighbors_i if list(
                G[node_i][neighbor_i]['bond_type']) == double_bond]) == 2) and (len(
            [neighbor_i for neighbor_i in neighbors_i if
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 2):
            "meaning C double bonded to two Cs"
            "-C=C=C-"
            all_c_neighbors_i = []
            for neighbor_i in neighbors_i:
                c1_neighbors = list(G.neighbors(neighbors_i[0]))
                c2_neighbors = list(G.neighbors(neighbors_i[1]))
                c1_neighbors.remove(node_i)
                c2_neighbors.remove(node_i)
                all_c_neighbors_i = c1_neighbors + c2_neighbors
            nlist = list(dict.fromkeys([node_i] + neighbors_i + all_c_neighbors_i))
            allene.append(nlist)

        # C triple bond C
        elif (all(nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum for neighbor_i in
                    neighbors_i)) and (len([neighbor_i for neighbor_i in neighbors_i if
                                            list(G[node_i][neighbor_i]['bond_type']) == triple_bond]) == 1):
            for neighbor_i in neighbors_i:
                if list(G[node_i][neighbor_i]['bond_type']) == triple_bond:
                    c_neighbor_i = list(G.neighbors(neighbor_i))
            nlist = list(dict.fromkeys([node_i] + neighbors_i + c_neighbor_i))
            nlist.sort()
            alkyne.append(nlist)
            alkyne.sort()
            alkyne = list(alkyne for alkyne, _ in itertools.groupby(alkyne))

    # O as main atom in molecule
    for node_i in oxygens:
        neighbors_i = list(G.neighbors(node_i))
        if len(neighbors_i) == 2:
            "meaning O only bonded to two atoms"
            c_neighbors_i = []
            all_c_neighbors_i = []
            if len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 2:
                "meaning O bonded to 2 C"
                c1_neighbors = list(G.neighbors(neighbors_i[0]))
                c2_neighbors = list(G.neighbors(neighbors_i[1]))
                c1_neighbors.remove(node_i)
                c2_neighbors.remove(node_i)
                all_c_neighbors_i = c1_neighbors + c2_neighbors
                if (any(node_i in sl for sl in carboxyl)) == False:
                    "makes sure -O- is not from carboxyl group"
                    nlist = list(dict.fromkeys([node_i] + neighbors_i))
                    ether.append(nlist)
                ether.sort()
                ether = list(ether for ether, _ in itertools.groupby(ether))

            elif (len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1) and len(
                [neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 1:
                "meaning O bonded to one O and one C"
                for neighbor_i in neighbors_i:
                    if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum:
                        o_neighbor_i = list(G.neighbors(neighbor_i))
                nlist = list(dict.fromkeys([node_i] + neighbors_i + o_neighbor_i))
                nlist.sort()
                peroxide.append(nlist)
                peroxide.sort()
                peroxide = list(peroxide for peroxide, _ in itertools.groupby(peroxide))

    # S as main atom in molecule
    for node_i in sulfurs:
        neighbors_i = list(G.neighbors(node_i))
        if len([neighbor_i for neighbor_i in neighbors_i if
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 2:
            "meaning S bonded to 2 C"
            if len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 0:
                thioether.append([node_i] + neighbors_i)
            elif len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 1:
                "S bonded to one O"
                sulfoxide.append([node_i] + neighbors_i)
            elif len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 2:
                "S bonded to two Os"
                sulfone.append([node_i] + neighbors_i)

        elif len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1 and len(
            [neighbor_i for neighbor_i in neighbors_i if
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 2 and len(
            [neighbor_i for neighbor_i in neighbors_i if
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 1:
            "S bonded to two Os, one C and one N"
            for neighbor_i in neighbors_i:
                if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                    all_n_neighbors_i = list(nx.all_neighbors(G, neighbor_i))
            nlist = list(dict.fromkeys(neighbors_i + all_n_neighbors_i))
            sulfonamide.append(nlist)

        elif (len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 3) and (len(
            [neighbor_i for neighbor_i in neighbors_i if
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1):
            "meaning S bonded to three Os and one C"
            for neighbor_i in neighbors_i:
                if (nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum) and (
                        list(G[node_i][neighbor_i]['bond_type']) == single_bond):
                    o_neighbor_i = list(G.neighbors(neighbor_i))
            nlist = list(dict.fromkeys([node_i] + neighbors_i + o_neighbor_i))
            sulfonate.append(nlist)

        elif len(neighbors_i) == 2:
            if (len([neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1) and len(
                [neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == sulfur_atnum]) == 1:
                "meaning S bonded to one S and one C"
                for neighbor_i in neighbors_i:
                    if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == sulfur_atnum:
                        s_neighbor_i = list(G.neighbors(neighbor_i))
                nlist = list(dict.fromkeys([node_i] + neighbors_i + s_neighbor_i))
                nlist.sort()
                disulfide.append(nlist)
                disulfide.sort()
                disulfide = list(disulfide for disulfide, _ in itertools.groupby(disulfide))

    # N as main atom in molecule
    for node_i in nitrogens:
        neighbors_i = list(G.neighbors(node_i))

        if all(nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum for neighbor_i in
                neighbors_i) and all(
            list(G[node_i][neighbor_i]['bond_type']) == single_bond for neighbor_i in neighbors_i):
            "meaning N is only single bonded to C or H"
            if (any(node_i in sl for sl in thioamide)) == False and (
                    any(node_i in sl for sl in sulfonamide)) == False and (
                    any(node_i in sl for sl in amide)) == False and (
                    any(node_i in sl for sl in carbamide)) == False and (
                    any(node_i in sl for sl in carbamate)) == False:
                "makes sure that there is no overlap to the other groups"
                nlist = list(dict.fromkeys([node_i] + neighbors_i))
                amine.append(nlist)

        elif all(nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum for neighbor_i in
                    neighbors_i) and all(
            list(G[node_i][neighbor_i]['bond_type']) == triple_bond for neighbor_i in neighbors_i):
            "meaning N triple bonded to C"
            all_c_neighbors_i = []
            c_neighbors_i = [list(G.neighbors(neighbor_i)) for neighbor_i in neighbors_i]
            for sublist in c_neighbors_i:
                for item in sublist:
                    all_c_neighbors_i.append(item)
            nlist = list(dict.fromkeys([node_i] + neighbors_i + all_c_neighbors_i))
            nitrile.append(nlist)

        elif len(neighbors_i) == 2:
            if all(list(G[node_i][neighbor_i]['bond_type']) == single_bond for neighbor_i in neighbors_i):
                "meaning N only single bonded to neighbors"
                if (len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1) and len(
                    [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 1:
                    "meaning N bonded to one N and one C"
                    for neighbor_i in neighbors_i:
                        if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum:
                            n_neighbor_i = list(G.neighbors(neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + n_neighbor_i))
                    nlist.sort()
                    hydrazine.append(nlist)
                    hydrazine.sort()
                    hydrazine = list(hydrazine for hydrazine, _ in itertools.groupby(hydrazine))

            elif len([neighbor_i for neighbor_i in neighbors_i if
                        list(G[node_i][neighbor_i]['bond_type']) == double_bond]) == 1:
                "meaning only one double bond"
                if (len([neighbor_i for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1) and len(
                    [neighbor_i for neighbor_i in neighbors_i if
                        nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == nitrogen_atnum]) == 1:
                    "meaning N bonded to one N and one C "
                    for neighbor_i in neighbors_i:
                        if list(G[node_i][neighbor_i]['bond_type']) == double_bond:
                            n_neighbor_i = list(G.neighbors(neighbor_i))
                    nlist = list(dict.fromkeys([node_i] + neighbors_i + n_neighbor_i))
                    nlist.sort()
                    azo.append(nlist)
                    azo.sort()
                    azo = list(azo for azo, _ in itertools.groupby(azo))

                elif [(nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum and list(
                        G[node_i][neighbor_i]['bond_type']) == double_bond) for neighbor_i in neighbors_i]:
                    "meaning N double bonded to O"
                    c_nitroso.append([node_i] + neighbors_i)

        elif len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 1:
            "meaning N bonded to one O"
            if len([neighbor_i for neighbor_i in neighbors_i if
                    list(G[node_i][neighbor_i]['bond_type']) == double_bond]) == 0:
                "meaning no double bonds"
                for neighbor_i in neighbors_i:
                    if nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum:
                        all_o_neighbors_i = list(G.neighbors(neighbor_i))
                nlist = list(dict.fromkeys([node_i] + neighbors_i + all_o_neighbors_i))
                hydroamine.append(nlist)

        elif len([neighbor_i for neighbor_i in neighbors_i if
                    nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == oxygen_atnum]) == 2 and len(
            [neighbor_i for neighbor_i in neighbors_i if
                nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum]) == 1:
            "meaning N bonded to two Os and one C"
            if len(neighbors_i) == 3:
                nitro.append([node_i] + neighbors_i)

    # P as main atom in molecule
    for node_i in phosphorus:
        neighbors_i = list(G.neighbors(node_i))
        all_x_neighbors_i = []
        x_neighbors_i = [list(G.neighbors(neighbor_i)) for neighbor_i in neighbors_i if
                            nx.get_node_attributes(G, 'atomic_number')[neighbor_i] != carbon_atnum]
        for sublist in x_neighbors_i:
            for item in sublist:
                all_x_neighbors_i.append(item)
        nlist = list(dict.fromkeys([node_i] + neighbors_i + all_x_neighbors_i))
        phosphorus_grps.append(nlist)

    for node_i in noncyc_carbons:
        neighbors_i = list(G.neighbors(node_i))

        # C double bond C
        if (all(nx.get_node_attributes(G, 'atomic_number')[neighbor_i] == carbon_atnum for neighbor_i in
                neighbors_i)) and (len([neighbor_i for neighbor_i in neighbors_i if
                                        list(G[node_i][neighbor_i]['bond_type']) == double_bond]) == 1):
            for neighbor_i in neighbors_i:
                if list(G[node_i][neighbor_i]['bond_type']) == double_bond:
                    c_neighbor_i = list(G.neighbors(neighbor_i))
            nlist = list(dict.fromkeys([node_i] + neighbors_i + c_neighbor_i))
            nlist.sort()
            alkene.append(nlist)
            alkene.sort()
            alkene = list(alkene for alkene, _ in itertools.groupby(alkene))



    return carboxyl, carbamate, carbamide, amide, ketone, aldehyde, ketene, isocynate, isothiocynate, thione, thioamide, thiourea, oxime, hydrazone, carboximidamide, imine, carbodiimide, alcohol, thiol, allene, alkyne, ether, peroxide, thioether, sulfoxide, sulfone, sulfonamide, sulfonate, disulfide, amine, nitrile, hydrazine, azo, c_nitroso, hydroamine, nitro, phosphorus_grps, alkene

# 官能团所在边
def edges_in_func_groups(G):
    edges_in_alkene = []
    edges_in_alkyne = []
    edges_in_allene = []
    edges_in_carboxyl = []
    edges_in_ketene = []
    edges_in_alcohol = []
    edges_in_ketone = []
    edges_in_aldehyde = []
    edges_in_ether = []
    edges_in_peroxide = []
    edges_in_carbamate = []

    edges_in_thioether = []
    edges_in_disulfide = []
    edges_in_sulfone = []
    edges_in_thioamide = []
    edges_in_thiourea = []
    edges_in_thiol = []
    edges_in_thione = []
    edges_in_sulfoxide = []
    edges_in_isothiocynate = []
    edges_in_sulfonamide = []
    edges_in_sulfonate = []

    edges_in_amine = []
    edges_in_hydroamine = []
    edges_in_amide = []
    edges_in_imine = []
    edges_in_carbamide = []
    edges_in_nitrile = []
    edges_in_hydrazine = []
    edges_in_hydrazone = []
    edges_in_azo = []
    edges_in_isocynate = []
    edges_in_nitro = []
    edges_in_carbodiimide = []
    edges_in_oxime = []
    edges_in_c_nitroso = []
    edges_in_carboximidamide = []

    edges_in_phosphorus_grps = []
    (carboxyl, carbamate, carbamide, amide, ketone, aldehyde, ketene, isocynate, isothiocynate, thione, thioamide,
     thiourea, oxime, hydrazone, carboximidamide, imine, carbodiimide, alcohol, thiol, allene, alkyne, ether, peroxide,
     thioether, sulfoxide, sulfone, sulfonamide, sulfonate, disulfide, amine, nitrile, hydrazine, azo, c_nitroso,
     hydroamine, nitro, phosphorus_grps, alkene) = func_groups(G)


    for alkene_idx, nodes_in_alkene in enumerate(alkene):
        pairs = list(itertools.combinations(nodes_in_alkene, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_alkene = [u, v]
                edge_in_alkene.sort()
                edges_in_alkene.append(edge_in_alkene)
        edges_in_alkene.sort()
        edges_in_alkene = list(edges_in_alkene for edges_in_alkene, _ in itertools.groupby(edges_in_alkene))


    for alkyne_idx, nodes_in_alkyne in enumerate(alkyne):
        pairs = list(itertools.combinations(nodes_in_alkyne, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_alkyne = [u, v]
                edge_in_alkyne.sort()
                edges_in_alkyne.append(edge_in_alkyne)
        edges_in_alkyne.sort()
        edges_in_alkyne = list(edges_in_alkyne for edges_in_alkyne, _ in itertools.groupby(edges_in_alkyne))

    for allene_idx, nodes_in_allene in enumerate(allene):
        pairs = list(itertools.combinations(nodes_in_allene, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_allene = [u, v]
                edge_in_allene.sort()
                edges_in_allene.append(edge_in_allene)
        edges_in_allene.sort()
        edges_in_allene = list(edges_in_allene for edges_in_allene, _ in itertools.groupby(edges_in_allene))

    for carboxyl_idx, nodes_in_carboxyl in enumerate(carboxyl):
        pairs = list(itertools.combinations(nodes_in_carboxyl, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_carboxyl = [u, v]
                edge_in_carboxyl.sort()
                edges_in_carboxyl.append(edge_in_carboxyl)
        edges_in_carboxyl.sort()
        edges_in_carboxyl = list(
            edges_in_carboxyl for edges_in_carboxyl, _ in itertools.groupby(edges_in_carboxyl))

    for ketene_idx, nodes_in_ketene in enumerate(ketene):
        pairs = list(itertools.combinations(nodes_in_ketene, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_ketene = [u, v]
                edge_in_ketene.sort()
                edges_in_ketene.append(edge_in_ketene)
        edges_in_ketene.sort()
        edges_in_ketene = list(edges_in_ketene for edges_in_ketene, _ in itertools.groupby(edges_in_ketene))

    for alcohol_idx, nodes_in_alcohol in enumerate(alcohol):
        pairs = list(itertools.combinations(nodes_in_alcohol, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_alcohol = [u, v]
                edge_in_alcohol.sort()
                edges_in_alcohol.append(edge_in_alcohol)
        edges_in_alcohol.sort()
        edges_in_alcohol = list(
            edges_in_alcohol for edges_in_alcohol, _ in itertools.groupby(edges_in_alcohol))

    for ketone_idx, nodes_in_ketone in enumerate(ketone):
        pairs = list(itertools.combinations(nodes_in_ketone, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_ketone = [u, v]
                edge_in_ketone.sort()
                edges_in_ketone.append(edge_in_ketone)
        edges_in_ketone.sort()
        edges_in_ketone = list(edges_in_ketone for edges_in_ketone, _ in itertools.groupby(edges_in_ketone))

    for aldehyde_idx, nodes_in_aldehyde in enumerate(aldehyde):
        pairs = list(itertools.combinations(nodes_in_aldehyde, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_aldehyde = [u, v]
                edge_in_aldehyde.sort()
                edges_in_aldehyde.append(edge_in_aldehyde)
        edges_in_aldehyde.sort()
        edges_in_aldehyde = list(
            edges_in_aldehyde for edges_in_aldehyde, _ in itertools.groupby(edges_in_aldehyde))

    for ether_idx, nodes_in_ether in enumerate(ether):
        pairs = list(itertools.combinations(nodes_in_ether, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_ether = [u, v]
                edge_in_ether.sort()
                edges_in_ether.append(edge_in_ether)
        edges_in_ether.sort()
        edges_in_ether = list(edges_in_ether for edges_in_ether, _ in itertools.groupby(edges_in_ether))

    for peroxide_idx, nodes_in_peroxide in enumerate(peroxide):
        pairs = list(itertools.combinations(nodes_in_peroxide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_peroxide = [u, v]
                edge_in_peroxide.sort()
                edges_in_peroxide.append(edge_in_peroxide)
        edges_in_peroxide.sort()
        edges_in_peroxide = list(
            edges_in_peroxide for edges_in_peroxide, _ in itertools.groupby(edges_in_peroxide))

    for carbamate_idx, nodes_in_carbamate in enumerate(carbamate):
        pairs = list(itertools.combinations(nodes_in_carbamate, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_carbamate = [u, v]
                edge_in_carbamate.sort()
                edges_in_carbamate.append(edge_in_carbamate)
        edges_in_carbamate.sort()
        edges_in_carbamate = list(
            edges_in_carbamate for edges_in_carbamate, _ in itertools.groupby(edges_in_carbamate))

    for thioether_idx, nodes_in_thioether in enumerate(thioether):
        pairs = list(itertools.combinations(nodes_in_thioether, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_thioether = [u, v]
                edge_in_thioether.sort()
                edges_in_thioether.append(edge_in_thioether)
        edges_in_thioether.sort()
        edges_in_thioether = list(
            edges_in_thioether for edges_in_thioether, _ in itertools.groupby(edges_in_thioether))

    for disulfide_idx, nodes_in_disulfide in enumerate(disulfide):
        pairs = list(itertools.combinations(nodes_in_disulfide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_disulfide = [u, v]
                edge_in_disulfide.sort()
                edges_in_disulfide.append(edge_in_disulfide)
        edges_in_disulfide.sort()
        edges_in_disulfide = list(
            edges_in_disulfide for edges_in_disulfide, _ in itertools.groupby(edges_in_disulfide))

    for sulfone_idx, nodes_in_sulfone in enumerate(sulfone):
        pairs = list(itertools.combinations(nodes_in_sulfone, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_sulfone = [u, v]
                edge_in_sulfone.sort()
                edges_in_sulfone.append(edge_in_sulfone)
        edges_in_sulfone.sort()
        edges_in_sulfone = list(
            edges_in_sulfone for edges_in_sulfone, _ in itertools.groupby(edges_in_sulfone))

    for thioamide_idx, nodes_in_thioamide in enumerate(thioamide):
        pairs = list(itertools.combinations(nodes_in_thioamide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_thioamide = [u, v]
                edge_in_thioamide.sort()
                edges_in_thioamide.append(edge_in_thioamide)
        edges_in_thioamide.sort()
        edges_in_thioamide = list(
            edges_in_thioamide for edges_in_thioamide, _ in itertools.groupby(edges_in_thioamide))

    for thiourea_idx, nodes_in_thiourea in enumerate(thiourea):
        pairs = list(itertools.combinations(nodes_in_thiourea, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_thiourea = [u, v]
                edge_in_thiourea.sort()
                edges_in_thiourea.append(edge_in_thiourea)
        edges_in_thiourea.sort()
        edges_in_thiourea = list(
            edges_in_thiourea for edges_in_thiourea, _ in itertools.groupby(edges_in_thiourea))

    for thiol_idx, nodes_in_thiol in enumerate(thiol):
        pairs = list(itertools.combinations(nodes_in_thiol, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_thiol = [u, v]
                edge_in_thiol.sort()
                edges_in_thiol.append(edge_in_thiol)
        edges_in_thiol.sort()
        edges_in_thiol = list(edges_in_thiol for edges_in_thiol, _ in itertools.groupby(edges_in_thiol))

    for thione_idx, nodes_in_thione in enumerate(thione):
        pairs = list(itertools.combinations(nodes_in_thione, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_thione = [u, v]
                edge_in_thione.sort()
                edges_in_thione.append(edge_in_thione)
        edges_in_thione.sort()
        edges_in_thione = list(edges_in_thione for edges_in_thione, _ in itertools.groupby(edges_in_thione))

    for sulfoxide_idx, nodes_in_sulfoxide in enumerate(sulfoxide):
        pairs = list(itertools.combinations(nodes_in_sulfoxide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_sulfoxide = [u, v]
                edge_in_sulfoxide.sort()
                edges_in_sulfoxide.append(edge_in_sulfoxide)
        edges_in_sulfoxide.sort()
        edges_in_sulfoxide = list(
            edges_in_sulfoxide for edges_in_sulfoxide, _ in itertools.groupby(edges_in_sulfoxide))

    for isothiocynate_idx, nodes_in_isothiocynate in enumerate(isothiocynate):
        pairs = list(itertools.combinations(nodes_in_isothiocynate, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_isothiocynate = [u, v]
                edge_in_isothiocynate.sort()
                edges_in_isothiocynate.append(edge_in_isothiocynate)
        edges_in_isothiocynate.sort()
        edges_in_isothiocynate = list(
            edges_in_isothiocynate for edges_in_isothiocynate, _ in itertools.groupby(edges_in_isothiocynate))

    for sulfonamide_idx, nodes_in_sulfonamide in enumerate(sulfonamide):
        pairs = list(itertools.combinations(nodes_in_sulfonamide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_sulfonamide = [u, v]
                edge_in_sulfonamide.sort()
                edges_in_sulfonamide.append(edge_in_sulfonamide)
        edges_in_sulfonamide.sort()
        edges_in_sulfonamide = list(
            edges_in_sulfonamide for edges_in_sulfonamide, _ in itertools.groupby(edges_in_sulfonamide))

    for sulfonate_idx, nodes_in_sulfonate in enumerate(sulfonate):
        pairs = list(itertools.combinations(nodes_in_sulfonate, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_sulfonate = [u, v]
                edge_in_sulfonate.sort()
                edges_in_sulfonate.append(edge_in_sulfonate)
        edges_in_sulfonate.sort()
        edges_in_sulfonate = list(
            edges_in_sulfonate for edges_in_sulfonate, _ in itertools.groupby(edges_in_sulfonate))

    for amine_idx, nodes_in_amine in enumerate(amine):
        pairs = list(itertools.combinations(nodes_in_amine, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_amine = [u, v]
                edge_in_amine.sort()
                edges_in_amine.append(edge_in_amine)
        edges_in_amine.sort()
        edges_in_amine = list(edges_in_amine for edges_in_amine, _ in itertools.groupby(edges_in_amine))

    for hydroamine_idx, nodes_in_hydroamine in enumerate(hydroamine):
        pairs = list(itertools.combinations(nodes_in_hydroamine, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_hydroamine = [u, v]
                edge_in_hydroamine.sort()
                edges_in_hydroamine.append(edge_in_hydroamine)
        edges_in_hydroamine.sort()
        edges_in_hydroamine = list(
            edges_in_hydroamine for edges_in_hydroamine, _ in itertools.groupby(edges_in_hydroamine))

    for amide_idx, nodes_in_amide in enumerate(amide):
        pairs = list(itertools.combinations(nodes_in_amide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_amide = [u, v]
                edge_in_amide.sort()
                edges_in_amide.append(edge_in_amide)
        edges_in_amide.sort()
        edges_in_amide = list(edges_in_amide for edges_in_amide, _ in itertools.groupby(edges_in_amide))

    for imine_idx, nodes_in_imine in enumerate(imine):
        pairs = list(itertools.combinations(nodes_in_imine, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_imine = [u, v]
                edge_in_imine.sort()
                edges_in_imine.append(edge_in_imine)
        edges_in_imine.sort()
        edges_in_imine = list(edges_in_imine for edges_in_imine, _ in itertools.groupby(edges_in_imine))

    for carbamide_idx, nodes_in_carbamide in enumerate(carbamide):
        pairs = list(itertools.combinations(nodes_in_carbamide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_carbamide = [u, v]
                edge_in_carbamide.sort()
                edges_in_carbamide.append(edge_in_carbamide)
        edges_in_carbamide.sort()
        edges_in_carbamide = list(
            edges_in_carbamide for edges_in_carbamide, _ in itertools.groupby(edges_in_carbamide))

    for nitrile_idx, nodes_in_nitrile in enumerate(nitrile):
        pairs = list(itertools.combinations(nodes_in_nitrile, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_nitrile = [u, v]
                edge_in_nitrile.sort()
                edges_in_nitrile.append(edge_in_nitrile)
        edges_in_nitrile.sort()
        edges_in_nitrile = list(
            edges_in_nitrile for edges_in_nitrile, _ in itertools.groupby(edges_in_nitrile))

    for hydrazine_idx, nodes_in_hydrazine in enumerate(hydrazine):
        pairs = list(itertools.combinations(nodes_in_hydrazine, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_hydrazine = [u, v]
                edge_in_hydrazine.sort()
                edges_in_hydrazine.append(edge_in_hydrazine)
        edges_in_hydrazine.sort()
        edges_in_hydrazine = list(
            edges_in_hydrazine for edges_in_hydrazine, _ in itertools.groupby(edges_in_hydrazine))

    for hydrazone_idx, nodes_in_hydrazone in enumerate(hydrazone):
        pairs = list(itertools.combinations(nodes_in_hydrazone, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_hydrazone = [u, v]
                edge_in_hydrazone.sort()
                edges_in_hydrazone.append(edge_in_hydrazone)
        edges_in_hydrazone.sort()
        edges_in_hydrazone = list(
            edges_in_hydrazone for edges_in_hydrazone, _ in itertools.groupby(edges_in_hydrazone))

    for carboximidamide_idx, nodes_in_carboximidamide in enumerate(carboximidamide):
        pairs = list(itertools.combinations(nodes_in_carboximidamide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_carboximidamide = [u, v]
                edge_in_carboximidamide.sort()
                edges_in_carboximidamide.append(edge_in_carboximidamide)
        edges_in_carboximidamide.sort()
        edges_in_carboximidamide = list(edges_in_carboximidamide for edges_in_carboximidamide, _ in
                                            itertools.groupby(edges_in_carboximidamide))

    for azo_idx, nodes_in_azo in enumerate(azo):
        pairs = list(itertools.combinations(nodes_in_azo, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_azo = [u, v]
                edge_in_azo.sort()
                edges_in_azo.append(edge_in_azo)
        edges_in_azo.sort()
        edges_in_azo = list(edges_in_azo for edges_in_azo, _ in itertools.groupby(edges_in_azo))
    
    for isocynate_idx, nodes_in_isocynate in enumerate(isocynate):
        pairs = list(itertools.combinations(nodes_in_isocynate, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_isocynate = [u, v]
                edge_in_isocynate.sort()
                edges_in_isocynate.append(edge_in_isocynate)
        edges_in_isocynate.sort()
        edges_in_isocynate = list(
            edges_in_isocynate for edges_in_isocynate, _ in itertools.groupby(edges_in_isocynate))

    for nitro_idx, nodes_in_nitro in enumerate(nitro):
        pairs = list(itertools.combinations(nodes_in_nitro, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_nitro = [u, v]
                edge_in_nitro.sort()
                edges_in_nitro.append(edge_in_nitro)
        edges_in_nitro.sort()
        edges_in_nitro = list(edges_in_nitro for edges_in_nitro, _ in itertools.groupby(edges_in_nitro))

    for carbodiimide_idx, nodes_in_carbodiimide in enumerate(carbodiimide):
        pairs = list(itertools.combinations(nodes_in_carbodiimide, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_carbodiimide = [u, v]
                edge_in_carbodiimide.sort()
                edges_in_carbodiimide.append(edge_in_carbodiimide)
        edges_in_carbodiimide.sort()
        edges_in_carbodiimide = list(
            edges_in_carbodiimide for edges_in_carbodiimide, _ in itertools.groupby(edges_in_carbodiimide))

    for oxime_idx, nodes_in_oxime in enumerate(oxime):
        pairs = list(itertools.combinations(nodes_in_oxime, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_oxime = [u, v]
                edge_in_oxime.sort()
                edges_in_oxime.append(edge_in_oxime)
        edges_in_oxime.sort()
        edges_in_oxime = list(edges_in_oxime for edges_in_oxime, _ in itertools.groupby(edges_in_oxime))

    for c_nitroso_idx, nodes_in_c_nitroso in enumerate(c_nitroso):
        pairs = list(itertools.combinations(nodes_in_c_nitroso, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_c_nitroso = [u, v]
                edge_in_c_nitroso.sort()
                edges_in_c_nitroso.append(edge_in_c_nitroso)
        edges_in_c_nitroso.sort()
        edges_in_c_nitroso = list(
            edges_in_c_nitroso for edges_in_c_nitroso, _ in itertools.groupby(edges_in_c_nitroso))

    for phosphorus_grps_idx, nodes_in_phosphorus_grps in enumerate(phosphorus_grps):
        pairs = list(itertools.combinations(nodes_in_phosphorus_grps, 2))
        for (u, v) in pairs:
            if G.has_edge(u, v) == True:
                edge_in_phosphorus_grps = [u, v]
                edge_in_phosphorus_grps.sort()
                edges_in_phosphorus_grps.append(edge_in_phosphorus_grps)
        edges_in_phosphorus_grps.sort()
        edges_in_phosphorus_grps = list(edges_in_phosphorus_grps for edges_in_phosphorus_grps, _ in
                                            itertools.groupby(edges_in_phosphorus_grps))

    return edges_in_alkene, edges_in_alkyne, edges_in_allene, edges_in_carboxyl, edges_in_ketene, edges_in_alcohol, edges_in_ketone, edges_in_aldehyde, edges_in_ether, edges_in_peroxide, edges_in_carbamate, edges_in_thioether, edges_in_disulfide, edges_in_sulfone, edges_in_thioamide, edges_in_thiourea, edges_in_thiol, edges_in_thione, edges_in_sulfoxide, edges_in_isothiocynate, edges_in_sulfonamide, edges_in_sulfonate, edges_in_amine, edges_in_hydroamine, edges_in_amide, edges_in_imine, edges_in_carbamide, edges_in_nitrile, edges_in_hydrazine, edges_in_hydrazone, edges_in_azo, edges_in_isocynate, edges_in_nitro, edges_in_carbodiimide, edges_in_oxime, edges_in_c_nitroso, edges_in_carboximidamide, edges_in_phosphorus_grps

def find_index(lst, num, hy_idx):
    return [i for i, sub_lst in enumerate(lst) if ((num in sub_lst) and (i !=hy_idx ))]
        
def get_incidence_matrix(G, use_cycle = True):

    carboxyl, carbamate, carbamide, amide, ketone, aldehyde, ketene, isocynate, isothiocynate, thione, thioamide, thiourea, oxime, hydrazone, carboximidamide, imine, carbodiimide, alcohol, thiol, allene, alkyne, ether, peroxide, thioether, sulfoxide, sulfone, sulfonamide, sulfonate, disulfide, amine, nitrile, hydrazine, azo, c_nitroso, hydroamine, nitro, phosphorus_grps, alkene = func_groups(G)
    edges_in_alkene, edges_in_alkyne, edges_in_allene, edges_in_carboxyl, edges_in_ketene, edges_in_alcohol, edges_in_ketone, edges_in_aldehyde, edges_in_ether, edges_in_peroxide, edges_in_carbamate, edges_in_thioether, edges_in_disulfide, edges_in_sulfone, edges_in_thioamide, edges_in_thiourea, edges_in_thiol, edges_in_thione, edges_in_sulfoxide, edges_in_isothiocynate, edges_in_sulfonamide, edges_in_sulfonate, edges_in_amine, edges_in_hydroamine, edges_in_amide, edges_in_imine, edges_in_carbamide, edges_in_nitrile, edges_in_hydrazine, edges_in_hydrazone, edges_in_azo, edges_in_isocynate, edges_in_nitro, edges_in_carbodiimide, edges_in_oxime, edges_in_c_nitroso, edges_in_carboximidamide, edges_in_phosphorus_grps = edges_in_func_groups(G)
    cycle_list, edges_in_cycle, nodes_in_cycle = cyclic_atoms(G)

    node_list = list(G.nodes())
    edge_list = list(G.edges())
    all_groups = (alkene +
                    alkyne +
                    allene +
                    carboxyl +
                    ketene +
                    alcohol +
                    ketone +
                    aldehyde +
                    ether +
                    peroxide +
                    carbamate +
                    thioether +
                    disulfide +
                    sulfone +
                    thioamide +
                    thiourea +
                    thiol +
                    thione +
                    sulfoxide +
                    isothiocynate +
                    sulfonamide +
                    sulfonate +
                    amine +
                    hydroamine +
                    amide +
                    imine +
                    carbamide +
                    nitrile +
                    hydrazine +
                    hydrazone +
                    azo +
                    isocynate +
                    nitro +
                    carbodiimide +
                    oxime +
                    c_nitroso +
                    carboximidamide +
                    phosphorus_grps)

    edges_in_groups = (edges_in_alkene +
                            edges_in_alkyne +
                            edges_in_allene +
                            edges_in_carboxyl +
                            edges_in_ketene +
                            edges_in_alcohol +
                            edges_in_ketone +
                            edges_in_aldehyde +
                            edges_in_ether +
                            edges_in_peroxide +
                            edges_in_carbamate +
                            edges_in_thioether +
                            edges_in_disulfide +
                            edges_in_sulfone +
                            edges_in_thioamide +
                            edges_in_thiourea +
                            edges_in_thiol +
                            edges_in_thione +
                            edges_in_sulfoxide +
                            edges_in_isothiocynate +
                            edges_in_sulfonamide +
                            edges_in_sulfonate +
                            edges_in_amine +
                            edges_in_hydroamine +
                            edges_in_amide +
                            edges_in_imine +
                            edges_in_carbamide +
                            edges_in_nitrile +
                            edges_in_hydrazine +
                            edges_in_hydrazone +
                            edges_in_azo +
                            edges_in_isocynate +
                            edges_in_nitro +
                            edges_in_carbodiimide +
                            edges_in_oxime +
                            edges_in_c_nitroso +
                            edges_in_carboximidamide +
                            edges_in_phosphorus_grps)

    if use_cycle:
        all_groups += cycle_list
        edges_in_groups += edges_in_cycle
    
    edges_in_groups.sort()
    edges_in_groups = list(edges_in_groups for edges_in_groups, _ in itertools.groupby(edges_in_groups))
    res = [list(ele) for ele in edge_list]    # g中所有边转换成list形式[u, v]
    edges_not_grouped = [x for x in res if x not in edges_in_groups]
    nodes_in_hyperedges = all_groups + edges_not_grouped

    num_atoms = len(G.nodes())
    "所有节点都得连上"
    num_hyedges = len(nodes_in_hyperedges)

    # 1. 返回hypergraph_edgeindex
    row = []
    col = []
    if num_hyedges != 0:
        for h_idx, fg_i in enumerate(nodes_in_hyperedges):
            row += fg_i
            col += ([h_idx for i in range(len(fg_i))])
        A = torch.tensor([row, col])
            
    else:
        A = torch.tensor([[i for i in range(num_atoms)], [1 for i in range(num_atoms)]])

    # 2. 返回fg_edgeindex
    edge_pair = []

    if num_hyedges != 0:
        for h_idx, fg_i in enumerate(nodes_in_hyperedges):  #   [7,6,8]
            for _, atomi_in_fgi in enumerate(fg_i): #   7
                for _, neighbor_of_atomi in enumerate(list(G[atomi_in_fgi].keys())):    #   7的邻居
                    # 邻居所在的官能团（超边）
                    ends = find_index(nodes_in_hyperedges, neighbor_of_atomi, h_idx)
                    for x in itertools.product([h_idx], ends):
                        edge_pair.append(tuple(x))
                    
                    
        edge_pair = list(set(sublst for sublst in edge_pair))
        edge_pair.sort()
        fg_edgeindex = torch.tensor([list(x) for x in zip(*edge_pair)]).to(torch.long)




    return A, fg_edgeindex, nodes_in_hyperedges

    '''
        num_grps = len(all_groups)
    
        edges_in_groups.sort()
        edges_in_groups = list(edges_in_groups for edges_in_groups, _ in itertools.groupby(edges_in_groups))
        res = [list(ele) for ele in edge_list]    # g中所有边转换成list形式[u, v]
        edges_not_grouped = [x for x in res if x not in edges_in_groups]

        num_edges_not_grouped = len(edges_not_grouped)
        
        "所有节点都得连上"
        num_hyedges = num_grps + num_edges_not_grouped
        
        
        H: VxE
        num_nodes = len(node_list)

        # hyperedge list
        "超边数量：官能团数量 + 非官能团的边的数量"
        hyedge_list = []
        for j in range(num_hyedges):
            hyedge = 'e{}'.format(j)
            hyedge_list.append(hyedge)

        # all 0s matrix num_nodes x num_hyedges
        "make all 0s, then later =1 if the node is in the hyperedge"
        row_index = [i for i in range(len(node_list))]

        A = np.zeros((num_nodes, num_hyedges))
        A = pd.DataFrame(A, columns=hyedge_list, index=row_index)

        # node index
        node_index = dict((node, i) for i, node in enumerate(node_list))
        # indices 
        for ei, e in enumerate(edge_list):  # edge_list:图中所有边
            (u, v) = e[:2]

            if u == v: continue
            ui = node_index[u]
            vi = node_index[v]

            for k in range(num_grps):
                if set([u, v]).issubset(set(all_groups[k])) == True:
                    A.iloc[ui, k] = 1
                    A.iloc[vi, k] = 1

        p = num_grps
        q = len(hyedge_list)

        # 不在官能团里的边
        for i, (ri, r) in zip(range(p, q), enumerate(edges_not_grouped)):
            (u, v) = r[:2]

            if u == v: continue
            ui = node_index[u]
            vi = node_index[v]

            A.iloc[ui, i] = 1
            A.iloc[vi, i] = 1
        else:
            A = pd.DataFrame(np.ones((num_nodes, 1)), index=row_index)

    '''



# A = get_incidence_matrix(nx_g)

# print(A)

