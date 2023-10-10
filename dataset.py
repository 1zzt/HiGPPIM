import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
import itertools
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from get_fgs import *

ALLOWABLE_BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'conjugated': ['T/F'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
}


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()] + [atom.GetMass() * 0.01] + [int(atom.GetChiralTag())]

    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
    
    

    results = np.array(results).astype(np.float32)
    return torch.from_numpy(results)


def get_bond_feature(mol, edge_index):
    bond_features = []    
    for i in range(edge_index.shape[1]):
        a1 = int(edge_index[0][i])
        a2 = int(edge_index[1][i])
        if a1 != a2:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:  
                bond_features.append(np.array(
                        one_of_k_encoding_unk(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
                        [bond.GetIsConjugated()] +
                        one_of_k_encoding_unk(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
                    ))

    return torch.from_numpy(np.array(bond_features)).float()

def get_fg_feature(mol, nodes_in_hyperedges):


    fg_fea_dict = [defaultdict(int) for _ in range(len(nodes_in_hyperedges))]

    for fg_idx, fg in enumerate(nodes_in_hyperedges):
        for i in range(len(fg)):
            atom = mol.GetAtomWithIdx(fg[i])
            elem = atom.GetSymbol()
            if elem in ['C', 'O', 'N', 'P', 'S']:
                key = 'Symbol_'+elem
            elif elem in ['F', 'Cl', 'Br', 'I']:
                key = 'Symbol_X'
            else:
                key = 'Symbol_UNK'
            fg_fea_dict[fg_idx][key] += 1
            for j in range(i+1, len(fg)):    
                a1, a2 = fg[i], fg[j]
                bond = mol.GetBondBetweenAtoms(a1, a2)
                if bond is not None:
                    key = 'Bond_'+str(bond.GetBondType())
                    fg_fea_dict[fg_idx][key] += 1
                    if bond.IsInRing():
                        fg_fea_dict[fg_idx]['IsRing'] = 1


    # print(fg_fea_dict)
    fg_features = []
    for fg_dict in fg_fea_dict:
        fg_features.append(np.array(
            one_of_k_encoding_unk(fg_dict['Symbol_C'], range(11)) +  # 0-10, 10+
            one_of_k_encoding_unk(fg_dict['Symbol_O'], range(6)) +  # 0-5, 5+
            one_of_k_encoding_unk(fg_dict['Symbol_N'], range(6)) +
            one_of_k_encoding_unk(fg_dict['Symbol_P'], range(6)) +
            one_of_k_encoding_unk(fg_dict['Symbol_S'], range(6)) +
            [fg_dict['Symbol_X'] > 0] +
            [fg_dict['Symbol_UNK'] > 0] +
            one_of_k_encoding_unk(fg_dict['Bond_SINGLE'], range(11)) +  # 0-10, 10+
            one_of_k_encoding_unk(fg_dict['Bond_DOUBLE'], range(8)) +  # 0-6, 6+
            one_of_k_encoding_unk(fg_dict['Bond_TRIPLE'], range(8)) +
            one_of_k_encoding_unk(fg_dict['Bond_AROMATIC'], range(8)) +
            [fg_dict['IsRing']]
            ))
    # print(fg_features)
    return torch.from_numpy(np.array(fg_features)).float()

def get_nx_g(mol, edge_index):
    atomic_number = []
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
        atomic_number.append(torch.tensor([atom.GetAtomicNum()]))

    single_bond = list(torch.tensor([1., 0., 0., 0.]))
    double_bond = list(torch.tensor([0., 1., 0., 0.]))
    triple_bond = list(torch.tensor([0., 0., 1., 0.]))
    aromatic_bond = list(torch.tensor([0., 0., 0., 1.]))
    bonds = {BT.SINGLE: single_bond, BT.DOUBLE: double_bond, BT.TRIPLE: triple_bond, BT.AROMATIC: aromatic_bond}
    bond_type = []   
    num_atoms = mol.GetNumAtoms()

    for i in range(edge_index.shape[1]):
        a1 = int(edge_index[0][i])
        a2 = int(edge_index[1][i])
        if a1 != a2:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is not None:  
                bond_type.append(bonds[bond.GetBondType()]) 
    # print(bond_type)
    # bond_type = []
    # for i in range(num_atoms):
    #     for j in range(num_atoms):
    #         if i != j:
    #             bond = mol.GetBondBetweenAtoms(i, j)
    #             if bond is not None:                 
    #                 bond_type.append(bonds[bond.GetBondType()])

    data = Data(edge_index=edge_index)
    data.atomic_number = atomic_number
    data.bond_type = bond_type
    data.num_atoms = num_atoms

    nx_g = to_networkx(data, node_attrs=['atomic_number'],edge_attrs=['bond_type'], to_undirected=True)
    nx_g = nx.Graph(nx_g)
    return nx_g

class MolData(Data):
    def __init__(self, x_pw = None, edge_index_pw = None, x_hyg=None, edge_index_hyg=None, x_fg = None, edge_index_fg = None, **kwargs):
        super(MolData, self).__init__(**kwargs)
        self.x_pw = x_pw
        self.edge_index_pw = edge_index_pw

        self.x_hyg = x_hyg
        self.edge_index_hyg = edge_index_hyg
        
        self.x_fg = x_fg
        self.edge_index_fg = edge_index_fg

    def __inc__(self, key, value, *args, **kwargs):
        # x_pw
        if key == 'edge_index_pw': 
            return int(self.x_pw.size(0))
        if key == 'edge_index_fg': 
            return int(self.x_fg.size(0))
        if key == 'edge_index_hyg':
            return torch.tensor([[self.x_hyg.size(0)], [int(self.edge_index_hyg[1].max()) + 1]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    # def __cat_dim__(self, key, value, *args, **kwargs):
    #     if key == 'hyperedge_attr':
    #         return None
    #     else:
    #         return super().__cat_dim__(key, value, *args, **kwargs)

class MyDataset(InMemoryDataset):
    def __init__(self, root = None, smiles=None, labels=None):

        super(MyDataset, self).__init__(root)

        self.smiles = smiles
        self.labels = labels
        self.data_list = []
        self.preprocess()

    def preprocess(self):
        

        for i in tqdm(range(len(self.smiles))):

            mol = Chem.MolFromSmiles(self.smiles[i])

            rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
            rows = torch.tensor(rows, dtype=torch.long)
            cols = torch.tensor(cols, dtype=torch.long)

            edge_index = torch.stack([rows, cols], dim = 0)
            label = torch.tensor(self.labels[i], dtype=torch.long)

            atom_features = [(atom.GetIdx(), get_atom_features(atom)) for atom in mol.GetAtoms()]
            atom_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
            _, atom_features = zip(*atom_features)
            atom_features = torch.stack(atom_features)
            bond_features = get_bond_feature(mol, edge_index)

            nx_g = get_nx_g(mol, edge_index)
            hyg_edge_index, fg_edgeindex, nodes_in_hyperedges = get_incidence_matrix(nx_g)
            
            fg_features = get_fg_feature(mol, nodes_in_hyperedges)
            
            # hyperedge_attr = torch.randn(num_hyedges, 512)


            aGraph = MolData(
                x_pw = atom_features,
                edge_index_pw = edge_index,

                x_hyg = atom_features,
                edge_index_hyg = hyg_edge_index,
                
                x_fg = fg_features,
                edge_index_fg = fg_edgeindex,

                y = label,
                smile = self.smiles[i],
                edge_attr = bond_features,


            )

            self.data_list.append(aGraph)
            
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx] 
