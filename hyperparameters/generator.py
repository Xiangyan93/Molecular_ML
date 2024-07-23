import json


tensorproduct = {
    'a_type': 'Tensorproduct',
    'b_type': 'Tensorproduct',
    'p_type': 'Additive_p',
    'q': [0.01, [0.0001, 1.0]],
    'atom_atomic_number': [['kDelta', 0.75, [0.1, 1.0]]],
    'atom_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_hybridization': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_chiral': [['kDelta', 0.5, [0.1, 1.0]]],
    'atom_morgan_hash': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_elemental_mode1': [['sExp', 2.0, [0.1, 10.0]]],
    'atom_elemental_mode2': [['sExp', 3.0, [0.1, 10.0]]],
    'atom_charge': [['sExp', 2.5, [0.1, 10.0]]],
    'atom_hcount': [['sExp', 2.5, [0.1, 10.0]]],
    'atom_ring_membership': [['kConv', 0.9, [0.1, 1.0]]],
    'bond_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
    'bond_conjugated': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
    'bond_order': [['sExp', 1.5, [0.1, 10.0]]],
    'probability_atomic_number': [['constant', 1.0, "fixed"]]
}


tensorproduct_inhomogeneous = {
    'a_type': 'Tensorproduct',
    'b_type': 'Tensorproduct',
    'p_type': 'Additive_p',
    'q': [0.01, [0.0001, 1.0]],
    'atom_atomic_number': [['kDelta', 0.75, [0.1, 1.0]]],
    'atom_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_hybridization': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_chiral': [['kDelta', 0.5, [0.1, 1.0]]],
    'atom_morgan_hash': [['kDelta', 0.9, [0.1, 1.0]]],
    'atom_elemental_mode1': [['sExp', 2.0, [0.1, 10.0]]],
    'atom_elemental_mode2': [['sExp', 3.0, [0.1, 10.0]]],
    'atom_charge': [['sExp', 2.5, [0.1, 10.0]]],
    'atom_hcount': [['sExp', 2.5, [0.1, 10.0]]],
    'atom_ring_membership': [['kConv', 0.9, [0.1, 1.0]]],
    'bond_aromatic': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
    'bond_conjugated': [['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kDelta', 0.5, [0.1, 1.0]]],
    'bond_order': [['sExp', 1.5, [0.1, 10.0]]],
    'probability_an5': [['uniform', 1.0, [0.01, 100]]],
    'probability_an6': [['uniform', 1.0, [0.01, 100]]],
    'probability_an7': [['uniform', 1.0, [0.01, 100]]],
    'probability_an8': [['uniform', 1.0, [0.01, 100]]],
    'probability_an9': [['uniform', 1.0, [0.01, 100]]],
    'probability_an14': [['uniform', 1.0, [0.01, 100]]],
    'probability_an15': [['uniform', 1.0, [0.01, 100]]],
    'probability_an16': [['uniform', 1.0, [0.01, 100]]],
    'probability_an17': [['uniform', 1.0, [0.01, 100]]],
    'probability_an35': [['uniform', 1.0, [0.01, 100]]],
    'probability_an53': [['uniform', 1.0, [0.01, 100]]],
    'probability_an_chiral': [['uniform', 0.2, [0.01, 100]]]
}


additive = {
    'a_type': 'Additive',
    'b_type': 'Additive',
    'p_type': 'Additive_p',
    'q': [0.01, [0.0001, 1.0]],
    'atom_atomic_number': [['kC', 0.5, [0.0001, 1.0]],
                           ['kDelta', 0.75, [0.1, 1.0]]],
    'atom_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kC', 0.5, [0.0001, 1.0]],
                         ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_hybridization': [['kC', 0.5, [0.0001, 1.0]],
                           ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_chiral': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'atom_morgan_hash': [['kC', 0.5, [0.0001, 1.0]],
                         ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_elemental_mode1': [['kC', 0.5, [0.0001, 1.0]],
                             ['sExp', 2.0, [0.1, 10.0]]],
    'atom_elemental_mode2': [['kC', 0.5, [0.0001, 1.0]],
                             ['sExp', 3.0, [0.1, 10.0]]],
    'atom_charge': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, [0.1, 10.0]]],
    'atom_hcount': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, [0.1, 10.0]]],
    'atom_ring_membership': [['kC', 0.5, [0.0001, 1.0]],
                             ['kConv', 0.9, [0.1, 1.0]]],
    'bond_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'bond_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'bond_conjugated': [['kC', 0.5, [0.0001, 1.0]],
                        ['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kC', 0.5, [0.0001, 1.0]],
                         ['kDelta', 0.5, [0.1, 1.0]]],
    'bond_order': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 1.5, [0.1, 10.0]]],
    'probability_atomic_number': [['constant', 1.0, "fixed"]]
}


additive_inhomogeneous = {
    'a_type': 'Additive',
    'b_type': 'Additive',
    'p_type': 'Additive_p',
    'q': [0.01, [0.0001, 1.0]],
    'atom_atomic_number': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.75, [0.1, 1.0]]],
    'atom_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_ring_number': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_hybridization': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_chiral': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'atom_morgan_hash': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'atom_elemental_mode1': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.0, [0.1, 10.0]]],
    'atom_elemental_mode2': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 3.0, [0.1, 10.0]]],
    'atom_charge': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, [0.1, 10.0]]],
    'atom_hcount': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 2.5, [0.1, 10.0]]],
    'atom_ring_membership': [['kC', 0.5, [0.0001, 1.0]], ['kConv', 0.9, [0.1, 1.0]]],
    'bond_aromatic': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'bond_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'bond_conjugated': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.9, [0.1, 1.0]]],
    'bond_ring_stereo': [['kC', 0.5, [0.0001, 1.0]], ['kDelta', 0.5, [0.1, 1.0]]],
    'bond_order': [['kC', 0.5, [0.0001, 1.0]], ['sExp', 1.5, [0.1, 10.0]]],
    'probability_an5': [['uniform', 1.0, [0.01, 100]]],
    'probability_an6': [['uniform', 1.0, [0.01, 100]]],
    'probability_an7': [['uniform', 1.0, [0.01, 100]]],
    'probability_an8': [['uniform', 1.0, [0.01, 100]]],
    'probability_an9': [['uniform', 1.0, [0.01, 100]]],
    'probability_an14': [['uniform', 1.0, [0.01, 100]]],
    'probability_an15': [['uniform', 1.0, [0.01, 100]]],
    'probability_an16': [['uniform', 1.0, [0.01, 100]]],
    'probability_an17': [['uniform', 1.0, [0.01, 100]]],
    'probability_an35': [['uniform', 1.0, [0.01, 100]]],
    'probability_an53': [['uniform', 1.0, [0.01, 100]]]
}


open('tensorproduct.json', 'w').write(json.dumps(tensorproduct))
open('tensorproduct-inhomogeneous.json', 'w').write(json.dumps(tensorproduct_inhomogeneous))
open('addtive.json', 'w').write(json.dumps(additive))
open('addtive-inhomogeneous.json', 'w').write(json.dumps(additive_inhomogeneous))