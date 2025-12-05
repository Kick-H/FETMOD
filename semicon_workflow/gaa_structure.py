"""GAAFET/FinFET structure utilities promoted from the examples folder.

The helper functions here mirror the exploratory `examples/gaa_structure_generator.py`
script while exposing a cleaner programmatic entry point
(:func:`generate_gaafet_device`) and a small CLI (:func:`cli_generate_gaafet`).
"""
from __future__ import annotations

import argparse
import bisect
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def delete_door_region(atoms: Atoms, del_region: np.ndarray, oxyz: Sequence[float]) -> Atoms:
    pos = atoms.get_positions()
    mask = []
    Ox, Oy, Oz = oxyz
    for p in pos:
        in_hole = all(del_region[0, i] <= p[i] <= del_region[1, i] for i in range(3))
        out_box = any([p[0] < 0 or p[0] > Ox, p[1] < 0 or p[1] > Oy, p[2] < 0 or p[2] > Oz])
        mask.append(not (in_hole or out_box))

    return atoms[mask]


def get_block_shape(atoms: Atoms, oxyz: Sequence[float]) -> Atoms:
    pos = atoms.get_positions()
    mask = []
    Ox, Oy, Oz = oxyz
    for p in pos:
        out_box = any([p[0] < 0 or p[0] > Ox, p[1] < 0 or p[1] > Oy, p[2] < 0 or p[2] > Oz])
        mask.append(not out_box)
    return atoms[mask]


def delete_square_region(atoms: Atoms, del_region: np.ndarray, shiftz: Sequence[float] = (0, 0, 0)) -> Atoms:
    pos = atoms.get_positions()
    mask = []
    for p in pos:
        in_hole = all(del_region[0, i] <= p[i] - shiftz[i] <= del_region[1, i] for i in range(3))
        mask.append(not in_hole)

    return atoms[mask]


def delete_zero_square_region(atoms: Atoms, oxyz: Sequence[float]) -> Atoms:
    pos = atoms.get_positions()
    mask = []
    Ox, Oy, Oz = oxyz
    for p in pos:
        out_box = any([p[0] < 0 or p[0] > Ox, p[1] < 0 or p[1] > Oy, p[2] < 0 or p[2] > Oz])
        mask.append(not out_box)
    return atoms[mask]


def remove_internal_blocks(atoms: Atoms, oxyz: Sequence[float], ixyz: Iterable[Sequence[float]], nxyz: Sequence[int], amorphous: bool = False) -> Atoms:
    """Repeat atoms, trim to the outer block, and delete inner cuboids."""

    oxyz_array = np.array(oxyz)
    atoms_rep = atoms.repeat(nxyz)
    out_atoms = get_block_shape(atoms_rep, oxyz)

    for block_xyz in ixyz:
        ixyz_array = np.array(block_xyz)[:3]
        delxyz = np.array([(oxyz_array - ixyz_array) / 2, (oxyz_array + ixyz_array) / 2])
        shift_xyz = [0, 0, block_xyz[3]]
        out_atoms = delete_square_region(out_atoms, delxyz, shift_xyz)
    if amorphous:
        out_atoms.set_cell(oxyz)
    return out_atoms


def delete_tin_top_regions(tin_atoms: Atoms, ref_cell: Sequence[float], del_group: Optional[Sequence[int]] = None) -> Atoms:
    tin_positions = tin_atoms.get_positions()
    tin_cell = tin_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    group_old = tin_atoms.get_array("group")

    flag = 0
    for i in range(2):
        if ref_cell[i] == tin_cell[i]:
            flag = i
            continue
        top_z_min = tin_cell[2] - ref_cell[2]
        top_z_max = tin_cell[2]

        wide = ref_cell[i]
        tin_y_center = tin_cell[i] / 2
        y_min = tin_y_center - wide / 2
        y_max = tin_y_center + wide / 2

    if flag == 0:
        left_del_region = [[0, 0, top_z_min], [tin_cell[0], y_min, top_z_max]]
        right_del_region = [[0, y_max, top_z_min], [tin_cell[0], tin_cell[1], top_z_max]]
    else:
        left_del_region = [[0, 0, top_z_min], [y_min, tin_cell[1], top_z_max]]
        right_del_region = [[y_max, 0, top_z_min], [tin_cell[0], tin_cell[1], top_z_max]]

    mask: List[bool] = []
    for i, pos in enumerate(tin_positions):
        x, y, z = pos
        in_left = left_del_region[0][0] <= x <= left_del_region[1][0] and left_del_region[0][1] <= y <= left_del_region[1][1] and left_del_region[0][2] <= z <= left_del_region[1][2]
        in_right = right_del_region[0][0] <= x <= right_del_region[1][0] and right_del_region[0][1] <= y <= right_del_region[1][1] and right_del_region[0][2] <= z <= right_del_region[1][2]
        del_atom_id = False
        if del_group is not None:
            del_atom_id = group_old[i] in del_group
        mask.append(not (del_atom_id or in_left or in_right))

    return tin_atoms[mask]


def create_square(infile: str, lxyz: Sequence[float], ctype: str = "", amorphous: bool = False) -> Optional[Atoms]:
    try:
        atoms = read(infile)
    except FileNotFoundError:
        print(f"Error: file {infile} cannot be found.")
        return None
    atoms.center()
    cells = atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    nxyz = np.round(lxyz / cells).astype(int)
    if amorphous:
        nxyz = np.floor(lxyz / cells).astype(int) + [1, 1, 1]
    atoms_rep = atoms.repeat(nxyz)
    if amorphous:
        atoms_rep = delete_zero_square_region(atoms.repeat(nxyz), lxyz)
        atoms_rep.set_cell(lxyz)
    else:
        atoms_rep = atoms.repeat(nxyz)
    return atoms_rep


def set_group_id_by_set(atoms: Atoms, direct: str = "Z", nlist: Optional[Iterable[float]] = None, group_shift: int = 0) -> Optional[int]:
    if atoms is None:
        return None

    dir_num = {"X": 0, "Y": 1, "Z": 2, "x": 0, "y": 1, "z": 2}
    if direct not in dir_num:
        raise ValueError(f"direct must be one of X/Y/Z (case-insensitive), got {direct}")

    nlist = list(nlist) if nlist is not None else [0]

    nat = atoms.get_global_number_of_atoms()
    group = np.zeros(nat, dtype=int)
    posz = atoms.get_positions()[:, dir_num[direct]]

    nlist = sorted(nlist)
    for i in range(nat):
        group[i] = group_shift + bisect.bisect_left(nlist, posz[i])

    atoms.set_array("group", group, int, ())

    return int(group.max()) + 1


def set_heat_source_group(
    atoms: Atoms,
    direct: str = "X",
    nlist: Optional[Iterable[float]] = None,
    sel_group: int = 0,
    ellipse_center: Optional[Sequence[float]] = None,
    ellipse_radii: Optional[Sequence[float]] = None,
    ellipse_plane: str = "XY",
) -> Optional[int]:
    if atoms is None:
        return None

    dir_num = {"X": 0, "Y": 1, "Z": 2, "x": 0, "y": 1, "z": 2}
    if direct not in dir_num:
        raise ValueError(f"direct must be one of X/Y/Z (case-insensitive), got {direct}")

    nlist = list(nlist) if nlist is not None else [0]

    plane_map = {"XY": (0, 1), "xy": (0, 1), "YZ": (1, 2), "yz": (1, 2), "XZ": (0, 2), "xz": (0, 2)}
    use_ellipse = (ellipse_center is not None) and (ellipse_radii is not None)

    if use_ellipse:
        if ellipse_plane not in plane_map:
            raise ValueError("ellipse_plane must be one of 'XY', 'YZ', 'XZ' (case-insensitive)")
        ax1, ax2 = plane_map[ellipse_plane]
        ellipse_center = np.asarray(ellipse_center, dtype=float)
        ellipse_radii = np.asarray(ellipse_radii, dtype=float)
        if ellipse_center.shape != (2,) or ellipse_radii.shape != (2,):
            raise ValueError("ellipse_center and ellipse_radii must be length-2 sequences")
        if np.any(ellipse_radii <= 0):
            raise ValueError("ellipse_radii must be positive")

    nat = atoms.get_global_number_of_atoms()
    group = atoms.get_array("group")
    positions = atoms.get_positions()

    pos_dir = positions[:, dir_num[direct]]

    nlist = sorted(nlist)

    for i in range(nat):
        if group[i] == sel_group:
            group[i] = bisect.bisect_left(nlist, pos_dir[i]) + sel_group

    if use_ellipse:
        new_group_id = group.max() + 1
        ax1, ax2 = plane_map[ellipse_plane]
        u = positions[:, ax1]
        v = positions[:, ax2]
        du = (u - ellipse_center[0]) / ellipse_radii[0]
        dv = (v - ellipse_center[1]) / ellipse_radii[1]
        inside_ellipse = du * du + dv * dv <= 1.0
        for i in range(nat):
            if inside_ellipse[i]:
                group[i] = new_group_id

    atoms.set_array("group", group, int, ())
    return int(group.max()) + 1


def set_group_id(atoms: Optional[Atoms], nid: int) -> None:
    if atoms is not None:
        nat = atoms.get_global_number_of_atoms()
        group = np.array([nid] * nat)
        atoms.set_array("group", group, int, ())


def add_atomic_groups_by_id_order(atoms: Atoms, colname: str = "group", backup_old_scalar: bool = False, modify_group: Sequence[Sequence[int]] = (([0], [0]))) -> None:
    """Append a monotonically increasing per-atom group column."""
    if atoms is None:
        return

    nat = len(atoms)
    new_col = np.arange(nat, dtype=int).reshape(nat, 1)

    if colname in atoms.arrays:
        old = atoms.get_array(colname)
        for i, o in enumerate(old):
            if o in modify_group[0]:
                ind = modify_group[0].index(o)
                old[i] = modify_group[1][ind]
        if old.ndim == 1:
            if backup_old_scalar:
                atoms.set_array(colname + "_scalar_backup", old.copy(), int, ())
            old2d = old.reshape(nat, 1)
            del atoms.arrays[colname]
        else:
            if old.shape[0] != nat:
                raise ValueError(f"Existing '{colname}' shape mismatch: {old.shape} vs nat={nat}")
            old2d = old
            del atoms.arrays[colname]
    else:
        old2d = np.empty((nat, 0), dtype=int)

    group_mat = np.hstack([old2d.astype(int, copy=False), new_col])
    atoms.set_array(colname, group_mat, int, (group_mat.shape[1],))


def add_group_methods(atoms: Atoms, colname: str = "group", keep_scalar_backup: bool = False) -> None:
    if atoms is None:
        return
    nat = len(atoms)

    if colname in atoms.arrays:
        old = atoms.get_array(colname)
        if old.ndim == 1:
            old2d = old.reshape(nat, 1)
        else:
            if old.shape[0] != nat:
                raise ValueError(f"Existing '{colname}' shape mismatch: {old.shape} vs nat={nat}")
            old2d = old
        start_gid = int(np.max(old2d)) + 1 if old2d.size else 0
    else:
        old2d = np.empty((nat, 0), dtype=int)
        start_gid = 0

    atom_group = np.arange(start_gid, start_gid + nat, dtype=int).reshape(nat, 1)
    new_group = np.hstack([old2d.astype(int, copy=False), atom_group])

    if keep_scalar_backup and colname in atoms.arrays and atoms.get_array(colname).ndim == 1:
        atoms.set_array(colname + "_scalar_backup", atoms.get_array(colname).copy(), int, ())

    if colname in atoms.arrays:
        del atoms.arrays[colname]

    atoms.set_array(colname, new_group, int, (new_group.shape[1],))


def create_frame(infile: str, oxyz: Sequence[float], ixyz: Iterable[Sequence[float]], ctype: str = "", amorphous: bool = False, shift_z: float = 0) -> Optional[Atoms]:
    oxyz_array = np.array(oxyz)
    atoms = read(infile, format="extxyz")
    atoms.center()
    cells = atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    nxyz = np.round(oxyz / cells).astype(int) + 1

    atoms_rep = atoms.repeat(nxyz)
    out_atoms = get_block_shape(atoms_rep, oxyz)

    for block_xyz in ixyz:
        ixyz_array = np.array(block_xyz)[:3]
        delxyz = np.array([(oxyz_array - ixyz_array) / 2, (oxyz_array + ixyz_array) / 2])
        delxyz[0, 2] = ixyz_array[2]
        delxyz[1, 2] = block_xyz[3]
        out_atoms = delete_square_region(out_atoms, delxyz)
    if amorphous:
        out_atoms.set_cell(oxyz)
    if shift_z != 0:
        add_offset(out_atoms, [0, 0, 0], [0, 0, 0], shift_z)

    return out_atoms


def create_door(infile: str, oxyz: np.ndarray, ixyz: np.ndarray, ctype: str = "", amorphous: bool = False, shift_z: float = 0) -> Optional[Atoms]:
    oxyz_array = np.array(oxyz)
    ixyz_array = np.array(ixyz)
    if oxyz_array.shape != (3,) or ixyz_array.shape != (3,):
        raise ValueError("oxyz and ixyz must be 3-element vectors")
    if np.any(ixyz_array > oxyz_array):
        raise ValueError("Inner dimensions cannot exceed outer dimensions")

    try:
        atoms = read(infile, format="extxyz" if infile.endswith("xyz") else None)
    except FileNotFoundError:
        print(f"Error: file {infile} not found.")
        return None

    atoms.center()
    cells = atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    nxyz = np.round(oxyz_array / cells).astype(int) + 1

    atoms_rep = atoms.repeat(nxyz)
    if shift_z != 0:
        add_offset(atoms_rep, [0, 0, 0], [0, 0, 0], shift_z)
    delxyz = np.array([(oxyz_array - ixyz_array) / 2, (oxyz_array + ixyz_array) / 2])
    delxyz[1, 2] -= delxyz[0, 2]
    delxyz[0, 2] = 0
    door_atoms = delete_door_region(atoms_rep, delxyz, oxyz)
    if amorphous:
        door_atoms.set_cell(oxyz)
    return door_atoms


def add_offset(atoms: Optional[Atoms], substract_cell: Sequence[float], current_cell: Sequence[float], z_offset: float) -> Optional[Atoms]:
    if atoms is not None:
        atoms.positions += [
            (substract_cell[0] - current_cell[0]) / 2,
            (substract_cell[1] - current_cell[1]) / 2,
            z_offset,
        ]
    return atoms


def combine_FinFET(sub_del_atoms: Atoms, Gs1_atoms: Atoms, G_atoms: Atoms, del1_field_atoms: Atoms) -> Atoms:
    pos = sub_del_atoms.get_positions()
    mask = []
    for p in pos:
        out_box = any([p[0] < 10 and p[1] < 10])
        mask.append(out_box)
    sel_base_atoms = sub_del_atoms[mask]

    substract_cell = sub_del_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    base_high = np.max(sel_base_atoms.get_positions()[:, 2]) + 1

    del1_field_cell = del1_field_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    add_offset(del1_field_atoms, substract_cell, del1_field_cell, base_high)

    Gs1_cell = Gs1_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    add_offset(Gs1_atoms, substract_cell, Gs1_cell, base_high)

    G_cell = G_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    add_offset(G_atoms, substract_cell, G_cell, base_high)

    combined = sub_del_atoms + Gs1_atoms + G_atoms + del1_field_atoms

    combined.set_pbc([1, 1, 0])
    total_height = base_high + G_cell[2]
    combined.set_cell([substract_cell[0], substract_cell[1], total_height * 1.1])

    return combined


def combine_gate(substract_atoms: Atoms, substract1_atoms: Atoms, substract2_atoms: Atoms, field_atoms: Atoms, s_atoms: Atoms, d_atoms: Atoms, fin_atoms: Atoms, Gs1_atoms: Atoms, G_atoms: Atoms) -> Atoms:
    substract_cell = substract_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    substract1_cell = substract1_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    substract2_cell = substract2_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    field_cell = field_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    G_cell = G_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]

    substract1_atoms = add_offset(substract1_atoms, substract_cell, substract1_cell, substract_cell[2])
    substract2_atoms = add_offset(substract2_atoms, substract_cell, substract2_cell, substract_cell[2] + substract1_cell[2])

    base_offset = substract_cell[2] + substract1_cell[2] + substract2_cell[2]

    field_atoms = add_offset(field_atoms, substract_cell, field_cell, base_offset)

    common_offset = base_offset + field_cell[2]

    s_atoms = add_offset(s_atoms, substract_cell, s_atoms.get_cell()[[0, 1, 2], [0, 1, 2]], common_offset)
    d_atoms = add_offset(d_atoms, substract_cell, d_atoms.get_cell()[[0, 1, 2], [0, 1, 2]], common_offset)
    fin_atoms = add_offset(fin_atoms, substract_cell, fin_atoms.get_cell()[[0, 1, 2], [0, 1, 2]], common_offset)
    Gs1_atoms = add_offset(Gs1_atoms, substract_cell, Gs1_atoms.get_cell()[[0, 1, 2], [0, 1, 2]], common_offset)
    G_atoms = add_offset(G_atoms, substract_cell, G_cell, common_offset)

    combined = substract_atoms + substract1_atoms + substract2_atoms + field_atoms + s_atoms + d_atoms + fin_atoms + Gs1_atoms + G_atoms

    combined.set_pbc([1, 1, 0])
    total_height = common_offset + G_cell[2]
    combined.set_cell([substract_cell[0], substract_cell[1], total_height * 1.1])

    return combined


def separate_fin_sd(sd_atoms: Optional[Atoms], substract_cell: Sequence[float], fin_width_x: float) -> Tuple[Optional[Atoms], Optional[Atoms], Optional[Atoms]]:
    if sd_atoms is None:
        return None, None, None

    fin_width_y = substract_cell[1]

    base_center_x = substract_cell[0] / 2
    base_center_y = substract_cell[1] / 2
    sd_cell = sd_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    x_offset = (substract_cell[0] - sd_cell[0]) / 2
    y_offset = (substract_cell[1] - sd_cell[1]) / 2
    sd_positions = sd_atoms.get_positions() + [x_offset, y_offset, 0]

    fin_start_x = base_center_x - fin_width_x / 2
    fin_end_x = base_center_x + fin_width_x / 2
    fin_start_y = base_center_y - fin_width_y / 2
    fin_end_y = base_center_y + fin_width_y / 2

    fin_mask = (
        (sd_positions[:, 0] >= fin_start_x)
        & (sd_positions[:, 0] <= fin_end_x)
        & (sd_positions[:, 1] >= fin_start_y)
        & (sd_positions[:, 1] <= fin_end_y)
    )
    s_mask = sd_positions[:, 0] < fin_start_x
    d_mask = sd_positions[:, 0] > fin_end_x

    fin_atoms = sd_atoms[fin_mask]
    s_atoms = sd_atoms[s_mask]
    d_atoms = sd_atoms[d_mask]

    print("Successfully separated SD regions:")
    print(f"  Fin region: x[{fin_start_x:.1f}-{fin_end_x:.1f}], y[{fin_start_y:.1f}-{fin_end_y:.1f}]")
    print(
        f"  Source atoms: {len(s_atoms)}, Drain atoms: {len(d_atoms)}, Fin atoms: {len(fin_atoms)}"
    )

    return s_atoms, d_atoms, fin_atoms


# ---------------------------------------------------------------------------
# Doping helpers
# ---------------------------------------------------------------------------
def analyze_xyz_file(filename: str) -> Tuple[Optional[Atoms], List[int]]:
    try:
        atoms = read(filename)
        print(f"Successfully read file: {filename}")
        print(f"Total atoms: {len(atoms)}")

        si_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "Si"]
        print(f"Number of Si atoms: {len(si_indices)}")

        return atoms, si_indices

    except Exception as e:
        print(f"Failed to read file: {e}")
        return None, []


def dope_region(atoms: Atoms, si_indices: Sequence[int], region: Sequence[float], target_element: str, concentration: float) -> int:
    x_min, x_max, y_min, y_max, z_min, z_max = region
    region_si_indices = []

    for idx in si_indices:
        x, y, z = atoms.positions[idx]
        if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
            region_si_indices.append(idx)

    print(f"[Region Analysis] Number of Si atoms in target region: {len(region_si_indices)}")

    num_to_dope = max(1, int(len(region_si_indices) * concentration)) if concentration > 0 else 0
    num_to_dope = min(num_to_dope, len(region_si_indices))

    if num_to_dope > 0:
        doped_indices = random.sample(region_si_indices, num_to_dope)
        for idx in doped_indices:
            atoms[idx].symbol = target_element
        print(
            f"[Doping Completed] {target_element} doping: {num_to_dope} atoms (concentration: {concentration*100:.2f}%)"
        )
    else:
        print("[Skipping Doping] Concentration is 0 or no Si atoms in region")

    return num_to_dope


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate_gaafet_device(
    gaa_num: int,
    H6: float,
    H7: float,
    thermobase: float,
    substract_path: str = "Si.xyz",
    gate_dielectric_path: str = "amorphous_HfO2.xyz",
    oxide_path: str = "amorphous_SiO2.xyz",
    gate_path: str = "TiN.xyz",
    output_path: Optional[str | Path] = None,
) -> Atoms:
    """Construct and return a combined GAAFET device."""

    fin_same_with_gaa = True
    fix_high = 5
    th_si = 10
    th_tin = 20
    thin_HfO = 10

    H1, H2, H4, H8 = 80, 10, 20, 20
    if fin_same_with_gaa and gaa_num == 1:
        H6 += (H6 + H7) * 2

    H3 = (H6 + H7) * gaa_num + H8
    H5 = H2 + (H6 + H7) * gaa_num

    L, L1 = 348.0896, 140
    W, W1 = 212.1834, 80

    print(f"H1:{H1}, H2:{H2}, H3:{H3}, H4:{H4}, H5:{H5}, H6:{H6}, H7:{H7}")
    print(f"W:{W}, W1:{W1}, L:{L}, L1:{L1}")

    group_shift = 0

    si_xyz = np.array([L, W, fix_high + th_si + H1 + H5])
    substract_atoms = create_square(substract_path, si_xyz, ctype="sub_fix")
    if substract_atoms is None:
        raise FileNotFoundError(f"Required substrate file missing: {substract_path}")
    substract_cell = substract_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    L, W = substract_cell[0], substract_cell[1]
    delt_H = substract_cell[2] - (fix_high+th_si+H1+H5)
    
    gaa_gata_list = [fix_high, fix_high + th_si, fix_high + th_si + H1 + delt_H, fix_high + th_si + H1 + H2 + delt_H]
    del_group = [3]
    for _ in range(gaa_num):
        gaa_gata_list.append(gaa_gata_list[-1] + H6)
        gaa_gata_list.append(gaa_gata_list[-1] + H7)
        del_group.append(len(gaa_gata_list) - 2)
    del_group.append(len(gaa_gata_list))
    if gaa_num == 1:
        del_group = None

    ngroup_si = set_group_id_by_set(substract_atoms, direct="Z", nlist=gaa_gata_list)
    group_shift += int(ngroup_si or 0)

    sub_del_atoms = delete_tin_top_regions(substract_atoms, np.array([substract_cell[0], W1, H5]), del_group=del_group)

    ngroup_si = set_heat_source_group(
        sub_del_atoms,
        direct="X",
        nlist=[(L - L1) / 2],
        sel_group=group_shift - 1,
        ellipse_center=((L - L1) / 2, substract_cell[2]),
        ellipse_radii=(thermobase * 1.5, thermobase),
        ellipse_plane="XZ",
    )
    group_shift += 2
    fx, fy, fz = sub_del_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    print(f"Substrate cell after trimming: ({fx:.2f}, {fy:.2f}, {fz:.2f})")

    Gs1_out_xyz = np.array([L1, W1 + thin_HfO * 2, H5 + thin_HfO])
    Gs1_in_xyz: List[Sequence[float]] = [
        [L1, W1 + thin_HfO * 2, 0, H2],
        [L1, W1 + thin_HfO * 2, H2, H2 + H6 - thin_HfO],
    ]
    for i in range(gaa_num):
        Gs1_in_xyz.append([L1, W1, H2 + H6 * (i + 1) + H7 * i, H2 + (H6 + H7) * (i + 1)])
        Gs1_in_xyz.append([L1, W1 + thin_HfO * 2, H2 + (H6 + H7) * (i + 1) + thin_HfO, H2 + (H6 + H7) * (i + 1) + H6 - thin_HfO])
    if gaa_num == 1:
        Gs1_in_xyz = [[L1, W1, 0, H5]]
    Gs1_atoms = create_frame(gate_dielectric_path, Gs1_out_xyz, Gs1_in_xyz, ctype="Gs1", amorphous=True)
    if Gs1_atoms is None:
        raise FileNotFoundError(f"Gate dielectric file missing: {gate_dielectric_path}")
    set_group_id(Gs1_atoms, group_shift)
    group_shift += 1

    SiO_out_xyz = np.array([L, W, H5 + H8])
    SiO_in_xyz: List[Sequence[float]] = [[L1, W, H2, H2 + H3]]
    for i in range(gaa_num):
        SiO_in_xyz.append([L, W1, H2 + H6 * (i + 1) + H7 * i, H2 + (H6 + H7) * (i + 1)])
    if gaa_num == 1:
        SiO_in_xyz = [[L1, W, H2, H2 + H3], [L, W1, 0, H5]]
    SiO_atoms = create_frame(oxide_path, SiO_out_xyz, SiO_in_xyz, ctype="SiO2", amorphous=True)
    if SiO_atoms is None:
        raise FileNotFoundError(f"Oxide file missing: {oxide_path}")
    set_group_id(SiO_atoms, group_shift)

    G_out_xyz = np.array([L1, W, H5 + H8 + H4 + fix_high + th_tin])
    G_in_xyz: List[Sequence[float]] = [[L1, W, 0, H2]]
    for i in range(gaa_num):
        G_in_xyz.append([L1, W1 + thin_HfO * 2, H2 + H6 * (i + 1) + H7 * i - thin_HfO, H2 + (H6 + H7) * (i + 1) + thin_HfO])
    if gaa_num == 1:
        G_in_xyz = [[L1, W, 0, H2], [L1, W1 + thin_HfO * 2, 0, H5 + thin_HfO]]
    G_atoms = create_frame(gate_path, G_out_xyz, G_in_xyz, ctype="G", amorphous=True)
    if G_atoms is None:
        raise FileNotFoundError(f"Gate file missing: {gate_path}")
    G_cell = G_atoms.get_cell()[[0, 1, 2], [0, 1, 2]]
    ngroup_si = set_group_id_by_set(G_atoms, direct="Z", group_shift=group_shift, nlist=[fix_high, G_cell[2] - th_tin, G_cell[2] - fix_high])
    group_shift += int((ngroup_si or 0))

    combined = combine_FinFET(sub_del_atoms, Gs1_atoms, G_atoms, SiO_atoms)

    mgroup = [[0, 1, 2, 5], [0, 1, 2, 3]]
    for _ in range(gaa_num - 1):
        mgroup[0].append(mgroup[0][-1] + 2)
        mgroup[1].append(mgroup[1][-1] + 1)

    for _ in range(7):
        mgroup[0].append(mgroup[0][-1] + 1)
        mgroup[1].append(mgroup[1][-1] + 1)
    mgroup[1][-1] = 0
    mgroup[1][-2] = 1

    add_atomic_groups_by_id_order(combined, modify_group=mgroup)

    if output_path is None:
        output_path = Path(f"combined_GAAFET_{gaa_num}_{H6:.0f}_{H7:.0f}.xyz")
    write(output_path, combined, format="extxyz")
    print(f"Structure generated and saved as {output_path}")

    return combined


def cli_generate_gaafet() -> None:
    parser = argparse.ArgumentParser(description="Generate a GAAFET structure using ASE utilities.")
    parser.add_argument("gaa_num", type=int, help="Number of GAA stacks")
    parser.add_argument("H6", type=float, help="GAA H6 dimension")
    parser.add_argument("H7", type=float, help="GAA H7 dimension")
    parser.add_argument("thermobase", type=float, help="Thermal base radius for heat-source grouping")
    parser.add_argument("--substract", default="Si.xyz", help="Path to substrate XYZ file")
    parser.add_argument("--gate-dielectric", default="amorphous_HfO2.xyz", dest="gate_dielectric", help="Path to gate dielectric XYZ file")
    parser.add_argument("--oxide", default="amorphous_SiO2.xyz", help="Path to oxide XYZ file")
    parser.add_argument("--gate", default="TiN.xyz", help="Path to gate metal XYZ file")
    parser.add_argument("--output", default=None, help="Output extxyz path")

    args = parser.parse_args()

    generate_gaafet_device(
        gaa_num=args.gaa_num,
        H6=args.H6,
        H7=args.H7,
        thermobase=args.thermobase,
        substract_path=args.substract,
        gate_dielectric_path=args.gate_dielectric,
        oxide_path=args.oxide,
        gate_path=args.gate,
        output_path=args.output,
    )


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    cli_generate_gaafet()
