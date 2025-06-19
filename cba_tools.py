# A set of utility tools for the CompBioAsia Molecular
# Dynamics tutorials.
#
# These Python functions provide an easy interface to
# a range of third party tools that are useful for the
# preparartion of molecular systems for MD simulation.
# These include:
#
#   RDKit
#   OpenBabel
#   PDBFixer (a'spin-off' from OpenMM)
#   Chimera/ChimeraX
#   AmberTools
#   MDTraj
#
# The functions are:
#
#   smiles_to_pdb: Generates a PDB format file for a
#                  molecule from a SMILES string. Tries
#                  to be intelligent about protonation
#                  states of any ionizable groups. Useful
#                  for ligand preparation. Internally uses
#                  RDKit and OpenBabel.
#
#  fix:            Remediates macromolecule structure files
#                  obtained from the Protein Data Bank. Extracts
#                  chosen chains, fills in missing residues and
#                  atoms. (e.g. unresolved loops and side chains).
#                  Useful for protein preparation. Internally uses
#                  PDBFixer.
#
#  add_h:          Adds hydrogen atoms to heavy-atom only
#                  PDB format files. This is hard to get
#                  right every time with an automated tool,
#                  but the version here uses Chimera (or
#                  ChimeraX) which is often succesful. It
#                  also uses tools from AmberTools to 'clean
#                  up' the resulting structure, particularly
#                  setting the names of HIS residues to HID,
#                  HIE, or HIP depending on the predicted
#                  tautomer/ionization state. Internally uses
#                  Chimera(X) and pdb4amber.
#
#   param:         A complete AMBER-focussed workflow to
#                  prepare input files (coordinates and
#                  parameters) for MD simulation, from
#                  Complete PDB format files of the solute
#                  components (e.g. all-atom models of
#                  protein plus ligand). Includes automatic
#                  parameterization of non-standard
#                  residues (using gaff or gaff2), and addition
#                  of water boxes and neutralizing counterions.
#                  The tool only works for non-covalent ligands
#                  (no bonds between the ligand and the protein).
#                  Internally uses antechamber, parmchk2, and
#                  tleap.
#
#   alpha_loopfix: A tool to fix missing residues in a PDB file
#                  It uses ChimeraX to run AlphaFold and generates
#                  a new PDB file with the missing residues filled in.
#                  Note it does NOT complete non-loop residues, these
#                  may still lack heavy and/or hydrogen atoms.
#
#   complete:      A tool to complete a PDB file by adding missing
#                  atoms (heavy and hydrogen) using pdb4amber.
#
#
# Be aware that all these workflows can be confused by unusual
# or in some way particularly awkward systems (e.g. bad initial
# coordinates).
#
# SO PLEASE ALWAYS CHECK THE RESULTS CAREFULLY!
#

import mdtraj as mdt
import numpy as np
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from openbabel import openbabel as ob

from crossflow.tasks import SubprocessTask
from crossflow.filehandling import FileHandler
from functools import cache
import shutil

from pathlib import Path


def smiles_to_pdb(smi, pdb, pH=7.0):
    '''
    Convert an input SMILES representation to a PDB file

    Args:
        smi (str): Input SMILES string
        pdb (str): Name of out output PDB file
        pH (float): target pH

    Returns:
        charge (int): Formal charge on the molecule.

    '''
    obc = ob.OBConversion()
    obc.SetInAndOutFormats('smi', 'smi')

    obmol = ob.OBMol()
    obc.ReadString(obmol, smi)
    obmol.CorrectForPH(pH)
    smi_pH = obc.WriteString(obmol)
    charge = smi_pH.count('+]') - smi_pH.count('-]')
    mol_pH = Chem.MolFromSmiles(smi_pH)
    mol_pH_H = Chem.AddHs(mol_pH)
    rdDistGeom.EmbedMolecule(mol_pH_H)
    Chem.MolToPDBFile(mol_pH_H, pdb)
    return charge


def fix(inpdb, outpdb, keep_chains=None, trim=True):
    '''
    Remediate a PDB file using pdbfixer.

    Args:
        inpdb (str): input PDB file name
        outpdb (str): output PDB file name
        keep_chains (None or list): chains to keep (None = all)
        trim (bool): if True, do not rebuild any missing N- and C-terminal
                     residues

    '''
    _check_exists(inpdb)

    fixer = PDBFixer(filename=inpdb)
    if keep_chains:
        chains = list(fixer.topology.chains())
        removals = []
        for i, chain in enumerate(chains):
            if chain.id not in keep_chains:
                removals.append(i)

        print(f'Removing {len(removals)} unwanted chains')
        fixer.removeChains(removals)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    # keys are tuples (chain_index, residue_index)
    ignored_keys = []
    for key in keys:
        chain = chains[key[0]]
        if trim and (key[1] == 0 or key[1] == len(list(chain.residues()))):
            if key[1] == 0:
                print('Missing N-terminal residues (will be ignored):')
            else:
                print('Missing C-terminal residues (will be ignored):')
            print(' ', ', '.join(fixer.missingResidues[key]))
            ignored_keys.append(key)
        else:
            print('Missing at position ', key[1], '(will be fixed):')
            print(' ', ', '.join(fixer.missingResidues[key]))
    for key in ignored_keys:
        del fixer.missingResidues[key]

    fixer.findMissingAtoms()
    print('Missing heavy atoms:')
    for r in fixer.missingAtoms:
        print(' ', r.name, r.index, ':', ', '.join([a.name for a in
                                                   fixer.missingAtoms[r]]))

    fixer.addMissingAtoms()
    print(f'Remediated structure written to {outpdb}')
    PDBFile.writeFile(fixer.topology, fixer.positions, open(outpdb, 'w'))


def _check_available(cmd):
    '''
    Little utility to check a required command is available

    '''
    if shutil.which(cmd) is None:
        raise RuntimeError(f'Error - cannot find the {cmd} command')


def _check_exists(filename):
    '''
    Little utility to check if a required file is present

    '''
    if not Path(filename).exists():
        raise RuntimeError(f'Error - cannot find required file {filename}')


def add_h(inpdb, outpdb, chimera='chimera', mode='amber'):
    '''
    Add hydrogen atoms to a PDB format file, using Chimera or ChimeraX

    Args:
        inpdb (str): name of input PDB file
        outpdb (str): name of output PDB file
        chimera (str): command to invoke Chimera/ChimeraX
        mode (str): Adjust residue names for chosen software
                    (only 'amber' is currently supported).
    '''
    _check_exists(inpdb)
    _check_available(chimera)
    chimera_version = SubprocessTask(f"{chimera} --version > version")
    chimera_version.set_outputs(["version"])
    version = chimera_version.run()
    if 'ChimeraX' in version.read_text():
        chimera_type = 'chimerax'
    else:
        chimera_type = 'chimera'

    fh = FileHandler()
    script = fh.create('script')
    if chimera_type == 'chimerax':
        script.write_text('open infile.pdb\naddh\nsave outfile.pdb #1\nquit')
    else:
        script.write_text('open infile.pdb\naddh\nwrite 0  outfile.pdb\nstop')
    addh = SubprocessTask(f"{chimera} --nogui < script")
    addh.set_inputs(['script', 'infile.pdb'])
    addh.set_outputs(['outfile.pdb'])
    addh.set_constant('script', script)
    infile = fh.load(inpdb)
    outfile = addh.run(infile)
    if mode == 'amber':
        _check_available('pdb4amber')
        pdb4amber = SubprocessTask('pdb4amber -i infile.pdb -o outfile.pdb')
        pdb4amber.set_inputs(['infile.pdb'])
        pdb4amber.set_outputs(['outfile.pdb'])
        amberpdb = pdb4amber.run(outfile)
        amberpdb.save(outpdb)
    else:
        outfile.save(outpdb)
    t_orig = mdt.load_pdb(inpdb, standard_names=False)
    t_hydrated = mdt.load_pdb(outpdb, standard_names=False)
    n_h_orig = len(t_orig.topology.select('mass < 2.0'))
    n_h_hydrated = len(t_hydrated.topology.select('mass < 2.0'))
    n_h_added = n_h_hydrated - n_h_orig
    print(f'fix added {n_h_added} hydrogen atoms')
    if mode == 'amber':
        for i, r in enumerate(t_orig.topology.residues):
            r_new = t_hydrated.topology.residue(i)
            if r.name != r_new.name:
                print(f'{r.name}{r.resSeq} is now {r_new.name}')
    print(f'Hydrated structure written to {outpdb}')


def param(inpdb, outprmtop, outinpcrd, het_names=None, het_charges=None,
          forcefields=None,
          solvate=None,
          buffer=10.0):
    """
    Parameterize a PDB file for AMBER simulations.

    Args:
        inpdb (str): name of input PDB file
        outprmtop (str): name of output prmtop file
        outinpcrd (str): name of output inpcrd file
        het_names (None or list): 3-letter residue names for heterogens
        het_charges (None or list): Formal charges of each heterogen
        forcefields (None or list): List of forcefields to use
        solvate (None or str): Solvation option - can be 'box',
                               'cube', or 'oct'.
        buffer (float): minimum distance from any solute atom
                        to a periodic box boundary (Angstroms)

    """
    _check_exists(inpdb)
    if not forcefields:
        print('Warning: no forcefields specified, '
              'defaulting to "protein.ff14SB"')
        forcefields = ['protein.ff14SB']
    if solvate:
        if solvate not in ['oct', 'box', 'cube']:
            raise ValueError(f'Error: unrecognised solvate option "{solvate}"')
        water_ff = False
        for ff in forcefields:
            if 'water' in ff:
                water_ff = True
        if not water_ff:
            print('Warning: no water forcefield specified but'
                  ' solvation required.')
            print('Defaulting to "water.tip3p" forcefield.')
            forcefields.append('water.tip3p')

    if het_names is not None:
        if 'gaff' not in forcefields and 'gaff2' not in forcefields:
            print('Warning - heterogens are present but no gaff/gaff2 '
                  'forcefield has been specified.')
            print('Will default to using "gaff".')
            forcefields.append('gaff')

        for h_name, h_charge in zip(het_names, het_charges):
            print(f'parameterizing heterogen {h_name}')
            parameterize(inpdb, h_name, h_charge)

    prmtop, inpcrd = leap(inpdb, forcefields, het_names=het_names,
                          solvate=solvate, buffer=buffer)
    prmtop.save(outprmtop)
    print(f'Parameters written to {outprmtop}')
    inpcrd.save(outinpcrd)
    print(f'Coordinates written to {outinpcrd}')


@cache
def parameterize(source, residue_name, charge=0, gaff='gaff'):
    '''
    Paramaterize a non-standard residue (heterogen)

    Uses antechamber and parmchk2 to generate .mol2 and .frcmod files.

    Args:
       source (str): the PDB file name
       residue_name (str): the three-letter residue code for the heterogen
       charge: the formal charge on the heterogen

    Returns:
       list [mol2, frcmod]: crossflow.FileHandles

    '''
    if gaff not in ('gaff', 'gaff2'):
        raise ValueError(f'Error: unrecognised gaff option "{gaff}": '
                         'must be "gaff" or "gaff2"')
    _check_exists(source)
    traj = mdt.load(source)
    het_sel = traj.topology.select(f'resname {residue_name}')
    # A trajetory that contains all copies of the selected heterogen:
    traj_hets = traj.atom_slice(het_sel)
    # A trajetory that contains a single copy of the selected heterogen:
    traj_het = traj_hets.atom_slice(traj_hets.topology.select('resid 0'))
    # Remove bonds as they cause problems
    traj_het.topology._bonds = []
    # Run antechamber
    traj_het.save(f'{residue_name}.pdb')
    _check_available('antechamber')
    if gaff == 'gaff':
        antechamber = SubprocessTask('antechamber -i infile.pdb -fi pdb'
                                     ' -o outfile.mol2 -fo mol2 -c bcc'
                                     ' -nc {charge}')
    else:
        antechamber = SubprocessTask('antechamber -i infile.pdb -fi pdb'
                                     ' -o outfile.mol2 -fo mol2 -c bcc'
                                     ' -nc {charge} -at gaff2')
    antechamber.set_inputs(['infile.pdb', 'charge'])
    antechamber.set_outputs(['outfile.mol2'])
    outmol2 = antechamber.run(traj_het, charge)
    # run parmchk2
    _check_available('parmchk2')
    if gaff == 'gaff':
        parmchk = SubprocessTask('parmchk2 -i infile.mol2 -f mol2 -o'
                                 ' outfile.frcmod')
    else:
        parmchk = SubprocessTask('parmchk2 -s 2 -i infile.mol2 -f mol2'
                                 ' -o outfile.frcmod')
    parmchk.set_inputs(['infile.mol2'])
    parmchk.set_outputs(['outfile.frcmod'])
    frcmod = parmchk.run(outmol2)
    outmol2.save(f'{residue_name}.mol2')
    frcmod.save(f'{residue_name}.frcmod')


def leap(amberpdb, ff, het_names=None, solvate=None, buffer=10.0):
    '''
    Parameterize a molecular system using tleap.

    Args:
       amberpdb str): An Amber-compliant PDB file
       ff (list): The force fields to use.
       het_names (list): List of parameterised heterogens
       solvate (str or None): type of periodic box ('box', 'cube', or 'oct')
       buffer (float): Clearance between solute and any box edge (Angstroms)

    '''
    inputs = ['script', 'system.pdb']
    outputs = ['system.prmtop', 'system.inpcrd']
    script = "".join([f'source leaprc.{f}\n' for f in ff])

    if solvate:
        if solvate not in ['oct', 'box', 'cube']:
            raise ValueError(f'Error: unrecognised solvate option "{solvate}"')
    if het_names:
        if len(het_names) > 0:
            for r in het_names:
                _check_exists(f'{r}.frcmod')
                _check_exists(f'{r}.mol2')
                script += f'loadamberparams {r}.frcmod\n'
                script += f'{r} = loadmol2 {r}.mol2\n'
                inputs += [f'{r}.mol2', f'{r}.frcmod']

    script += "system = loadpdb system.pdb\n"
    if solvate == "oct":
        script += f"solvateoct system TIP3PBOX {buffer}\n"
    elif solvate == "cube":
        script += f"solvatebox system TIP3PBOX {buffer} iso\n"
    elif solvate == "box":
        script += f"solvatebox system TIP3PBOX {buffer}\n"
    if solvate is not None:
        script += "addions system Na+ 0\naddions system Cl- 0\n"
    script += "saveamberparm system system.prmtop system.inpcrd\nquit"

    tleap = SubprocessTask('tleap -f script')
    tleap.set_inputs(inputs)
    tleap.set_outputs(outputs)
    fh = FileHandler()
    scriptfile = fh.create('scriptfile')
    scriptfile.write_text(script)
    args = [scriptfile, amberpdb]
    if het_names:
        if len(het_names) > 0:
            for r in het_names:
                args += [f'{r}.mol2', f'{r}.frcmod']
    prmtop, inpcrd = tleap.run(*args)
    return prmtop, inpcrd


def alpha_loopfix(inpdb, outpdb,
                  max_shoulder_size=4,
                  min_ca_displacement=0.1,
                  chimerax='chimerax'):
    '''
    Fix missing residues in a PDB file using AlphaFold.
    Args:
        inpdb (str): input PDB file name
        outpdb (str): output PDB file name
        max_shoulder_size (int): maximum size of a loop shoulder
                                 to be replaced by AlphaFold
        min_ca_displacement (float): minimum displacement of the CA
                                     atom from the original structure
                                     to be replaced by AlphaFold
        chimerax (str): command to invoke ChimeraX

    '''
    _check_exists(inpdb)
    _check_available(chimerax)
    t = mdt.load(inpdb)

    bonds = [[b[0].index, b[1].index] for b in t.topology.bonds]
    bls = mdt.compute_distances(t, bonds)[0]
    bl_mean = bls.mean()
    for i, bl in enumerate(bls):
        if bl > bl_mean * 2:
            ir = t.topology.atom(bonds[i][0]).residue.index
            jr = t.topology.atom(bonds[i][1]).residue.index
            if ir != jr:
                t.topology._bonds.pop(i)
    ms = t.topology.find_molecules()
    if len(ms) == 1:
        print('No gaps to fill.')
        return
    elif len(ms) == 2:
        print('Structure contaons 1 gap.')
    else:
        print(f'Structure contains {len(ms)-1} gaps.')

    fh = FileHandler()
    script = fh.create('script')
    script.write_text('open infile.pdb\nalphafold match #1\n'
                      'save outfile.pdb #2\nquit')
    find_af = SubprocessTask(f"{chimerax} --nogui < script")
    find_af.set_inputs(['script', 'infile.pdb'])
    find_af.set_outputs(['outfile.pdb', 'STDOUT'])
    find_af.set_constant('script', script)

    outfile, log = find_af(t)

    for line in log.split('\n'):
        if 'sequence similarity' in line:
            print(line)
        elif 'WARNING' in line:
            print(line)

    ta = mdt.load(outfile)

    residues = []
    for i in range(ta.topology.n_residues):
        r = dict(alpha_res=ta.topology.residue(i), alpha_idx=i)
        residues.append(r)

    ref_seq = ta.topology.to_fasta()[0]
    new_indx = []
    d_from_gap = []
    for im, m in enumerate(ms):
        sel = [a.index for a in m]
        frag = t.topology.subset(sel)
        frag_seq = frag.to_fasta()[0]
        start = ref_seq.index(frag_seq)
        fl = len(frag_seq)
        for i in range(fl):
            new_indx.append(i + start)
            if im == 0:
                d_from_gap.append(fl - i)
            elif im == len(ms) - 1:
                d_from_gap.append(i+1)
            else:
                d_from_gap.append(min(i, fl-i))

    pair_d = []
    for i, j in enumerate(new_indx):
        sel_i = t.topology.select(f'resid {i} and name CA')[0]
        sel_j = ta.topology.select(f'resid {j} and name CA')[0]
        dxyz = t.xyz[0, sel_i] - ta.xyz[0, sel_j]
        pair_d.append(np.linalg.norm(dxyz))

    for i, j in enumerate(new_indx):
        residues[j]['xtal_idx'] = i
        residues[j]['d_from_gap'] = d_from_gap[i]
        residues[j]['xtal_res'] = t.topology.residue(i)
        residues[j]['pair_d'] = pair_d[i]

    if residues[0]['d_from_gap'] <= max_shoulder_size and \
       residues[0]['pair_d'] > min_ca_displacement:
        alist = [a.index for a in residues[0]['alpha_res'].atoms]
        new_traj = ta.atom_slice(alist)
    else:
        alist = [a.index for a in residues[0]['xtal_res'].atoms]
        new_traj = t.atom_slice(alist)
    for r in residues[1:]:
        if 'pair_d' not in r:
            # Use the alphafold structure for this residue
            alist = [a.index for a in r['alpha_res'].atoms]
            new_traj = new_traj.stack(ta.atom_slice(alist))
            print(f"Inserting missing residue {r['alpha_res']}.")
        elif (r['d_from_gap'] <= max_shoulder_size and
              r['pair_d'] > min_ca_displacement):
            # Use the alphafold structure for this residue
            alist = [a.index for a in r['alpha_res'].atoms]
            new_traj = new_traj.stack(ta.atom_slice(alist))
            print(f"Replacing {r['xtal_res']} with {r['alpha_res']}.")
        else:
            # Use the crystal structure for this residue
            alist = [a.index for a in r['xtal_res'].atoms]
            new_traj = new_traj.stack(t.atom_slice(alist))

    new_top = mdt.Topology()
    c = new_top.add_chain()
    for r in new_traj.topology.residues:
        nr = new_top.add_residue(r.name, c, r.resSeq)
        for a in r.atoms:
            new_top.add_atom(a.name, a.element, nr)
    new_traj.topology = new_top
    new_traj.save(outpdb)
    print(f'Fixed structure saved as {outpdb}.')


def complete(inpdb, outpdb):
    '''
    Complete a PDB file by adding missing atoms using pdb4amber.
    Args:
        inpdb (str): input PDB file name
        outpdb (str): output PDB file name
    '''
    _check_exists(inpdb)
    _check_available('pdb4amber')
    pdb4amber = SubprocessTask('pdb4amber -i in.pdb --add-missing-atoms'
                               ' --reduce > out.pdb')
    pdb4amber.set_inputs(['in.pdb'])
    pdb4amber.set_outputs(['out.pdb'])
    out = pdb4amber(inpdb)
    out.save(outpdb)
