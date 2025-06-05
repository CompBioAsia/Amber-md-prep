import mdtraj as mdt
from pdbfixer import PDBFixer
from openmm.app import PDBFile

from crossflow.kernels import SubprocessKernel
from crossflow.filehandling import FileHandler
from functools import cache
import shutil


def fix(inpdb, outpdb, keep_chains=None, trim=True):
    '''
    Remediate a PDB file using pdbfixer.

    Args:
        inpdb (str): input PDB file name
        outpdb (str): output PDB file name
        keep_chains (None or list): chains to keep
        trim (bool): if True, do not rebuild any missing N- and C-terminal
                     residues

    '''

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


def check_available(cmd):
    '''
    Little utility to check a required command is available

    '''
    if shutil.which(cmd) is None:
        raise RuntimeError(f'Error - cannot find the {cmd} command')


def add_h(inpdb, outpdb, chimera='chimera', mode='amber'):
    '''
    Add hydrogen atoms to a PDB format file, using Chimera

    Args:
        inpdb (str): name of input PDB file
        outpdb (str): name of output PDB file
        chimera (str): command to invoke Chimera
        mode (str): Adjust residue names for chosen software
                    (only 'amber' is currently supported).
    '''
    check_available(chimera)
    fh = FileHandler()
    script = fh.create('script')
    script.write_text('open infile.pdb\naddh\nwrite 0  outfile.pdb\nstop')
    addh = SubprocessKernel(f"{chimera} --nogui < script")
    addh.set_inputs(['script', 'infile.pdb'])
    addh.set_outputs(['outfile.pdb'])
    addh.set_constant('script', script)
    infile = fh.load(inpdb)
    outfile = addh.run(infile)
    if mode == 'amber':
        check_available('pdb4amber')
        pdb4amber = SubprocessKernel('pdb4amber -i infile.pdb -o outfile.pdb')
        pdb4amber.set_inputs(['infile.pdb'])
        pdb4amber.set_outputs(['outfile.pdb'])
        amberpdb = pdb4amber.run(outfile)
        amberpdb.save(outpdb)
    else:
        outfile.save(outpdb)
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

    """
    if not forcefields:
        print('Warning: no forcefields specified, '
              'defaulting to "protein.ff19SB"')
        forcefields = ['protein.ff19SB']
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
    traj = mdt.load(source)
    het_sel = traj.topology.select(f'resname {residue_name}')
    # A trajetory that contains all copies of the selected heterogen:
    traj_hets = traj.atom_slice(het_sel)
    # A trajetory that contains a single copy of the selected heterogen:
    traj_het = traj_hets.atom_slice(traj_hets.topology.select('resid 0'))
    # Run antechamber
    traj_het.save(f'{residue_name}.pdb')
    check_available('antechamber')
    if gaff == 'gaff':
        antechamber = SubprocessKernel('antechamber -i infile.pdb -fi pdb'
                                       ' -o outfile.mol2 -fo mol2 -c bcc'
                                       ' -nc {charge}')
    else:
        antechamber = SubprocessKernel('antechamber -i infile.pdb -fi pdb'
                                       ' -o outfile.mol2 -fo mol2 -c bcc'
                                       ' -nc {charge} -at gaff2')
    antechamber.set_inputs(['infile.pdb', 'charge'])
    antechamber.set_outputs(['outfile.mol2'])
    outmol2 = antechamber.run(traj_het, charge)
    # run parmchk2
    check_available('parmchk2')
    if gaff == 'gaff':
        parmchk = SubprocessKernel('parmchk2 -i infile.mol2 -f mol2 -o'
                                   ' outfile.frcmod')
    else:
        parmchk = SubprocessKernel('parmchk2 -s 2 -i infile.mol2 -f mol2'
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

    tleap = SubprocessKernel('tleap -f script')
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
