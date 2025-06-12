import gc
import json
import os
from typing import NamedTuple, Optional, Any
from scipy.spatial.transform import Rotation
import numpy as np

from .util.ppdb import ppdb2nucseq
from .util.mutate import mutate


def load_clustering(master_clustering, nconf):
    assert master_clustering[0] == nconf
    pos = nconf + 1
    clustering_ind_ind = master_clustering[1:pos]
    n_clustering_ind = int(clustering_ind_ind.sum())
    clustering_ind = master_clustering[pos : n_clustering_ind + pos]
    pos = n_clustering_ind + pos
    n_clustering = int(clustering_ind.sum())
    assert n_clustering == len(master_clustering) - pos
    clustering = master_clustering[pos:] - 1
    return clustering, clustering_ind, clustering_ind_ind


class Library(NamedTuple):
    """
    Nucleotide library for a fragment sequence (potentially filtered and concatenated)

    Attributes:

    coordinates: np.ndarray
        (X, Y, 3) coordinate array.
        Contains at least the primary array, potentially replaced using the replacement array.
        If both a primary and extension coordinate array were present, "coordinates" is their concatenation.
        The (extension part of the) array may have been filtered based on conformer origin.


    nprimary: int
        The first nprimary conformers of "coordinates" are primary conformers.

    sequence: str
        Nucleotide sequence

    atom_mask: Optional[np.ndarray]
        Optional mask that indicates which atoms were selected by residue selection

    conformer_mask: Optional[np.ndarray]
        Optional mask that indicates which conformers are valid.
            Where False, the (extension) conformer must not be used, because of its PDB origin.
        Mutually exclusive with conformer_mapping

    conformer_mapping: Optional[np.ndarray]
        Optional, present only if the conformer array were pruned based on PDB origin.
        Maps the filtered-out conformer indices to the original ones.
        Mutually exclusive with conformer_mask and rotaconformers.

    rotaconformers: Optional[np.ndarray]
        Optional: discrete rotamers (3x3 rotation matrices), concatenated for all conformers.
        Use get_rotamers() to get the rotamers for a specific conformer.

    rotaconformers_index: Optional[np.ndarray]
        Optional: for each conformer, the index of the first rotamer in the rotaconformer array
        Use get_rotamers() to get the rotamers for a specific conformer.

    rotaconformers_clustering:
        Optional: the 3A clustering of the rotamers for each conformer
    """

    coordinates: np.ndarray

    nprimary: int

    sequence: str

    atom_mask: Optional[np.ndarray]

    conformer_mask: Optional[np.ndarray]

    conformer_mapping: Optional[np.ndarray]

    rotaconformers: Optional[np.ndarray]

    rotaconformers_index: Optional[np.ndarray]

    rotaconformers_clustering: Optional[Any]

    def get_rotamers(self, conformer: int):
        """Get all discrete rotamers for a conformer"""
        if conformer < 0 or conformer >= len(self.coordinates):
            raise ValueError
        if self.rotaconformers is None:
            raise AttributeError
        last = self.rotaconformers_index[conformer]
        if conformer == 0:
            first = 0
        else:
            first = self.rotaconformers_index[conformer - 1]
        return self.rotaconformers[first:last]

    def get_rotamer_clusters(self, conformer: int):
        first, last = self.rotaconformers_clustering[2][conformer : conformer + 2]
        cluster_indices = self.rotaconformers_clustering[1][first : last + 1]
        clustering = self.rotaconformers_clustering[0][
            cluster_indices[0] : cluster_indices[-1]
        ]

        return (
            cluster_indices - cluster_indices[0],
            clustering,
        )

    def _validate(self):
        if self.conformer_mask is not None:
            assert not self.conformer_mapping is not None
        if self.rotaconformers is not None:
            assert self.rotaconformers_index is not None
            assert len(self.rotaconformers_index) == len(self.coordinates)
            assert self.rotaconformers_index[-1] == len(self.rotaconformers)
        if self.conformer_mapping is not None:
            assert self.conformer_mask is None
            assert self.rotaconformers is None
            assert self.rotaconformers_index is None


class LibraryFactory:
    def __init__(
        self,
        sequence,
        primary_coordinates,
        *,
        template,
        rna,
        replacement_origins=None,
        replacement_coordinates=None,
        extension_coordinates=None,
        extension_origins=None,
        rotaconformers_file=None,
        rotaconformers_index_file=None,
        rotaconformers_extension_file=None,
        rotaconformers_extension_index_file=None,
        rotaconformers_clustering_file=None,
        rotaconformers_extension_clustering_file=None,
    ):
        """Factory class for nucleotide libraries.
        Instantiated by LibraryDirectory.load.
        Use self.create() to create actual libraries
        """
        if rotaconformers_file:
            assert rotaconformers_index_file
            if extension_coordinates is not None:
                assert rotaconformers_extension_file
                assert rotaconformers_extension_index_file
        self.sequence = sequence
        self.template = template
        self.rna = rna
        self.primary_coordinates = primary_coordinates
        self.nprimary = len(primary_coordinates)
        self.replacement_coordinates = replacement_coordinates
        self.replacement_origins = replacement_origins
        self.extension_coordinates = extension_coordinates
        self.extension_origins = extension_origins
        self.rotaconformers = None
        self.rotaconformers_clustering = None
        self.rotaconformers_index = None
        self.rotaconformers_file = rotaconformers_file
        self.rotaconformers_index_file = rotaconformers_index_file
        self.rotaconformers_extension_file = rotaconformers_extension_file
        self.rotaconformers_extension_index_file = rotaconformers_extension_index_file
        self.rotaconformers_clustering_file = rotaconformers_clustering_file
        self.rotaconformers_extension_clustering_file = (
            rotaconformers_extension_clustering_file
        )
        if (
            rotaconformers_clustering_file is not None
            and rotaconformers_extension_file is not None
        ):
            assert rotaconformers_extension_clustering_file is not None

    def load_rotaconformers(self, *, with_clustering=True) -> None:
        """Load the rotamers from file into memory.
        Convert them into rotation matrix form.
        This is expensive, both in terms of disk I/O and in terms of memory,
          and takes ~10s of computation.

        If clustering has been defined, load that as well.
        This is also expensive in terms of disk I/O and takes ~10s of computation.

        If the rotaconformers have been loaded already, do nothing."""
        if self.rotaconformers is not None:
            return
        rotaconformers = None
        rotaconformers_index = None
        if self.rotaconformers_file:
            rotaconformers0 = np.load(self.rotaconformers_file)
            rotaconformers_index = np.load(self.rotaconformers_index_file)
            assert len(rotaconformers_index) == len(self.primary_coordinates)
            if self.extension_coordinates is not None:
                rotaconformers_extension0 = np.load(self.rotaconformers_extension_file)
                rotaconformers_extension_index = np.load(
                    self.rotaconformers_extension_index_file
                )
                assert len(rotaconformers_extension_index) == len(
                    self.extension_coordinates
                )
                offset = len(rotaconformers0)
                rotaconformers0 = np.concatenate(
                    (rotaconformers0, rotaconformers_extension0)
                )
                del rotaconformers_extension0
                rotaconformers_index = np.concatenate(
                    (rotaconformers_index, rotaconformers_extension_index + offset)
                )

        rotaconformers = Rotation.from_rotvec(rotaconformers0).as_matrix()

        self.rotaconformers_clustering = None
        if self.rotaconformers_clustering_file is not None and with_clustering:
            master_clustering = np.load(self.rotaconformers_clustering_file)
            clustering, clustering_ind, clustering_ind_ind = load_clustering(
                master_clustering, len(self.primary_coordinates)
            )
            assert len(clustering_ind_ind) == len(self.primary_coordinates)
            ncoor = len(self.primary_coordinates)

            if self.extension_coordinates is not None and clustering is not None:
                master_clustering = np.load(
                    self.rotaconformers_extension_clustering_file
                )
                ext_clustering, ext_clustering_ind, ext_clustering_ind_ind = (
                    load_clustering(master_clustering, len(self.extension_coordinates))
                )
                assert len(ext_clustering_ind_ind) == len(self.extension_coordinates)
                clustering = np.concatenate(
                    (clustering, ext_clustering),
                )
                clustering_ind = np.concatenate((clustering_ind, ext_clustering_ind))
                clustering_ind_ind = np.concatenate(
                    (clustering_ind_ind, ext_clustering_ind_ind)
                )

                ncoor += len(self.extension_coordinates)
                assert len(clustering_ind_ind) == ncoor

            assert len(clustering) == len(rotaconformers)

            cs_clustering_ind = np.zeros(len(clustering_ind) + 1, int)
            cs_clustering_ind[1:] = np.cumsum(clustering_ind.astype(int))
            clustering_ind = cs_clustering_ind

            cs_clustering_ind_ind = np.zeros(len(clustering_ind_ind) + 1, int)
            cs_clustering_ind_ind[1:] = np.cumsum(clustering_ind_ind.astype(int))
            clustering_ind_ind = cs_clustering_ind_ind

            for conf in range(ncoor):
                i1, i2 = clustering_ind_ind[conf : conf + 2]
                f1, f2 = clustering_ind[i1], clustering_ind[i2]
                if conf == 0:
                    assert f1 == 0
                else:
                    assert f1 == rotaconformers_index[conf - 1]
                assert f2 == rotaconformers_index[conf]

            self.rotaconformers_clustering = (
                clustering,
                clustering_ind,
                clustering_ind_ind,
            )

        self.rotaconformers = rotaconformers
        self.rotaconformers_index = rotaconformers_index

    def unload_rotaconformers(self) -> None:
        """Detach the rotaconformers from the factory.
        Note that they remain attached to any created library objects.
        Memory will only be freed if all created libraries have been destroyed."""
        self.rotaconformers = None
        self.rotaconformers_index = None
        gc.collect()

    def rotamer_index_uint16(self) -> bool:
        """Returns if all rotamer indices for all conformers can fit in an uint16"""
        if self.rotaconformers_file is None:
            raise RuntimeError("No rotaconformers")
        if self.rotaconformers is None:
            raise RuntimeError("Rotaconformers must be loaded first")

        return np.diff(self.rotaconformers_index, prepend=0).max() < 2**16

    def create(
        self,
        pdb_code: Optional[str],
        *,
        prune_conformers: bool = False,
        nucleotide_mask: Optional[np.ndarray] = None,
        only_base: bool = False,
        with_rotaconformers: bool = False,
    ) -> Library:
        """Creates a Library, filtered by origin and/or nucleotide selection.

        - Filtering by origin. Provide a pdb_code.
            All conformers in the primary list with that PDB code as origin will be replaced.
            All conformers in the extension list with that PDB code will be invalidated.
                If prune_conformers, the conformers will be physically removed from the array,
                 and a mapping from
                This is incompatible with with_rotaconformers.

        - Filtering by nucleotide mask. Provide a boolean mask with the same length as the sequence.
            Only the atoms of the nucleotides where the mask is True are selected in the conformer coordinates
            Rotaconformers are unaffected.

        If only_base = True, only the coordinates of the base are returned.

        nucleotide mask and only_base are reflected in the atom mask of the library (Library.atom_mask).

        If with_rotaconformers is True, the factory's rotaconformers are attached to the library object.
        This is cheap in terms of memory. It requires that the rotaconformers have been loaded with .load_rotaconformers

        """
        if nucleotide_mask is not None:
            if len(nucleotide_mask) != len(self.sequence):
                raise ValueError(
                    "Nucleotide mask must have the same length as the sequence"
                )
        assert not (prune_conformers and with_rotaconformers)

        if (
            with_rotaconformers
            and self.rotaconformers_file is not None
            and self.rotaconformers is None
        ):
            raise RuntimeError("Rotaconformers must be loaded first")

        primary_coordinates = self.primary_coordinates
        if pdb_code is not None and self.replacement_origins is not None:
            to_replace = []
            for confnr, ori in enumerate(self.replacement_origins):
                if ori == pdb_code.lower():
                    to_replace.append(confnr)
            if to_replace:
                primary_coordinates = primary_coordinates.copy()
                primary_coordinates[to_replace] = self.replacement_coordinates[
                    to_replace
                ]
        conformer_mask = None
        conformer_mapping = None
        if self.extension_coordinates is not None:
            extension_coordinates = self.extension_coordinates
            if pdb_code is not None and self.extension_origins is not None:
                to_replace = []
                for confnr, ori in enumerate(self.extension_origins):
                    if ori == pdb_code.lower():
                        to_replace.append(confnr)
                if to_replace:
                    offset = len(primary_coordinates)
                    mask = np.zeros(len(extension_coordinates), bool)
                    mask[to_replace] = True
                    if prune_conformers:
                        mapping = np.arange(len(extension_coordinates), dtype=int)[
                            ~mask
                        ]
                        extension_coordinates = extension_coordinates[~mask]
                        conformer_mapping = np.concatenate(
                            (
                                np.arange(len(primary_coordinates), dtype=int),
                                mapping + offset,
                            )
                        )
                    else:
                        conformer_mask = np.concatenate(
                            (np.ones((len(primary_coordinates)), bool), ~mask)
                        )

            coordinates = np.concatenate((primary_coordinates, extension_coordinates))
            del primary_coordinates, extension_coordinates
        else:
            coordinates = primary_coordinates

        atom_mask = None
        if only_base or nucleotide_mask is not None:
            atom_mask = np.ones(len(self.template), bool)
            if nucleotide_mask is not None:
                nucseq, nuc_indices = ppdb2nucseq(
                    self.template, rna=self.rna, return_index=True
                )
                assert nucseq == self.sequence
                for n in range(len(self.sequence)):
                    if not nucleotide_mask[n]:
                        first, last = nuc_indices[n]
                        atom_mask[first:last] = 0
            if only_base:
                base = np.array(
                    [len(name) == 2 for name in self.template["name"]], bool
                )
                atom_mask &= base
            coordinates = coordinates[:, atom_mask]

        rotaconformers = None
        rotaconformers_index = None
        rotaconformers_clustering = None
        if with_rotaconformers:
            rotaconformers = self.rotaconformers
            rotaconformers_index = self.rotaconformers_index
            rotaconformers_clustering = self.rotaconformers_clustering

        result = Library(
            sequence=self.sequence,
            coordinates=coordinates,
            nprimary=self.nprimary,
            atom_mask=atom_mask,
            conformer_mask=conformer_mask,
            conformer_mapping=conformer_mapping,
            rotaconformers=rotaconformers,
            rotaconformers_index=rotaconformers_index,
            rotaconformers_clustering=rotaconformers_clustering,
        )
        result._validate()
        return result


class LibraryDirectory:
    def __init__(
        self,
        fraglen,
        filepattern,
        *,
        rna=True,
        replacement_filepattern=None,
        replacement_origin_filepattern=None,
        extension_filepattern=None,
        extension_origin_filepattern=None,
        rotaconformers_filepattern=None,
        rotaconformers_index_filepattern=None,
        rotaconformers_clustering_filepattern=None,
        rotaconformers_extension_filepattern=None,
        rotaconformers_extension_index_filepattern=None,
        rotaconformers_extension_clustering_filepattern=None,
    ):
        """
        Nucleotide fragment library directory. Must contain A/C conformer arrays.

        fraglen: length of nucleotide fragment.

        filepattern: file pattern for primary conformer array.

        When loading, every file pattern has (X * fraglen) replaced by the A/C sequence

        Optional:

        - rna: True if RNA (default), else DNA

        - replacement_filepattern: file pattern for the replacement conformer array.
        These contain intra-cluster (i.e. within the precision of the library)
        replacements for conformers that come from the same PDB that is being modeled.

        - replacement_origin_filepattern: file pattern that contains the PDB origin of
        each conformer, and the replacement index that is to be used.

        - extension_filepattern: file pattern for the extension conformer library.
        This library contains singletons (i.e. not replaceable) that improve the fit.

        - rotaconformers_filepattern: file pattern for the array of discrete rotamers
        Rotamers are scaled axis-angle 3-vectors, converted to 3x3 rotation matrices upon loading.
        The array is concatenated for all conformers.

        - rotaconformers_index_filepattern: file pattern for the rotaconformer indices.
        These are, for each conformer, the index of the first rotamer in the rotaconformer array,

        - rotaconformers_clustering_filepattern: file pattern for the internal 3A clustering of the rotaconformers.

        - rotaconformers_extension_filepattern: file pattern for array of discrete rotamers for the extension library

        - rotaconformers_extension_index_filepattern: file pattern for rotamers for the extension library

        - rotaconformers_extension_clustering_filepattern: file pattern for the internal 3A clustering of the rotaconformers.
        """
        self.fraglen = fraglen
        self.filepattern = filepattern
        self.rna = rna
        self.replacement_filepattern = replacement_filepattern
        self.replacement_origin_filepattern = replacement_origin_filepattern
        self.extension_filepattern = extension_filepattern
        self.extension_origin_filepattern = extension_origin_filepattern
        self.rotaconformers_filepattern = rotaconformers_filepattern
        self.rotaconformers_clustering_filepattern = (
            rotaconformers_clustering_filepattern
        )
        self.rotaconformers_index_filepattern = rotaconformers_index_filepattern
        self.rotaconformers_extension_filepattern = rotaconformers_extension_filepattern
        self.rotaconformers_extension_index_filepattern = (
            rotaconformers_extension_index_filepattern
        )
        self.rotaconformers_extension_clustering_filepattern = (
            rotaconformers_extension_clustering_filepattern
        )

    def load(
        self,
        sequence,
        template,
        *,
        with_extension=False,
        with_replacement=False,
        with_rotaconformers=False,
        with_clustering=False,
        loaded_libraries: dict[str, LibraryFactory] = None,
    ) -> LibraryFactory:
        """Load a library, returning it as a LibraryFactory.

        Optionally, a cache of previously loaded LibraryFactory instances
        may be provided, to prevent redundant loading from disk"""
        assert len(sequence) == self.fraglen

        if with_extension:
            assert self.extension_filepattern is not None
        if with_replacement:
            assert self.replacement_filepattern is not None
            assert self.replacement_origin_filepattern is not None
            if with_extension:
                assert self.extension_origin_filepattern is not None
        if with_rotaconformers:
            assert self.rotaconformers_filepattern is not None
            assert self.rotaconformers_index_filepattern is not None
            if with_extension:
                assert self.rotaconformers_extension_filepattern is not None
                assert self.rotaconformers_extension_index_filepattern is not None
        if with_clustering:
            assert with_rotaconformers
            assert self.rotaconformers_clustering_filepattern is not None
            if with_extension:
                assert self.rotaconformers_extension_clustering_filepattern is not None

        template_sequence = ppdb2nucseq(template, rna=self.rna)
        if template_sequence != sequence:
            raise ValueError(
                f"Template contains sequence {template_sequence}, whereas {sequence} is specified"
            )
        x = "X" * self.fraglen
        baseUT = "U" if self.rna else "T"
        seq0 = sequence.replace("G", "A").replace(baseUT, "C")
        if loaded_libraries is not None and seq0 in loaded_libraries:
            primary_coordinates0 = loaded_libraries[seq0].primary_coordinates
        else:
            primary_coordinates0 = np.load(self.filepattern.replace(x, seq0))
        primary_coordinates = mutate(primary_coordinates0, seq0, sequence)
        extension_coordinates = None
        if with_extension:
            if loaded_libraries is not None and seq0 in loaded_libraries:
                extension_coordinates0 = loaded_libraries[seq0].extension_coordinates
            else:
                extension_coordinates0 = np.load(
                    self.extension_filepattern.replace(x, seq0)
                )
            extension_coordinates = mutate(extension_coordinates0, seq0, sequence)
        replacement_coordinates = None
        replacement_origins = None
        extension_origins = None
        if with_replacement:
            replacement_coordinates0 = np.load(
                self.replacement_filepattern.replace(x, seq0)
            )
            if len(replacement_coordinates0) != len(primary_coordinates):
                raise ValueError(
                    "Shape mismatch between primary coordinates and replacement coordinates"
                )
            replacement_coordinates = mutate(replacement_coordinates0, seq0, sequence)
            with open(self.replacement_origin_filepattern.replace(x, seq0)) as f:
                replacement_origins0 = f.read()
            replacement_origins = [
                l.split()[0] for l in replacement_origins0.splitlines()[1:]
            ]
            if len(replacement_origins) != len(primary_coordinates):
                raise ValueError(
                    "Length mismatch between primary coordinates and origins"
                )
            if with_extension:
                with open(self.extension_origin_filepattern.replace(x, seq0)) as f:
                    extension_origins0 = f.read()
                extension_origins = [l.strip() for l in extension_origins0.splitlines()]
            if len(extension_origins) != len(extension_coordinates):
                raise ValueError(
                    "Length mismatch between extension coordinates and extension origins"
                )

        rotaconformers_file = None
        rotaconformers_index_file = None
        rotaconformers_extension_file = None
        rotaconformers_extension_index_file = None
        rotaconformers_clustering_file = None
        rotaconformers_extension_clustering_file = None
        if with_rotaconformers:
            rotaconformers_file = self.rotaconformers_filepattern.replace(x, seq0)
            assert os.path.exists(rotaconformers_file)
            rotaconformers_index_file = self.rotaconformers_index_filepattern.replace(
                x, seq0
            )
            assert os.path.exists(rotaconformers_index_file)
            if with_extension:
                rotaconformers_extension_file = (
                    self.rotaconformers_extension_filepattern.replace(x, seq0)
                )
                assert os.path.exists(rotaconformers_extension_file)
                rotaconformers_extension_index_file = (
                    self.rotaconformers_extension_index_filepattern.replace(x, seq0)
                )
                assert os.path.exists(rotaconformers_extension_index_file)
            if with_clustering:
                rotaconformers_clustering_file = (
                    self.rotaconformers_clustering_filepattern.replace(x, seq0)
                )
                assert os.path.exists(rotaconformers_clustering_file)
                rotaconformers_extension_clustering_file = (
                    self.rotaconformers_extension_clustering_filepattern.replace(
                        x, seq0
                    )
                )
                assert os.path.exists(rotaconformers_extension_clustering_file)

        return LibraryFactory(
            sequence,
            primary_coordinates,
            template=template,
            rna=self.rna,
            replacement_coordinates=replacement_coordinates,
            replacement_origins=replacement_origins,
            extension_coordinates=extension_coordinates,
            extension_origins=extension_origins,
            rotaconformers_file=rotaconformers_file,
            rotaconformers_extension_file=rotaconformers_extension_file,
            rotaconformers_index_file=rotaconformers_index_file,
            rotaconformers_extension_index_file=rotaconformers_extension_index_file,
            rotaconformers_clustering_file=rotaconformers_clustering_file,
            rotaconformers_extension_clustering_file=rotaconformers_extension_clustering_file,
        )
