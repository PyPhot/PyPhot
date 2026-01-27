"""
Routines for matching frames to certain types or each other.

Bookkeeping from PyPeIt.
"""


import os
import textwrap
import numpy as np
from collections import OrderedDict


class BitMask:
    r"""
    Generic class to handle and manipulate bitmasks.  The input list of
    bit names (keys) must be unique, except that values of 'NULL' are
    ignored.  The index in the input keys determines the bit value;
    'NULL' keys are included in the count.  For example::

        >>> from pyphot.par.framematch import BitMask
        >>> keys = [ 'key1', 'key2', 'NULL', 'NULL', 'key3' ]
        >>> bm = BitMask(keys)
        >>> bm.info()
                 Bit: key1 = 0

                 Bit: key2 = 1

                 Bit: key3 = 4

    .. todo::
        - Have the class keep the mask values internally instead of
          having it only operate on the mask array...

    Args:
        keys (:obj:`str`, :obj:`list`):
            List of keys (or single key) to use as the bit name.  Each
            key is given a bit number ranging from 0..N-1.

        descr (:obj:`str`, :obj:`list`, optional):
            List of descriptions (or single discription) provided by
            :func:`info` for each bit.  No descriptions by default.

    Raises:
        ValueError:
            Raised if more than 64 bits are provided.
        TypeError:
            Raised if the provided `keys` do not have the correct type.

    Attributes:
        nbits (int):
            Number of bits
        bits (dict):
            A dictionary with the bit name and value
        descr (`np.ndarray`_):
            List of bit descriptions
        max_value (int):
            The maximum valid bitmask value given the number of bits.
    """
    prefix = 'BIT'
    version = None

    def __init__(self, keys, descr=None):

        _keys = keys if hasattr(keys, '__iter__') else [keys]
        _keys = np.atleast_1d(_keys).ravel()
        _descr = None if descr is None else np.atleast_1d(descr).ravel()

        #        if not np.all([isinstance(k, str) for k in _keys]):
        #            raise TypeError('Input keys must have string type.')
        if _descr is not None:
            if not all([isinstance(d, str) for d in _descr]):
                raise TypeError('Input descriptions must have string type.')
            if len(_descr) != len(_keys):
                raise ValueError('Number of listed descriptions not the same as number of keys.')

        # Do not allow for more that 64 bits
        if len(_keys) > 64:
            raise ValueError('Can only define up to 64 bits!')

        # Allow for multiple NULL keys; but check the rest for
        # uniqueness
        diff = set(_keys) - set(['NULL'])
        if len(diff) != np.unique(_keys[[k != 'NULL' for k in _keys]]).size:
            raise ValueError('All input keys must be unique.')

        # Initialize the attributes
        self.nbits = len(_keys)
        self.bits = {k: i for i, k in enumerate(_keys)}
        self.max_value = (1 << self.nbits) - 1
        self.descr = _descr

    def _prep_flags(self, flag):
        """Prep the flags for use."""
        # Flags must be a numpy array
        _flag = np.array(self.keys()) if flag is None else np.atleast_1d(flag).ravel()
        # NULL flags not allowed
        if np.any([f == 'NULL' for f in _flag]):
            raise ValueError('Flag name NULL is not allowed.')
        # Flags should be among the bitmask keys
        indx = np.array([f not in self.keys() for f in _flag])
        if np.any(indx):
            raise ValueError('The following bit names are not recognized: {0}'.format(
                ', '.join(_flag[indx])))
        # # Flags should be strings
        #        if np.any([ not isinstance(f, str) for f in _flag ]):
        #            raise TypeError('Provided bit names must be strings!')
        return _flag

    @staticmethod
    def _fill_sequence(keys, vals, descr=None):
        r"""
        Fill bit sequence with NULL keys if bit values are not
        sequential.

        The instantiation of :class:`BitMask` does not include the
        value of the bit, it just assumes that the bits are in
        sequence such that the first key has a value of 0, and the
        last key has a value of N-1. This is a convenience function
        that fills the list of keys with 'NULL' for bit values that
        are non-sequential. This is used primarily for instantiation
        the BitMask from bits written to a file where the NULL bits
        have been skipped.

        Args:
            keys (:obj:`list`, :obj:`str`):
                Bit names
            vals (:obj:`list`, :obj:`int`):
                Bit values
            descr (:obj:`list`, :obj:`str`, optional):
                The description of each bit. If None, no bit
                descriptions are defined.

        Returns:
            `np.ndarray`_: Three 1D arrays with the filled keys,
            values, and descriptions.

        Raises:
            ValueError: Raised if a bit value is less than 0.
        """
        _keys = np.atleast_1d(keys).ravel()
        _vals = np.atleast_1d(vals).ravel()
        _descr = None if descr is None else np.atleast_1d(descr).ravel()

        if np.amin(_vals) < 0:
            raise ValueError('No bit cannot be less than 0!')
        minv = np.amin(_vals)
        maxv = np.amax(_vals)

        if minv != 0 or maxv != len(_vals) - 1:
            diff = list(set(np.arange(maxv)) - set(_vals))
            _vals = np.append(_vals, diff)
            _keys = np.append(_keys, np.array(['NULL'] * len(diff)))
            if _descr is not None:
                _descr = np.append(_descr, np.array([''] * len(diff)))

        return _keys, _vals, _descr

    def keys(self):
        """
        Return a list of the bit keywords.

        Keywords are sorted by their bit value and 'NULL' keywords are
        ignored.

        Returns:
            list: List of bit keywords.
        """
        k = np.array(list(self.bits.keys()))
        return k[[_k != 'NULL' for _k in k]].tolist()

    def info(self):
        """
        Print the list of bits and, if available, their descriptions.
        """
        try:
            tr, tcols = np.array(os.popen('stty size', 'r').read().split()).astype(int)
            tcols -= int(tcols * 0.1)
        except:
            tr = None
            tcols = None

        for k, v in sorted(self.bits.items(), key=lambda x: (x[1], x[0])):
            if k == 'NULL':
                continue
            print('         Bit: {0} = {1}'.format(k, v))
            if self.descr is not None:
                if tcols is not None:
                    print(textwrap.fill(' Description: {0}'.format(self.descr[v]), tcols))
                else:
                    print(' Description: {0}'.format(self.descr[v]))
            print(' ')

    def minimum_dtype(self, asuint=False):
        """
        Return the smallest int datatype that is needed to contain all
        the bits in the mask.  Output as an unsigned int if requested.

        Args:
            asuint (:obj:`bool`, optional):
                Return an unsigned integer type.  Signed types are
                returned by default.

        .. warning::
            uses int16 if the number of bits is less than 8 and
            asuint=False because of issue astropy.io.fits has writing
            int8 values.
        """
        if self.nbits < 8:
            return np.uint8 if asuint else np.int16
        if self.nbits < 16:
            return np.uint16 if asuint else np.int16
        if self.nbits < 32:
            return np.uint32 if asuint else np.int32
        return np.uint64 if asuint else np.int64

    def flagged(self, value, flag=None):
        """
        Determine if a bit is on in the provided bitmask value.  The
        function can be used to determine if any individual bit is on or
        any one of many bits is on.

        Args:
            value (int, array-like):
                Bitmask value.  It should be less than or equal to
                :attr:`max_value`; however, that is not checked.
            flag (str, array-like, optional):
                One or more bit names to check.  If None, then it checks
                if *any* bit is on.

        Returns:
            bool: Boolean flags that the provided flags (or any flag) is
            on for the provided bitmask value.  Shape is the same as
            `value`.

        Raises:
            KeyError: Raised by the dict data type if the input *flag*
                is not one of the valid bitmask names.
            TypeError: Raised if the provided *flag* does not contain
                one or more strings.
        """
        _flag = self._prep_flags(flag)

        out = value & (1 << self.bits[_flag[0]]) != 0
        if len(_flag) == 1:
            return out

        nn = len(_flag)
        for i in range(1, nn):
            out |= (value & (1 << self.bits[_flag[i]]) != 0)
        return out

    def flagged_bits(self, value):
        """
        Return the list of flagged bit names for a single bit value.

        Args:
            value (int):
                Bitmask value.  It should be less than or equal to
                :attr:`max_value`; however, that is not checked.

        Returns:
            list: List of flagged bit value keywords.

        Raises:
            KeyError:
                Raised by the dict data type if the input *flag* is not
                one of the valid bitmask names.
            TypeError:
                Raised if the provided *flag* does not contain one or
                more strings.
        """
        if not np.issubdtype(type(value), np.integer):
            raise TypeError('Input must be a single integer.')
        if value <= 0:
            return []
        keys = np.array(self.keys())
        indx = np.array([1 << self.bits[k] & value != 0 for k in keys])
        return (keys[indx]).tolist()

    def toggle(self, value, flag):
        """
        Toggle a bit in the provided bitmask value.

        Args:
            value (int, array-like):
                Bitmask value.  It should be less than or equal to
                :attr:`max_value`; however, that is not checked.
            flag (str, array-like):
                Bit name(s) to toggle.

        Returns:
            array-like: New bitmask value after toggling the selected
            bit.

        Raises:
            ValueError:
                Raised if the provided flag is None.
        """
        if flag is None:
            raise ValueError('Provided bit name cannot be None.')

        _flag = self._prep_flags(flag)

        out = value ^ (1 << self.bits[_flag[0]])
        if len(_flag) == 1:
            return out.astype(value.dtype)

        nn = len(_flag)
        for i in range(1, nn):
            out ^= (1 << self.bits[_flag[i]])
        return out.astype(value.dtype)

    def turn_on(self, value, flag):
        """
        Ensure that a bit is turned on in the provided bitmask value.

        Args:
            value (:obj:`int`, `np.ndarray`_):
                Bitmask value. It should be less than or equal to
                :attr:`max_value`; however, that is not checked.
            flag (:obj:`list`, `np.ndarray`, :obj:`str`):
                Bit name(s) to turn on.

        Returns:
            :obj:`int`: New bitmask value after turning on the
            selected bit.

        Raises:
            ValueError:
                Raised by the dict data type if the input ``flag`` is
                not one of the valid bitmask names or if it is None.
        """
        if flag is None:
            raise ValueError('Provided bit name cannot be None.')

        _flag = self._prep_flags(flag)

        out = value | (1 << self.bits[_flag[0]])
        if len(_flag) == 1:
            return out.astype(value.dtype)

        nn = len(_flag)
        for i in range(1, nn):
            out |= (1 << self.bits[_flag[i]])
        return out.astype(value.dtype)

    def turn_off(self, value, flag):
        """
        Ensure that a bit is turned off in the provided bitmask value.

        Args:
            value (int, array-like):
                Bitmask value.  It should be less than or equal to
                :attr:`max_value`; however, that is not checked.
            flag (str, array-like):
                Bit name(s) to turn off.

        Returns:
            int: New bitmask value after turning off the selected bit.

        Raises:
            ValueError:
                Raised by the dict data type if the input ``flag`` is
                not one of the valid bitmask names or if it is None.
        """
        if flag is None:
            raise ValueError('Provided bit name cannot be None.')

        _flag = self._prep_flags(flag)

        out = value & ~(1 << self.bits[_flag[0]])
        if len(_flag) == 1:
            return out.astype(value.dtype)

        nn = len(_flag)
        for i in range(1, nn):
            out &= ~(1 << self.bits[_flag[i]])
        return out.astype(value.dtype)

    def consolidate(self, value, flag_set, consolidated_flag):
        """
        Consolidate a set of flags into a single flag.
        """
        indx = self.flagged(value, flag=flag_set)
        value[indx] = self.turn_on(value[indx], consolidated_flag)
        return value

    def unpack(self, value, flag=None):
        """
        Construct boolean arrays with the selected bits flagged.

        Args:
            value (`np.ndarray`_):
                The bitmask values to unpack.
            flag (:obj:`str`, :obj:`list`, optional):
                The specific bits to unpack.  If None, all values are
                unpacked.
        Returns:
            tuple: A tuple of boolean np.ndarrays flagged according
            to each bit.
        """
        _flag = self._prep_flags(flag)
        return tuple([self.flagged(value, flag=f) for f in _flag])

    def to_header(self, hdr, prefix=None, quiet=False):
        """
        Write the bits to a fits header.

        .. todo::
            - This is very similar to the function in ParSet.  Abstract
              to a general routine?
            - The comment might have a limited length and be truncated.

        Args:
            hdr (`astropy.io.fits.Header`):
                Header object for the parameters. Modified in-place.
            prefix (:obj:`str`, optional):
                Prefix to use for the header keywords, which
                overwrites the string defined for the class. If None,
                uses the default for the class.
            quiet (:obj:`bool`, optional):
                Suppress print statements.
        """
        if prefix is None:
            prefix = self.prefix
        maxbit = max(list(self.bits.values()))
        ndig = int(np.log10(maxbit)) + 1
        for key, value in sorted(self.bits.items(), key=lambda x: (x[1], x[0])):
            if key == 'NULL':
                continue
            hdr['{0}{1}'.format(prefix, str(value).zfill(ndig))] = (key, self.descr[value])

    @classmethod
    def from_header(cls, hdr, prefix=None):
        """
        Instantiate the BitMask using data parsed from a fits header.

        .. todo::
            - This is very similar to the function in ParSet.  Abstract
              to a general routine?
            - If comments are truncated by the comment line length,
              they'll be different than a direct instantiation.

        Args:
            hdr (`astropy.io.fits.Header`):
                Header object with the bits.
            prefix (:obj:`str`, optional):
                Prefix of the relevant header keywords, which
                overwrites the string defined for the class. If None,
                uses the default for the class.
        """
        if prefix is None:
            prefix = cls.prefix
        # Parse the bits from the header
        keys, values, descr = cls.parse_bits_from_hdr(hdr, prefix)
        # Fill in any missing bits
        keys, values, descr = cls._fill_sequence(keys, values, descr=descr)
        # Make sure the bits are sorted
        srt = np.argsort(values)
        # Instantiate the BitMask
        return cls(keys[srt], descr=descr[srt])

    @staticmethod
    def parse_bits_from_hdr(hdr, prefix):
        """
        Parse bit names, values, and descriptions from a fits header.

        .. todo::
            - This is very similar to the function in ParSet.  Abstract
              to a general routine?

        Args:
            hdr (`astropy.io.fits.Header`):
                Header object with the bits.
            prefix (:obj:`str`):
                The prefix used for the header keywords.

        Returns:
            Three lists are returned providing the bit names, values,
            and descriptions.
        """
        keys = []
        values = []
        descr = []
        for k, v in hdr.items():
            # Check if this header keyword starts with the required
            # prefix
            if k[:len(prefix)] == prefix:
                try:
                    # Try to convert the keyword without the prefix
                    # into an integer. Bits are 0 indexed and written
                    # to the header that way.
                    i = int(k[len(prefix):])
                except ValueError:
                    # Assume the value is some other random keyword that
                    # starts with the prefix but isn't a parameter
                    continue

                # Assume we've found a bit entry. Parse the bit name
                # and description and add to the compiled list
                keys += [v]
                values += [i]
                descr += [hdr.comments[k]]
        return keys, values, descr


class FrameTypeBitMask(BitMask):
    """
    Define a bitmask to set the frame types.

    Frame types can be arc, bias, dark, pixelflat, science,
    standard, or trace.
    """
    def __init__(self):
        # TODO: This needs to be an OrderedDict for now to ensure that
        # the bits assigned to each key is always the same. As of python
        # 3.7, normal dict types are guaranteed to preserve insertion
        # order as part of its data model. When/if we require python
        # 3.7, we can remove this (and other) OrderedDict usage in favor
        # of just a normal dict.
        frame_types = OrderedDict([
                        ('bias', 'Bias readout for detector bias subtraction'),
                        ('dark', 'Shuttered exposure to measure dark current'),
                        ('pixelflat', 'Flat-field exposure used for pixel-to-pixel response'),
                        ('illumflat', 'Flat-field exposure used for illumination flat'),
                        ('supersky', 'On-sky observation used for super sky flat'),
                        ('fringe', 'On-sky observation used for fringe frame'),
                        ('science', 'On-sky observation of a primary target'),
                        ('standard', 'On-sky observation of a flux calibrator'),
                                  ])
        super(FrameTypeBitMask, self).__init__(list(frame_types.keys()),
                                               descr=list(frame_types.values()))

    def type_names(self, type_bits, join=True):
        """
        Use the type bits to get the type names for each frame.

        .. todo::
            - This should probably be a general function in
              :class:`pyphot.bitmask.BitMask`
    
        Args:
            type_bits (int, list, `np.ndarray`_):
                The bit mask for each frame.
            bitmask (:class:`pyphot.bitmask.BitMask`, optional):
                The bit mask used to pull out the bit names.  Uses
                :class:`FrameTypeBitMask` by default.
            join (:obj:`bool`, optional):
                Instead of providing a list of type names for items with
                multiple bits tripped, joint the list into a single,
                comma-separated string.
    
        Returns:
            list: List of the frame types for each frame.  Each frame can
            have multiple types, meaning the 2nd axis is not necessarily the
            same length for all frames.
        """
        _type_bits = np.atleast_1d(type_bits)
        out = []
        for b in _type_bits:
            n = self.flagged_bits(b)
            if len(n) == 0:
                n = ['None']
            out += [','.join(n)] if join else [n]
        return out[0] if isinstance(type_bits, np.integer) else out
    

def check_frame_exptime(exptime, exprng):
    """
    Check that the exposure time is within the provided range.
        
    Args:
        exptime (`np.ndarray`_):
            Exposure times to check; allowed to be None.
        exprng (array-like):
            An array with the minimum and maximum exposure.  The limits
            are *exclusive* and a limit of None means there is no limit.
        
    Returns:
        `np.ndarray`_: A boolean array that is True for all times within
        the provided range. The value is False for any exposure time that is
        None or outside the provided range.
        
    Raises:
        ValueError:
            Raised if the length of `exprng` is not 2.
    """
    # Instantiate with all true
    indx = exptime != None
    if exprng is None:
        # No range specified
        return indx
    if len(exprng) != 2:
        # Range not correctly input
        raise ValueError('exprng must have two elements.')
    if exprng[0] is not None:
        indx[indx] &= (exptime[indx] > exprng[0])
    if exprng[1] is not None:
        indx[indx] &= (exptime[indx] <= exprng[1])
    return indx

