# DEFINITION:
# Reads Igor's (Wavemetric) binary wave format, .ibw, files.
#
# ALGORITHM:
# Parsing proper to version 2, 3, or version 5 (see Technical notes TN003.ifn:
# http://mirror.optus.net.au/pub/wavemetrics/IgorPro/Technical_Notes/) and data
# type 2 or 4 (non complex, single or double precision vector, real values). 
# 
# AUTHORS:
# Python port: A. Seeholzer October 2008
#
# VERSION: 0.1
#
# COMMENTS:
# Only tested for version 2 igor files for now, testing for 3 and 5 remains to be done.
# More header data could be passed back if wished. For significace of ignored bytes see
# the technical notes linked above. 

import struct
from time import clock
import numpy

def flatten(tup):
    out = ''
    for ch in tup:
        out += ch
    return out

def read(filename):
    '''
    DEFINITION
    Reads Igor's (Wavemetric) binary wave format, .ibw, files.

    INPUT
    filename: filename with path in string, e.g. "usr/home/example.ibw" 

    OUTPUT
    data: vector (single precision) of values of the wave containted in the file
    dUnits: string: saved values of the units of data (char).
    '''
    
    f = open(filename,"rb")
    
    ####################### ORDERING
    # machine format for IEEE floating point with big-endian
    # byte ordering
    # MacIgor use the Motorola big-endian 'b'
    # WinIgor use Intel little-endian 'l'
    # If the first byte in the file is non-zero, then the file is a WinIgor
    firstbyte = struct.unpack('b',f.read(1))[0]
    if firstbyte==0:
        format = '>'
    else:
        format = '<'
    
    #######################  CHECK VERSION
    
    f.seek(0)
    version = struct.unpack(format+'h',f.read(2))[0]
    
    #######################  READ DATA AND ACCOMPANYING INFO
    if version == 2 or version==3:

        # pre header
        wfmSize = struct.unpack(format+'i',f.read(4))[0] # The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
        noteSize = struct.unpack(format+'i',f.read(4))[0] # The size of the note text.
        if version==3:
            formulaSize = struct.unpack(format+'i',f.read(4))[0]
        pictSize = struct.unpack(format+'i',f.read(4))[0] # Reserved. Write zero. Ignore on read.
        checksum = struct.unpack(format+'H',f.read(2))[0] # Checksum over this header and the wave header.

        # wave header
        dtype = struct.unpack(format+'h',f.read(2))[0]
        if dtype == 2:
            dtype = numpy.float32(.0).dtype
        elif dtype == 4:
            dtype = numpy.double(.0).dtype
        else:
            assert False, "Wave is of type '%i', not supported" % dtype
        dtype = dtype.newbyteorder(format)
        
        ignore = f.read(4) # 1 uint32
        bname = flatten(struct.unpack(format+'20c',f.read(20)))
        ignore = f.read(4) # 2 int16
        ignore = f.read(4) # 1 uint32
        dUnits = flatten(struct.unpack(format+'4c',f.read(4)))
        xUnits = flatten(struct.unpack(format+'4c',f.read(4)))
        npnts = struct.unpack(format+'i',f.read(4))[0]
        amod = struct.unpack(format+'h',f.read(2))[0]
        dx = struct.unpack(format+'d',f.read(8))[0]
        x0 = struct.unpack(format+'d',f.read(8))[0]
        ignore = f.read(4) # 2 int16
        fsValid = struct.unpack(format+'h',f.read(2))[0]
        # RICHARD: why is this here??
        #x0 = float(x0*fsValid);
        #dx = float(dx*fsValid);
        topFullScale = struct.unpack(format+'d',f.read(8))[0]
        botFullScale = struct.unpack(format+'d',f.read(8))[0]
        ignore = f.read(16) # 16 int8
        modDate = struct.unpack(format+'I',f.read(4))[0]
        ignore = f.read(4) # 1 uint32
        # Numpy algorithm works a lot faster than struct.unpack 
        wdata = numpy.fromfile(f,dtype)
        
    elif version == 5:
        # pre header
        checksum = struct.unpack(format+'H',f.read(2))[0] # Checksum over this header and the wave header.
        wfmSize = struct.unpack(format+'i',f.read(4))[0] # The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
        formulaSize = struct.unpack(format+'i',f.read(4))[0]
        noteSize = struct.unpack(format+'i',f.read(4))[0] # The size of the note text.
        dataEUnitsSize =  struct.unpack(format+'i',f.read(4))[0]
        dimEUnitsSize=  struct.unpack(format+'4i',f.read(16))
        dimLabelsSize =  struct.unpack(format+'4i',f.read(16))
        sIndicesSize =  struct.unpack(format+'i',f.read(4))[0]
        optionSize1 =  struct.unpack(format+'i',f.read(4))[0]
        optionSize2 =  struct.unpack(format+'i',f.read(4))[0]

        # header
        ignore = f.read(4)
        CreationDate =  struct.unpack(format+'I',f.read(4))[0]
        modData =  struct.unpack(format+'I',f.read(4))[0]
        npnts =  struct.unpack(format+'i',f.read(4))[0]
        # wave header
        dtype = struct.unpack(format+'h',f.read(2))[0]
        if dtype == 2:
            dtype = numpy.float32(.0).dtype
        elif dtype == 4:
            dtype = numpy.double(.0).dtype
        else:
            assert False, "Wave is of type '%i', not supported" % dtype
        dtype = dtype.newbyteorder(format)
        
        ignore = f.read(2) # 1 int16
        ignore = f.read(6) # 6 schar, SCHAR = SIGNED CHAR?         ignore = fread(fid,6,'schar'); #
        ignore = f.read(2) # 1 int16
        bname = flatten(struct.unpack(format+'32c',f.read(32)))
        ignore = f.read(4) # 1 int32
        ignore = f.read(4) # 1 int32
        ndims = struct.unpack(format+'4i',f.read(16)) # Number of of items in a dimension -- 0 means no data.
        sfA = struct.unpack(format+'4d',f.read(32))
        sfB = struct.unpack(format+'4d',f.read(32))
        dUnits = flatten(struct.unpack(format+'4c',f.read(4)))
        xUnits = flatten(struct.unpack(format+'16c',f.read(16)))
        fsValid = struct.unpack(format+'h',f.read(2))
        whpad3 = struct.unpack(format+'h',f.read(2))
        ignore = f.read(16) # 2 double
        ignore = f.read(40) # 10 int32
        ignore = f.read(64) # 16 int32
        ignore = f.read(6) # 3 int16
        ignore = f.read(2) # 2 char
        ignore = f.read(4) # 1 int32
        ignore = f.read(4) # 2 int16
        ignore = f.read(4) # 1 int32
        ignore = f.read(8) # 2 int32
        
        wdata = numpy.fromfile(f,dtype)
    else:
        assert False, "Fileversion is of type '%i', not supported" % dtype
        wdata = []
        
    f.close()
   
    return wdata