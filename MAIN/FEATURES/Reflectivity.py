import numpy as np


def Reflectivity(data):
    center, radi = (509, 546), 55
    y,x = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[1])**2+(y-center[0])**2)
    ind = np.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] + csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return np.mean(radialprofile)
