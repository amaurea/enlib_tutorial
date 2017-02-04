import numpy as np
from enlib import enmap, utils, powspec
from matplotlib.pyplot import *

deg = utils.degree
# Build an empty map from scratch (usually you don't
# have to do this, as you'll be reading in an existing map)
shape, wcs = enmap.geometry(pos=[[-5*deg,-5*deg],[5*deg,5*deg]], res=0.5*utils.arcmin, proj="car")
map = enmap.zeros(shape, wcs)
print "Made a map with shape %s and wcs %s" % (str(shape),str(wcs))

# Let's set some columns to 1 and some rows to 2
map[:,100:200] = 1
map[300:400,:] = 2

# And display it
print "Displaying map with columns of 1 and rows of 2"
matshow(map); show()

# Query the location of each pixels. The result will be a
# new enmap with two fields. The first is dec and the second
# is ra.
pos = map.posmap()
print "Displaying the dec of every pixel"
matshow(pos[0]); show()
print "Displaying the ra  of every pixel"
matshow(pos[1]); show()

# Let's try to make a map with a CMB power spectrum. We first
# read in a spectrum, and then make a random realization using it.
ps = powspec.read_spectrum("cl_lensed.dat")
map = enmap.rand_map(shape, wcs, ps)

# The result is a random T map
print "Displaying generated T map"
matshow(map); show()

# We didn't get Q and U because we didn't specify a 3-component
# shape. Let's add this to the shape:
shape = (3,)+shape

# When we do a realization now we will get out a (3,ny,nx)-shaped
# map.
map = enmap.rand_map(shape, wcs, ps)
print "Displaying full T,Q,U map"
for m in map: matshow(m)
show()

# Let's write this map to disk
print "Writing to test.fits"
enmap.write_map("test.fits", map)

# And read it back in
print "Reading back in from test.fits"
map = enmap.read_map("test.fits")

# Enmaps are derived from numpy arrays, so you can use them
# with all numpy array operations. For example, we can
# get the absolute value
absmap = np.abs(map)
print "Displaying absolute value of T map"
matshow(absmap[0]); show()

polamp = np.sum(map[1:]**2,0)**0.5
print "Displaying polarization amplitude map"
matshow(polamp); show()

# We can convert the map from T,Q,U to T,E,B
teb = enmap.ifft(enmap.map2harm(map)).real
print "Displaying T,E,B maps"
for m in teb: matshow(m)
show()

# Let's apply a lowpass filter. We could do that
# simply by using enmap.smooth_gauss, but we will
# do it manually to show how it's done. The lowpass
# filter is defined in fourier space. map2harm takes
# you from real-space TQU to fourier-space TEB.
# It's the flat-sky equivalent of map2alm.
# The result is ({t,e,b},nly,nlx).
fmap = enmap.map2harm(map)
# Get the 2d fourier mode (ly,lx) corresponding to
# each fourier-space pixel
lmap = fmap.lmap()
# I only want the 1d multipole (this is just pythagoras)
l    = np.sum(lmap**2,0)**0.5
# Apply a gaussian filter with standard deviation of l=1000
fmap *= np.exp(-0.5*(l/1000)**2)
# And transform back
smooth = enmap.harm2map(fmap)
print "Displaying smoothed map"
for m in smooth: matshow(m)
show()

# Let's try to project to a different coordinate system.
# Our example map was in CAR. Let's try switching to CEA
# with the same fiducial resolution.
shape2, wcs2 = enmap.geometry(pos=map.box(), res=0.5*utils.arcmin, proj="cea")
map_cea = map.project(shape2, wcs2)
print "Displaying car anc cea T maps. On a map this small the difference isn't really visible"
matshow(map[0]); matshow(map_cea[0]); show()

# We can also use this interpolation to zoom. Let's
# create a very high res coordinate system in a small
# area.
shape3, wcs3 = enmap.geometry(pos=[[-0.2*deg,-0.2*deg],[0.2*deg,0.2*deg]],res=0.05*utils.arcmin)
map_zoom = map.project(shape3, wcs3)
print "Displaying zoomed map"
matshow(map_zoom[0]); show()

# The interpolation uses 3rd order spline interpolation by default.
# If we want to see how the old pixels map to the new ones we can
# use 0th order interpolation (nearest neighbor). This is very fast,
# but not what you usually want to do.
map_zoom_nn = map.project(shape3, wcs3, order=0)
print "Displaying 0th order zoomed map"
matshow(map_zoom_nn[0]); show()
