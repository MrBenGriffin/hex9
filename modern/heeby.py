import geopandas as gpd
from shapely.geometry import LineString
import pyproj
import numpy as np
# import mapstuff
import matplotlib.pyplot as plt
from scipy.special import hyp2f1


# [(-120, 45), (0, 45), (120, 45)]

# geod = pyproj.Geod(ellps='WGS84')
# # trix = np.array([[90., 45.],  [0, 90.],  [0, 0]])
# trix = np.array([(-120, 45), (0, 45), (120, 45)])
#
# grat = gpd.GeoDataFrame(geometry=gpd.points_from_xy(trix[:, 0], trix[:, 1]), crs='EPSG:4326')
# # grat = gpd.GeoSeries(data=trix, crs='EPSG:4326')
# stereograt = grat.to_crs('proj=stere +lat_0=90')
# vx = stereograt.bounds.values
# scale = max(vx.flatten())
# stereograts = stereograt.scale(xfact = 1/scale, yfact=1/scale, origin=(0,0))
# agrats = np.array([l.xy for l in stereograts])
# complexgrats = agrats[:,0] + 1j*agrats[:,1]
# complexgrats = complexgrats.T
# z = complexgrats
# stepb = 4*z**2 / (1 + z**2)**2
# stepb[~np.isfinite(stepb)] = np.inf
# stepb[abs(stepb) > 1E30] = np.inf
# stepb.imag[stepb.imag < 0] = 0
# alpha = 1/3
# beta = 1/3
# stepc = stepb**alpha * hyp2f1(alpha, alpha+beta, alpha+1, stepb)
# stepc.imag = np.where(stepc.imag >= 0, stepc.imag, -stepc.imag)
# x1 = stepc[0,0].real
# stepc[np.isinf(stepb)] = x1 * (1 + 1j*np.sqrt(3))/2
# x = stepc.real
# y = stepc.imag
# xy = np.vstack([x, y]).T

done = True
# >> > coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
# >> > polygon = Polygon(coords)
# >> > polygon.area
grat = mapstuff.graticule(lonrange=[90, 180], latrange=[0, 90])
orthograt = grat.to_crs('proj=ortho +lat_0=45 +lon_0=135')

n=50
fl1 = np.array([np.linspace(45, 225, n), np.zeros(n)]).T
fl2 = np.array([90*np.ones(n), np.linspace(-45, 135, n)]).T
fl3 = np.array([180*np.ones(n), np.linspace(-45, 135, n)]).T
index = fl2[:,1] > 90
fl2[index, 0] -= 180
fl2[index, 1] = 180 - fl2[index, 1]
index = fl3[:,1] > 90
fl3[index, 0] -= 180
fl3[index, 1] = 180 - fl3[index, 1]
fl = [LineString(fl1),
      LineString(fl2),
      LineString(fl3)]
frontlines = gpd.GeoSeries(fl)
frontlines.crs = {'init': 'epsg:4326'}
ofl = frontlines.to_crs('proj=ortho +lat_0=45 +lon_0=135')

fig, ax = plt.subplots(figsize=(2, 2))
ax.axis('off')
ofl.plot(ax=ax, color='grey')
orthograt.plot(ax=ax)
a = 6500000
circle1 = plt.Circle((0, 0), geod.a, facecolor='w', edgecolor='k')
ax.set_xlim(-a, a)
ax.set_ylim(-a, a)
ax.add_patch(circle1)
plt.tight_layout()
fig.savefig('conformaltriangle1.svg', transparent=True)

# Up to now we are just defining our values.
# Use gpdgpd to manage the stereographic projection. 'proj=stere +lat_0=90'

stereograt = grat.to_crs('proj=stere +lat_0=90')
scale = max(stereograt[0].xy[0])
stereograts = stereograt.scale(xfact = 1/scale, yfact=1/scale, origin=(0,0))
agrats = np.array([l.xy for l in stereograts])
complexgrats = agrats[:,0] + 1j*agrats[:,1]
complexgrats = complexgrats.T
fig, ax = plt.subplots(figsize=(2, 2))
ax.axis('equal')
stereograts.plot(ax=ax)
plt.tight_layout()
fig.savefig('conformaltriangle2.svg')#, transparent=True)


z = complexgrats
stepb = 4*z**2 / (1 + z**2)**2
stepb[~np.isfinite(stepb)] = np.inf
stepb[abs(stepb) > 1E30] = np.inf
stepb.imag[stepb.imag < 0] = 0
fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(stepb.real, stepb.imag, color=u'#1f77b4')
ax.axis('equal')
ax.set_xlim(-15,16)
plt.tight_layout()
fig.savefig('conformaltriangle3.svg')#, transparent=True)

alpha = 1/3
beta = 1/3
stepc = stepb**alpha * hyp2f1(alpha, alpha+beta, alpha+1, stepb)
stepc.imag = np.where(stepc.imag >= 0, stepc.imag, -stepc.imag)
x1 = stepc[0,0].real
stepc[np.isinf(stepb)] = x1 * (1 + 1j*np.sqrt(3))/2
fig, ax = plt.subplots(figsize=(2, 2))
plt.tight_layout()
ax.plot(stepc.real, stepc.imag, color=u'#1f77b4')
ax.axis('equal')
fig.savefig('conformaltriangle4.svg')
