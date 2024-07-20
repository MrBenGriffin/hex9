import numpy as np
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import LightSource
from netCDF4 import Dataset

# convert data to rgb array including shading from light source.
# (must specify color map)

if __name__ == '__main__':
    mpl.rcParams['figure.figsize'] = (36, 36)
    mpl.rcParams['axes.xmargin'] = 0.0
    mpl.rcParams['axes.ymargin'] = 0.0
    mpl.rcParams['figure.frameon'] = False
    mpl.rcParams['figure.constrained_layout.use'] = True
    from mpl_toolkits.basemap import Basemap, shiftgrid, cm
    import numpy as np
    import matplotlib.pyplot as plt
    from netCDF4 import Dataset

    # read in etopo5 topography/bathymetry.
    url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
    etopodata = Dataset(url)

    topoin = etopodata.variables['ROSE'][:]
    lons = etopodata.variables['ETOPO05_X'][:]
    lats = etopodata.variables['ETOPO05_Y'][:]
    # shift data so lons go from -180 to 180 instead of 20 to 380.
    topoin, lons = shiftgrid(180., topoin, lons, start=False)

    # plot topography/bathymetry as an image.

    # create the figure and axes instances.
    fig = plt.figure()

    # setup of basemap ('lcc' = lambert conformal conic).
    # use major and minor sphere radii from WGS84 ellipsoid.
    lat, lon = 51.58737, -0.09635
    dx = 10
    #     llcrnrlat=lat-dx, urcrnrlat=lat+dx,
    #     llcrnrlon=lon-dx, urcrnrlon=lon+dx,

    m = Basemap(
            llcrnrlat=lat-dx, urcrnrlat=lat+dx,
            llcrnrlon=lon-dx, urcrnrlon=lon+dx,
            resolution='i'
        )
    # transform to nx x ny regularly spaced 5km native projection grid
    # 1 degree is 111km
    nx = int(8. * dx * 111 / 5.)
    ny = nx

    topodat = m.transform_scalar(topoin, lons, lats, nx, ny)
    # plot image over map with imshow. mpl.colormaps['gist_earth'] cm.GMT_relief
    im = m.imshow(topodat, cm.GMT_haxby)
    # draw coastlines and political boundaries.
    m.drawcoastlines()
    m.drawcountries()
    # m.drawstates()
    # draw parallels and meridians.
    # label on left and bottom of map.
    plt.show()
    #
    # url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
    # etopodata = Dataset(url)
    # topoin = etopodata.variables['ROSE'][:]
    # lons = etopodata.variables['ETOPO05_X'][:]
    # lats = etopodata.variables['ETOPO05_Y'][:]
    # # shift data so lons go from -180 to 180 instead of 20 to 380.
    #
    # mpl.rcParams['figure.figsize'] = (36, 36)
    # mpl.rcParams['axes.xmargin'] = 0.0
    # mpl.rcParams['axes.ymargin'] = 0.0
    # mpl.rcParams['figure.frameon'] = False
    # mpl.rcParams['figure.constrained_layout.use'] = True
    #
    # lat, lon = 51.58737, -0.09635
    # dx = 1.5
    # m = Basemap(  # lat_0=51.58737, lon_0=-0.09635,
    #     llcrnrlat=lat-dx, urcrnrlat=lat+dx,
    #     llcrnrlon=lon-dx, urcrnrlon=lon+dx,
    #     resolution='f')
    #
    # topoin, lons = shiftgrid(180., topoin, lons, start=False)
    # nx = int((m.xmax - m.xmin) / 5000.) + 1
    # ny = int((m.ymax - m.ymin) / 5000.) + 1
    # topodat = m.transform_scalar(topoin, lons, lats, nx, ny)
    # ls = LightSource(azdeg=90, altdeg=20)
    # rgb = ls.shade(topodat, cm.GMT_haxby)
    #
    # im = m.imshow(topodat, cm.GMT_haxby)
    # m.drawcoastlines()
    # # map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
    # # map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),labels=[0,0,0,1])
    # # fill continents 'coral' (with zorder=0), color wet areas 'aqua'
    # # m.drawmapboundary(fill_color='aqua')
    # # m.fillcontinents(color='coral', lake_color='aqua')
    # # shade the night areas, with alpha transparency so the
    # # map shows through. Use current time in UTC.
    # # date = datetime.now()
    # # CS = map.nightshade(date)
    # # plt.title('Day/Night Map for %s (UTC)' % date.strftime("%d %b %Y %H:%M:%S"))
    # plt.show()
