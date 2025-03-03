from modern.util import Util
from modern.display import Display

if __name__ == '__main__':
    tf = Util.tri_eff(50000)
    Display.show_pts_2d(tf, (-.8, .8), (-0.55, 0.9), 'F Triangle')
