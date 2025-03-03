from modern.util import Util
from modern.display import Display

if __name__ == '__main__':
    srd = Util.ball_rnd((0, 0, 0), 50000)
    pts = Util.d3_2(srd)
    Display.show_pts_2d(pts, (-.05, .05), (-0.05, 0.05), 'Random Ball')
