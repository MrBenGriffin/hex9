import svg
import math


def is_zero(v):
    return -0.0000001 < v < 0.0000001

def point_inside(clip_window, x, y):
    return clip_window[0] <= x <= clip_window[2] and clip_window[1] <= y <= clip_window[3]


def clip_t(num, denominator, t_e, t_l):
    if is_zero(denominator):
        return num < 0.0, t_e, t_l
    t = num / denominator
    if denominator > 0.0:
        if t > t_l:
            return False, t_e, t_l
        if t > t_e:
            t_e = t
    else:
        if t < t_e:
            return False, t_e, t_l
        if t < t_l:
            t_l = t
    return True, t_e, t_l


def clip_line(left, top, right, bottom, x1, y1, x2, y2):
    clip = (left, top, right, bottom)
    dx = x2 - x1
    dy = y2 - y1
    if is_zero(dx) and is_zero(dy) and point_inside(clip, x1, y1):
        return True, x1, y1, x2, y2
    t_e = 0
    t_l = 1
    cr = clip_t(clip[0] - x1, dx, t_e, t_l)
    t_e = cr[1]
    t_l = cr[2]
    if cr[0]:
        cr = clip_t(x1 - clip[2], -dx, t_e, t_l)
        t_e = cr[1]
        t_l = cr[2]
        if cr[0]:
            cr = clip_t(clip[1] - y1, dy, t_e, t_l)
            t_e = cr[1]
            t_l = cr[2]
            if cr[0]:
                cr = clip_t(y1 - clip[3], -dy, t_e, t_l)
                t_e = cr[1]
                t_l = cr[2]
                if cr[0]:
                    if t_l < 1:
                        x2 = x1 + t_l * dx
                        y2 = y1 + t_l * dy
                    if t_e > 0:
                        x1 = x1 + t_e * dx
                        y1 = y1 + t_e * dy
                    return True, x1, y1, x2, y2
    return False, x1, y1, x2, y2


def rotate_points(pts, theta=0.0):
    # Rotate one or more 2D points counterclockwise by a given angle (in degrees) around a given center.
    theta = theta % 360.0
    ang_rad = math.radians(theta)
    cos_ang, sin_ang = (math.cos(ang_rad), math.sin(ang_rad))
    return [(cx + cos_ang * (x - cx) - sin_ang * (y - cy), cy + sin_ang * (x - cx) + cos_ang * (y - cy)) for x, y in pts]


def add_line(x1, y1, x2, y2, theta=0.0):
    if theta != 0.0:
        (lx3, ly3), (lx4, ly4) = rotate_points(((x1, y1), (x2, y2)), theta=theta)
    else:
        lx3, ly3, lx4, ly4 = x1, y1, x2, y2
    ok, x3, y3, x4, y4 = clip_line(vb_min_x, vb_min_y, vb_max_x, vb_max_y, lx3, ly3, lx4, ly4)
    if ok:
        line = svg.Line(x1=x3, y1=y3, x2=x4, y2=y4, stroke_width=0.1, stroke="black")
        canvas.elements.append(line)


def add_dash_p(x1, y1, x2, y2, theta: float, ev: bool):
    xco = (cx-x1) % a
    ll = a / 3.0
    y_o = h / 3.0
    for j in range(line_count):
        x_o = xco + j*a + a/6 if ev else xco + j*a - a/3
        px3, py3, px4, py4 = x1+x_o, y1 - y_o, x1+x_o+ll, y2 - y_o
        if theta != 0.0:
            (lx3, ly3), (lx4, ly4) = rotate_points(((px3, py3), (px4, py4)), theta=theta)
        else:
            lx3, ly3, lx4, ly4 = px3, py3, px4, py4

        ok, x3, y3, x4, y4 = clip_line(vb_min_x, vb_min_y, vb_max_x, vb_max_y, lx3, ly3, lx4, ly4)
        if ok:
            line = svg.Line(x1=x3, y1=y3, x2=x4, y2=y4, stroke_width=0.1, stroke="black")
            canvas.elements.append(line)


def add_dash_q(x1, y1, x2, y2, theta: float, ev: bool):
    xco = (cx-x1) % a  # need the offset from the edge vs. centre
    ll = a/3.0     # line length is a*1/3
    y_o = 2.0*h/3.0    # y offset is 2/3 for q.
    for j in range(line_count):
        x_o = xco + j*a + 2*ll if ev else xco + j*a - a/6 + ll
        px3, py3, px4, py4 = x1+x_o, y1 - y_o, x1+x_o+ll, y2 - y_o
        if theta != 0.0:
            (lx3, ly3), (lx4, ly4) = rotate_points(((px3, py3), (px4, py4)), theta=theta)
        else:
            lx3, ly3, lx4, ly4 = px3, py3, px4, py4

        ok, x3, y3, x4, y4 = clip_line(vb_min_x, vb_min_y, vb_max_x, vb_max_y, lx3, ly3, lx4, ly4)
        if ok:
            line = svg.Line(x1=x3, y1=y3, x2=x4, y2=y4, stroke_width=0.1, stroke="black")
            canvas.elements.append(line)


def add_dashes(x1, y1, x2, y2, even: bool):
    add_dash_p(x1, y1, x2, y2, 0.0, even)
    add_dash_q(x1, y1, x2, y2, 0.0, even)
    add_dash_p(x1, y1, x2, y2, 120.0, even)
    add_dash_q(x1, y1, x2, y2, 120.0, even)
    add_dash_p(x1, y1, x2, y2, 240.0, even)
    add_dash_q(x1, y1, x2, y2, 240.0, even)


def add_lines(x1, y1, x2, y2):
    add_line(x1, y1, x2, y2, theta=0.0)
    add_line(x1, y1, x2, y2, theta=120.0)
    add_line(x1, y1, x2, y2, theta=240.0)


if __name__ == '__main__':
    # h = (√3)/2a where a=length of triangle.
    # we want to make sure that 'a' is the unit of measure, not 'h'
    # need to calculate h from a; and a should be good for division by 3
    g = 8
    a = 9.0
    h = math.sqrt(3.0)/2.0 * a
    vb_min_x, vb_min_y, vb_max_x, vb_max_y = 0, 0, math.pow(3, g), math.pow(3, g)
    ext = math.pow(3, g+2)
    min_x, min_y, max_x, max_y = vb_min_x-ext, vb_min_y-ext, vb_max_x+ext, vb_max_y+ext
    cx, cy = (vb_max_x-vb_min_x)/2.0, (vb_max_y-vb_min_y)/2.0  # this is the centre of the map.
    canvas = svg.SVG(
        # ViewBoxSpec are the coordinates used in the world of the drawing
        # width/height are the size of the rendered svg.
        viewBox=svg.ViewBoxSpec(vb_min_x, vb_min_y, vb_max_x, vb_max_y),
        width=math.pow(3, 7),
        height=math.pow(3, 7),
        elements=[]
    )
    line_count = int(0.5 * (max_y-min_y) / h)

    for i in range(line_count):
        add_lines(min_x, cy+i*h, max_x, cy+i*h)
        add_dashes(min_x, cy + i * h, max_x, cy + i * h, i % 2 == 0)
        add_lines(min_x, cy-(i+1)*h, max_x, cy-(i+1)*h,)
        add_dashes(min_x, cy-(i+1)*h, max_x, cy-(i+1)*h, i % 2 == 1)

    f = open(f"grid_{int(g)}.svg", "w")
    f.write(canvas.as_str())
    f.close()
