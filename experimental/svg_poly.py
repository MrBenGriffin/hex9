# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

from svgelements import SVG, Polygon
import random

def run():
    # Initialize the SVG canvas
    # You can pass standard SVG attributes like width, height, and viewBox
    canvas = SVG(width=800, height=800, viewBox="0 0 800 800")

    # Define the focal point for cluster
    center_x, center_y = 400, 400
    num_polygons = 25

    for _ in range(num_polygons):
        # Decide if this polygon is a triangle, quadrilateral, or pentagon
        num_points = random.randint(3, 5)

        # Generate random points clustered tightly around the center
        points = []
        for _ in range(num_points):
            x = center_x + random.randint(-150, 150)
            y = center_y + random.randint(-150, 150)
            points.append((x, y))

        # Create a random, semi-transparent color so overlapping shapes look good
        r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
        fill_color = f"rgba({r}, {g}, {b}, 0.5)"

        # 3. Create the Polygon object
        # Unpack the list of point tuples directly into the Polygon constructor.
        # Keyword arguments map directly to SVG presentation attributes.
        poly = Polygon(*points, fill=fill_color, stroke="#111111", stroke_width=2)

        # 4. Append the polygon to the SVG canvas
        canvas.append(poly)

    # 5. Write the final SVG data to a file
    output_file = "polygon_cluster.svg"
    canvas.write_xml(output_file)

    print(f"Successfully generated {output_file}!")


if __name__ == '__main__':
    run()
