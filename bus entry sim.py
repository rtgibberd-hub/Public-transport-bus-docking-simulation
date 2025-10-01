import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter


#ctrl + / over highlighted section of code to uncomment it
#winget install "FFmpeg (Essentials Build)"

#bus parameters


#ALEXANDER-DENNIS
# bus_colour= "#8b0000"
# bus_width= 2.54
# bus_length= 10.383
# front_overhang= 2.823
# rear_overhang= 2.12
# title ="Bus manouevre - max overlap - ALEXANDER-DENNIS"
# file_save_name= "ALEXANDER-DENNIS_bus_entry.mp4"
# global_path =  {'ALEXANDER-DENNIS_bus_entry.mp4':r'C:\Users\rtgib\OneDrive\Documents\mummy simulation bus project\ALEXANDER-DENNIS_bus_entry.mp4'}

#MAN LIONS CITY 10E
# bus_colour= "white"
# bus_width= 2.55
# bus_length= 10.575
# front_overhang= 2.775
# rear_overhang= 3.405
# title = "Bus manouevre - max overlap - MAN LIONS CITY 10E"
# file_save_name= "MAN_LIONS_CITY_10E_bus_entry.mp4"
# global_path =  {'MAN_LIONS_CITY_10E_bus_entry.mp4':r'C:\Users\rtgib\OneDrive\Documents\mummy simulation bus project\MAN_LIONS_CITY_10E_bus_entry.mp4'}

#MAN LIONS CITY 12
# bus_colour= "#ADD8E6"
# bus_width= 2.55
# bus_length= 12.185
# front_overhang= 2.775
# rear_overhang= 3.405
# title = "Bus manouevre - max overlap - MAN LIONS CITY 12"
# file_save_name= "MAN_LIONS_CITY_12_bus_entry.mp4"
# global_path =  {'MAN_LIONS_CITY_12_bus_entry.mp4':r'C:\Users\rtgib\OneDrive\Documents\mummy simulation bus project\MAN_LIONS_CITY_12_bus_entry.mp4'}

#MAN LIONS CITY 12E
bus_colour= "#90EE90"
bus_width= 2.55
bus_length= 12.2
front_overhang= 2.775
rear_overhang= 3.405
title = "Bus manouevre - max overlap - MAN LIONS CITY 12E"
file_save_name= "MAN_LIONS_CITY_12E_bus_entry.mp4"
global_path =  {'MAN_LIONS_CITY_12E_bus_entry.mp4':r'C:\Users\rtgib\OneDrive\Documents\mummy simulation bus project\MAN_LIONS_CITY_12E_bus_entry.mp4'}

#MAN LIONS CITY 12G
# bus_colour= "#D3D3D3"
# bus_width= 2.55
# bus_length= 12.185
# front_overhang= 2.775
# rear_overhang= 3.405
# title = "Bus manouevre - max overlap - MAN LIONS CITY 12G"
# file_save_name= "MAN_LIONS_CITY_12G_bus_entry.mp4"
# global_path =  {'MAN_LIONS_CITY_12G_bus_entry.mp4':r'C:\Users\rtgib\OneDrive\Documents\mummy simulation bus project\MAN_LIONS_CITY_12G_bus_entry.mp4'}


# Coordinates dictionary
coords = {
    "bus_stop": {
        "starting_pt": [-2-4.357447380663206, 0],
        "contact_pt": [0, 0],
        "end_pt": [20-4.357447380663206, 0],
    },
    "guiding_track": {
        "start": [-9.848-4.357447380663206, -4.986],
        "corner_pt": [0.79-4.357447380663206, -3.25],
        "end_pt": [10, -3.25],
        "actual_end": [29.406, -8.088]
    }
}

show_wheel_markers = False

# Headings and distances
steps = 50
heading_1 = 10
heading_2 = 0
heading_3 = -14
heading_1_rad = np.radians(heading_1)
heading_2_rad = np.radians(heading_2)
heading_3_rad = np.radians(heading_3)
distance_1 = 20 - front_overhang
distance_2 = 20
distance_3 = 20

# Vectors
origin = np.array([0, 0])
dir_vec_1 = np.array([np.cos(heading_1_rad), np.sin(heading_1_rad)])
dir_vec_2 = np.array([np.cos(heading_2_rad), np.sin(heading_2_rad)])
dir_vec_3 = np.array([np.cos(heading_3_rad), np.sin(heading_3_rad)])

pt_1 = origin  - dir_vec_1 * (distance_1 )
pt_2 = origin
pt_3 = origin + dir_vec_2 * distance_2
pt_4 = pt_3 + dir_vec_3 * distance_3


#turning amounts
weights = 0.5 * np.cos (np.linspace(0, np.pi, steps)) + 0.5
weights /= weights.sum()

rotation_changes_1 = weights * heading_1_rad
rotation_changes_3 = weights * heading_3_rad

# Geometry functions
def compute_bus_geometry(centre, vec_bus):
    perp_vec = np.array([-vec_bus[1], vec_bus[0]])
    fl = centre + perp_vec * (bus_width / 2) + vec_bus * (bus_length / 2)
    fr = centre - perp_vec * (bus_width / 2) + vec_bus * (bus_length / 2)
    bl = centre + perp_vec * (bus_width / 2) - vec_bus * (bus_length / 2)
    br = centre - perp_vec * (bus_width / 2) - vec_bus * (bus_length / 2)
    flw = centre + perp_vec * (bus_width / 2) + vec_bus * ((bus_length / 2) - front_overhang)
    blw = centre + perp_vec * (bus_width / 2) - vec_bus * ((bus_length / 2) - rear_overhang)
    bus_poly = np.array([fl, fr, br, bl, blw, flw, fl])
    return bus_poly, flw, blw, fl

def move(pt, heading_rad, distance):
    return pt + np.array([distance * np.cos(heading_rad), distance * np.sin(heading_rad)])

def vector_rotate(vector, theta):
    matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    return matrix @ vector

def angle_finder(v1, v2):
    dot = np.dot(v1, v2)
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return np.abs(np.arctan2(det, dot))

def distance_calc(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.linalg.norm(pt2 - pt1)


# Initialize animation state
centre = pt_1.copy()
vec_bus = dir_vec_1.copy()
bus_poly, flw, blw, fl = compute_bus_geometry(centre, vec_bus)

# Precompute frames
frames = []

# Stage 1: Straight move
while flw[1] < 0:
    centre = move(centre, heading_1_rad, (distance_1) / steps)
    bus_poly, flw, blw, fl = compute_bus_geometry(centre, vec_bus)
    frames.append(bus_poly)

#print (f"the distance from the centre to y=0 after stage 1 is:{np.abs(centre[1])}")
# Stage 2: Weighted rotation from heading_1 to heading_2



pivot_1 = flw.copy()  # This point stays fixed on the curb

# Step 2: Rotate around the anchor
for rot in rotation_changes_1:
    # Rotate vector around anchor
    vec_bus = vector_rotate(vec_bus, rot)
    pivot_1 = move(flw, heading_2_rad, distance_2/steps)

    # Recompute centre so that flw stays at anchor
    # This assumes compute_bus_geometry returns flw based on centre and vec_bus
    temp_centre = centre.copy()
    temp_bus_poly, temp_flw, _, _ = compute_bus_geometry(temp_centre, vec_bus)
    offset = pivot_1 - temp_flw

    centre += offset  # Shift centre so flw stays anchored
    # Final geometry
    bus_poly, flw, blw, fl = compute_bus_geometry(centre, vec_bus)
    frames.append(bus_poly)





#for rot in rotation_changes_1:
  #  centre = move(centre, np.pi/2, np.abs(flw[1]))
  #  vec_bus = vector_rotate(vec_bus, rot)
   # centre = move(centre, heading_2_rad, distance_2 / steps)
  #  bus_poly, flw, blw = compute_bus_geometry(centre, vec_bus)
  #  frames.append(bus_poly)
    
#print(f"distance between front of bus and end of bus stop is:{np.linalg.norm(np.array(coords['bus_stop']['end_pt']) - (np.array(flw) + front_overhang*dir_vec_2))}")

swept_pts = []

for frame in frames:
    fl = frame[0]  # front-left corner
    if fl[1] > 0:  # below bus stop line
        swept_pts.append(fl)

# Optionally close the polygon
if swept_pts:
    swept_polygon = np.array(swept_pts + [swept_pts[-1] + [0, 0.1], swept_pts[0] + [0, 0.1]])

print(f"Swept points collected: {len(swept_pts)}")
# Pause after Stage 2
for _ in range(int(2000 / 50)):
    frames.append(bus_poly.copy())

# Stage 3: Weighted rotation from heading_2 to heading_3
for rot in rotation_changes_3:
    vec_bus = vector_rotate(vec_bus, -rot)
    centre = move(centre, heading_3_rad, distance_3 / steps)
    bus_poly, flw, blw, fl = compute_bus_geometry(centre, vec_bus)
    frames.append(bus_poly)


    
# Animation setup
fig, ax = plt.subplots(figsize=(8, 6))
if swept_pts:
    swept_patch = Polygon(swept_polygon, closed=True, color='red', alpha=0.4, label='Swept Area', zorder = 5)
    ax.add_patch(swept_patch)
ax.legend()
swept_pts_array = np.array(swept_pts)
max_dist_into_pavement = np.max(swept_pts_array[:, 1])
range_of_overlap =np.max(swept_pts_array[:, 0]) - np.min(swept_pts_array[:, 0])

ax.text(x= -20, y= 10, s = f"Max distance into the pavement is:{round(max_dist_into_pavement, 2)}m\nThe range of the overlap is:{range_of_overlap:.2f}", fontsize = 6, color = 'black', bbox = dict(facecolor = 'white', edgecolor = 'black', boxstyle = 'round, pad = 0.3'))
#width of overlap
ax.set_xlim(-20, 30)
ax.set_ylim(-10, 2)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title(title)
bus_patch = Polygon(frames[0], closed=True, facecolor=bus_colour, edgecolor='black', zorder = 9)
ax.add_patch(bus_patch)
# Plot static elements
bs = coords["bus_stop"]
gt = coords["guiding_track"]
ax.plot(*zip(bs["starting_pt"], bs["contact_pt"], bs["end_pt"]), 'k--', label='Bus Stop')
ax.plot(*zip(gt["start"], gt["corner_pt"], gt["end_pt"], gt["actual_end"]), 'b--', label='Guiding Track')
ax.axhline(y = -6, color = 'white', linestyle = '--', linewidth = 1.5, zorder = 2)

if show_wheel_markers:
    flw_marker, = ax.plot([], [], 'o', color='orange', label='FLW')
    blw_marker, = ax.plot([], [], 'o', color='purple', label='BLW')

# Define the rectangle
waiting_area = Rectangle(
    (-2-4.357447380663206, 0),            # Bottom-left corner (x, y)
    22,                 # Width: 30 - (-2)
    2,                  # Height: 2 - 0
    facecolor='black',
    alpha=0.2, 
    zorder = 1
)

# Add it to the plot
ax.add_patch(waiting_area)
outer_area_rhs = Rectangle(
    (-20-4.35744738066320, 0), 
    18, 
    2, 
    facecolor = 'green', 
    alpha = 0.7,
    zorder = 1

)

outer_area_lhs = Rectangle(
    (20-4.35744738066320, 0),
    20, 
    2, 
    facecolor = 'green', 
    alpha = 0.7,
    zorder = 1
)

ax.add_patch(outer_area_rhs)
ax.add_patch (outer_area_lhs)

road = Rectangle(
    (-20, -10), 
    50, 
    10,
    facecolor = 'black',
    zorder = 1

)
ax.add_patch(road)
ax.grid(False)


def update(frame):
    bus_patch.set_xy(frame)
    artists = [bus_patch]

    if show_wheel_markers:
        flw_marker.set_data([frame[5][0]], [frame[5][1]])
        blw_marker.set_data([frame[4][0]], [frame[4][1]])
        artists.extend([flw_marker, blw_marker])

    return tuple(artists)

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
writer = FFMpegWriter(fps=6, metadata=dict(artist='Raphael'), bitrate=1800)
#ani.save(global_path.get(file_save_name), writer=writer, dpi=150)
plt.show()