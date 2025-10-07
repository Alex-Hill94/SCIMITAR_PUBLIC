from SCIMITAR import *

# Panel layout: 'P_arrangement' activates left, right, top, and bottom, and centre panels (4 of 9)
activate = P_arrangement

# Define intersection planes: detector surface (z=0) plus mid-height plane for 3D overlap analysis
intersection_heights = detector_and_mid

# Initialise and build scanner geometry
S = Scimitar()
S.Scene(
    grid = 4,                           # 4×4 grid = 16 emitters per panel
    SID = 0.976,                         # Source-to-image distance: 97.6 cm (panel to detector)
    activate = activate,                # Active panel configuration (cross pattern)
    cone_angle = 31.9,                    # X-ray cone full opening angle: 31.9 degrees
    panel_theta = 12.5,                   # Panel tilt angle relative to horizontal: 12.5 degrees
    emitter_pitch = 1.0e-2,            # Spacing between adjacent emitters on each panel: 1.0 cm
    cone_resolution = 4,                # Polygonal approximation: 4 vertices = square cone cross-section
    panel_pitch = 22.9e-2,               # Spacing between adjacent panel centers: 22.9 cm
    intersection_heights = intersection_heights,  # Planes for geometric analysis
    panel_setting = 'standard grid',    # Layout mode: 'standard grid', 'staggered', or 'custom'
    stagger = 0.0,                      # Vertical offset for center panel (only used if panel_setting='staggered'): 0.0 cm
    panel_positions = None,             # Manual panel coordinates (only used if panel_setting='custom')
    force_vis = False,                  # If True, visualise even if geometry checks fail
    silence_tqdm = True                 # Suppress progress bars during construction
        )

# Check that X-ray projection stays within acceptable bounds of a 43 cm detector
S.CheckStray(d_width = 0.43)

# Generate 2D pixel-wise irradiation distributions at all intersection heights
S.Irradiation(pix_per_cm = 2)  # Detector resolution: 2 pixels per cm

# Plot irradiation map at detector plane (z=0)
S.PlotMaps(
    plot_level = 'detector only',       # Options: 'detector only', 'all', or None
    plot_lims = [-0.40, 0.40],          # Display region: ±40 cm square
    d_width = 0.43,                     # Overlay 43 cm detector boundary
    save_plot = False,                  # Display interactively rather than saving to file
    plot_filename = 'dummy.png',        # Output filename (ignored if save_plot=False)
    plot_title = 'Example Figure'                # Custom plot title
        )

# Compute coverage, overlap, angular range, and stray radiation metrics
S.Metrics(d_width = 0.43)

# Launch interactive VTK viewer with panels, emitters, cones, and patient model
S.Visualise(include_patient = True)

# Extract performance metrics dataframe
df = S.df
print(df.T)  # Transpose for readable column-wise display