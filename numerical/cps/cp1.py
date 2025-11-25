import cv2
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ---------- CONFIG ----------
VIDEO_PATH = "highway.mp4"  # your downloaded video
MAX_TRACK_DISTANCE = 50      # pixels – max distance to match same car between frames
MAX_MISSED_FRAMES = 10       # how many frames we allow a track to be missing
MIN_CONTOUR_AREA = 200       # filter out tiny blobs (noise)
ROI_SINGLE_OBJECT = (640, 360)  # approximate point near the car you want to treat as "single object"
SAVGOL_WINDOW = 11           # smoothing window (must be odd and < len(time series))
SAVGOL_POLY = 3


# ---------- TRACK STRUCTURE ----------
class Track:
    def __init__(self, track_id, centroid):
        self.id = track_id
        self.centroids = [centroid]
        self.missed = 0

    @property
    def last_centroid(self):
        return self.centroids[-1]

    def add(self, centroid):
        self.centroids.append(centroid)
        self.missed = 0

    def mark_missed(self):
        self.missed += 1


# ---------- DERIVATIVE HELPERS ----------
def smooth_series(y, window=SAVGOL_WINDOW, poly=SAVGOL_POLY):
    y = np.array(y, dtype=float)
    if len(y) < window:
        # too short for Savitzky–Golay, just return original
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)


def compute_derivatives(s, dt):
    """
    s: 1D array of position over time (pixels or meters)
    returns: v, a, j, jounce
    """
    s = np.array(s, dtype=float)
    s_smooth = smooth_series(s)
    v = np.gradient(s_smooth, dt)
    a = np.gradient(v, dt)
    j = np.gradient(a, dt)
    jo = np.gradient(j, dt)
    return s_smooth, v, a, j, jo


# ---------- GEOMETRY HELPERS ----------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def project_along_road(centroids):
    """
    For simplicity we just use the y-coordinate (cars moving mostly vertical in this video).
    If you want, you can rotate coordinates to align with road.
    """
    return np.array([c[1] for c in centroids], dtype=float)


# ---------- MAIN VIDEO PROCESSING ----------
def extract_tracks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    tracks = []
    next_track_id = 0

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = back_sub.apply(gray)

        # clean mask
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))

        # Associate centroids to existing tracks by nearest neighbor
        unmatched_centroids = centroids.copy()

        for track in tracks:
            if not unmatched_centroids:
                track.mark_missed()
                continue

            # find closest centroid to this track's last centroid
            dists = [distance(track.last_centroid, c) for c in unmatched_centroids]
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < MAX_TRACK_DISTANCE:
                track.add(unmatched_centroids[min_idx])
                unmatched_centroids.pop(min_idx)
            else:
                track.mark_missed()

        # create new tracks for remaining centroids
        for c in unmatched_centroids:
            t = Track(next_track_id, c)
            next_track_id += 1
            tracks.append(t)

        # drop dead tracks
        tracks = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]

    cap.release()
    return tracks, dt


# ---------- SINGLE OBJECT SELECTION ----------
def pick_single_object_track(tracks, roi_point=ROI_SINGLE_OBJECT):
    """
    Choose the track whose average centroid is closest to the ROI.
    """
    best_track = None
    best_dist = float("inf")
    for t in tracks:
        if len(t.centroids) < 5:
            continue
        mean_c = np.mean(np.array(t.centroids), axis=0)
        d = distance(mean_c, roi_point)
        if d < best_dist:
            best_dist = d
            best_track = t
    return best_track


# ---------- CLUSTERING ----------
def build_feature_vector(track, dt):
    """
    For each track, build a feature vector:
    [mean_velocity, std_velocity, mean_accel, std_accel]
    """
    if len(track.centroids) < 5:
        return None

    s = project_along_road(track.centroids)
    s_smooth, v, a, j, jo = compute_derivatives(s, dt)

    feat = np.array([
        np.mean(v),
        np.std(v),
        np.mean(a),
        np.std(a)
    ], dtype=float)
    return feat


def cluster_tracks(tracks, dt, n_clusters=3):
    features = []
    valid_tracks = []

    for t in tracks:
        feat = build_feature_vector(t, dt)
        if feat is not None:
            features.append(feat)
            valid_tracks.append(t)

    if len(features) < n_clusters:
        n_clusters = max(1, len(features))

    if len(features) == 0:
        return [], []

    X = np.vstack(features)

    # You can experiment with scaling or different norms here.
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X)
    return valid_tracks, labels


# ---------- PLOTS ----------
def plot_single_track_kinematics(track, dt):
    centroids = np.array(track.centroids)
    s = project_along_road(centroids)
    t = np.arange(len(s)) * dt

    # ---- Compute derivatives in pixels ----
    s_smooth, v, a, j, jo = compute_derivatives(s, dt)

    # ---- Pixel → meter conversion ----
    LANE_WIDTH_REAL = 3.7        # meters (standard highway lane width)
    LANE_WIDTH_PIXELS = 120      # measure manually from frame
    scale = LANE_WIDTH_REAL / LANE_WIDTH_PIXELS

    s_meters = s_smooth * scale
    v_meters = v * scale
    a_meters = a * scale
    j_meters = j * scale
    jo_meters = jo * scale

    # ---- Plot kinematics in meters ----
    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

    axs[0].plot(t, s_meters)
    axs[0].set_ylabel("Position (m)")

    axs[1].plot(t, v_meters)
    axs[1].set_ylabel("Velocity (m/s)")

    axs[2].plot(t, a_meters)
    axs[2].set_ylabel("Acceleration (m/s²)")

    axs[3].plot(t, j_meters)
    axs[3].set_ylabel("Jerk (m/s³)")

    axs[4].plot(t, jo_meters)
    axs[4].set_ylabel("Jounce (m/s⁴)")
    axs[4].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def main():
    print("Extracting tracks...")
    tracks, dt = extract_tracks_from_video(VIDEO_PATH)
    print(f"Total tracks detected: {len(tracks)}")

    # Single object analysis
    single = pick_single_object_track(tracks, ROI_SINGLE_OBJECT)
    if single is None:
        print("No suitable single-object track found.")
    else:
        print(f"Chosen single-object track ID: {single.id}, length={len(single.centroids)}")
        plot_single_track_kinematics(single, dt)

    # Multi-object clustering
    valid_tracks, labels = cluster_tracks(tracks, dt, n_clusters=3)
    print(f"Clustered {len(valid_tracks)} tracks into {len(set(labels))} clusters.")

    # Simple visualization: scatter mean velocity vs mean accel colored by cluster
    feats = [build_feature_vector(t, dt) for t in valid_tracks]
    X = np.vstack(feats)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 2], c=labels)
    plt.xlabel("Mean velocity")
    plt.ylabel("Mean acceleration")
    plt.title("Track clustering (v_mean vs a_mean)")
    plt.show()


if __name__ == "__main__":
    main()
