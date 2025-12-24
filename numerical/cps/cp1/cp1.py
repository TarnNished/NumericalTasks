import cv2
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ---------- CONFIG ----------
VIDEO_PATH = "highway.mp4"
OUTPUT_VIDEO_PATH = "tracked_highway.mp4"
MAX_TRACK_DISTANCE = 50      # pixels – max distance to match same car between frames
MAX_MISSED_FRAMES = 10       # how many frames we allow a track to be missing
MIN_CONTOUR_AREA = 200       # filter out tiny blobs (noise)
SAVGOL_WINDOW = 11           # smoothing window (must be odd and < len(time series))
SAVGOL_POLY = 3


ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 100, 200, 1000, 500  # Define the bounds of the highway road


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
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)


def compute_derivatives(s, dt):
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


def project_along_road(centroids, p0, p1):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    d = p1 - p0
    d = d / np.linalg.norm(d)  # unit direction vector

    s = []
    for c in centroids:
        c = np.array(c, dtype=float)
        s.append(np.dot(c - p0, d))  # projection length
    return np.array(s, dtype=float)


# ---------- ROI Helper ----------
def is_inside_roi(centroid, roi_x1, roi_y1, roi_x2, roi_y2):
    """
    Checks if the centroid (x, y) is within the defined ROI.
    """
    x, y = centroid
    return roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2


# ---------- MAIN VIDEO PROCESSING ----------
def extract_tracks_from_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

            # Check if centroid is inside the defined ROI
            if is_inside_roi((cx, cy), ROI_X1, ROI_Y1, ROI_X2, ROI_Y2):
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

        # Draw bounding boxes for each track
        for track in tracks:
            for centroid in track.centroids:
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            x, y = track.centroids[-1]
            cv2.putText(frame, f"ID:{track.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show or save the frame with bounding boxes
        if frame_idx % 10 == 0:  # Reduce update frequency
            cv2.imshow("Tracked Video", frame)
            cv2.waitKey(1)

        if out is not None:
            out.write(frame)

    cap.release()
    if out is not None:
        out.release()
    return tracks, fps


# ---------- SINGLE OBJECT SELECTION ----------
def pick_single_object_track(tracks):
    """
    Choose the longest track with centroids.
    """
    best_track = max(tracks, key=lambda t: len(t.centroids), default=None)
    return best_track


# ---------- CLUSTERING ----------
def build_feature_vector(track, dt):
    if len(track.centroids) < 5:
        return None

    s = project_along_road(track.centroids, (0, 0), (1, 1))  # Example points for road direction
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    return valid_tracks, labels


# ---------- PLOTS ----------
def plot_single_track_kinematics(track, dt):
    centroids = np.array(track.centroids)
    s = project_along_road(centroids, (0, 0), (1, 1))  # Same road projection
    t = np.arange(len(s)) * dt

    # ---- Compute derivatives in pixels ----
    s_smooth, v, a, j, jo = compute_derivatives(s, dt)

    # ---- Plot kinematics ----
    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

    axs[0].plot(t, s_smooth)
    axs[0].set_ylabel("Position (pixels)")

    axs[1].plot(t, v)
    axs[1].set_ylabel("Velocity (pixels/s)")

    axs[2].plot(t, a)
    axs[2].set_ylabel("Acceleration (pixels/s²)")

    axs[3].plot(t, j)
    axs[3].set_ylabel("Jerk (pixels/s³)")

    axs[4].plot(t, jo)
    axs[4].set_ylabel("Jounce (pixels/s⁴)")
    axs[4].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


# ---------- MAIN FUNCTION ----------
def main():
    print("Extracting tracks...")
    tracks, fps = extract_tracks_from_video(VIDEO_PATH, OUTPUT_VIDEO_PATH)

    # Pick one car to analyze (choose longest track)
    track_to_analyze = pick_single_object_track(tracks)
    if track_to_analyze is None:
        print("No track to analyze.")
    else:
        print(f"Selected track ID {track_to_analyze.id}")
        plot_single_track_kinematics(track_to_analyze, 1.0 / fps)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
