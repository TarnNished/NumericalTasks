import cv2
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = "highway.mp4"
FRAME_DIFF_THRESHOLD = 30
MIN_CONTOUR_AREA = 200
MAX_TRACK_DISTANCE = 50
MAX_MISSED_FRAMES = 10


class Track:
    def __init__(self, track_id, centroid):
        self.id = track_id
        self.centroids = [centroid]
        self.missed = 0

    @property
    def last_centroid(self):
        return self.centroids[-1]

    def add(self, c):
        self.centroids.append(c)
        self.missed = 0

    def miss(self):
        self.missed += 1


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def kmeans_from_scratch(X, k, n_iters=20):
    """
    Simple k-means implementation.
    X: (n_samples, n_features)
    """
    n_samples, n_features = X.shape
    # randomly pick initial centers
    rng = np.random.default_rng(0)
    indices = rng.choice(n_samples, size=k, replace=False)
    centers = X[indices]

    for _ in range(n_iters):
        # assign step
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n_samples, k)
        labels = np.argmin(dists, axis=1)

        # update step
        new_centers = np.zeros_like(centers)
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = np.mean(cluster_points, axis=0)
            else:
                # if a cluster is empty, keep the old center
                new_centers[j] = centers[j]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def extract_tracks_with_frame_diff(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Empty video")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    tracks = []
    next_track_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        prev_gray = gray

        # Threshold on frame difference
        _, thresh = cv2.threshold(diff, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # simple morphology
        kernel = np.ones((3, 3), np.uint8)
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

        # tracking same as before, but we keep it here so it's clearly "our logic"
        unmatched = centroids.copy()
        for t in tracks:
            if not unmatched:
                t.miss()
                continue
            dists = [distance(t.last_centroid, c) for c in unmatched]
            idx = int(np.argmin(dists))
            if dists[idx] < MAX_TRACK_DISTANCE:
                t.add(unmatched[idx])
                unmatched.pop(idx)
            else:
                t.miss()

        for c in unmatched:
            tracks.append(Track(next_track_id, c))
            next_track_id += 1

        tracks = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]

    cap.release()
    return tracks, dt


def build_simple_features(tracks, dt):
    """
    Build a very basic feature: [mean_y_speed] for each track
    """
    features = []
    valid_tracks = []
    for t in tracks:
        if len(t.centroids) < 5:
            continue
        ys = np.array([c[1] for c in t.centroids], dtype=float)
        vs = np.gradient(ys, dt)
        feat = np.array([np.mean(vs)], dtype=float)
        features.append(feat)
        valid_tracks.append(t)
    return np.vstack(features), valid_tracks


def main():
    print("From-scratch frame-diff tracking...")
    tracks, dt = extract_tracks_with_frame_diff(VIDEO_PATH)
    print(f"Tracks found: {len(tracks)}")

    X, valid_tracks = build_simple_features(tracks, dt)
    if len(valid_tracks) == 0:
        print("No valid tracks to cluster.")
        return

    k = min(3, len(valid_tracks))
    labels, centers = kmeans_from_scratch(X, k=k)
    print("Cluster centers (mean_y_velocity):", centers.flatten())

    # simple visualization
    plt.figure()
    plt.scatter(np.arange(len(valid_tracks)), X[:, 0], c=labels)
    plt.xlabel("Track index")
    plt.ylabel("Mean vertical velocity")
    plt.title("From-scratch k-means clustering")
    plt.show()


if __name__ == "__main__":
    main()
