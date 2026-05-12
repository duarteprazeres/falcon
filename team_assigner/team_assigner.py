import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    Assigns players to teams by clustering jersey colours.

    Two-stage approach:
      Stage 1 — per crop: run KMeans(k=2) on the top half of the bounding box
                and use the 4 corners to decide which cluster is background
                (grass/stands) vs jersey.  This is the "corner trick".
      Stage 2 — across crops: collect all per-player jersey colours and run a
                second KMeans(k=2) to split them into Team 1 and Team 2.

    Assignment is PURELY appearance-based — it does NOT use ByteTrack IDs.
    This is critical for Veo cameras: constant zoom-and-pan causes frequent
    ID resets, making any ID-dependent approach (majority voting, etc.) fail.
    """

    def __init__(self, device='cpu'):  # 'device' kept for API compatibility
        self._kmeans: KMeans | None = None

        # BGR display colours (overwritten by auto-detection after fit)
        self.team_colors: dict[int, tuple] = {
            1: (147, 20, 255),   # Magenta / Pink  — default fallback
            2: (255, 191,   0),  # Blue / Gold     — default fallback
        }
        self.is_fitted = False

    # ------------------------------------------------------------------  helpers

    def get_crop(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]

    def _extract_jersey_color(self, crop: np.ndarray) -> np.ndarray | None:
        """
        Extracts the dominant jersey colour from a single player crop.

        Runs KMeans(k=2) on the top half of the crop to separate the player
        from the background.  The 4 corners of the image are almost always
        background (grass, stands), so whichever cluster dominates the corners
        is discarded — the other cluster's centroid is the jersey colour.

        Returns a (3,) BGR array, or None if the crop is too small.
        """
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
            return None

        top_half = crop[: max(1, int(crop.shape[0] * 0.5))]
        pixels = top_half.reshape(-1, 3).astype(np.float64)

        if len(pixels) < 4:
            return None

        km = KMeans(n_clusters=2, random_state=0, n_init=5)
        km.fit(pixels)

        labels_2d = km.labels_.reshape(top_half.shape[0], top_half.shape[1])

        # Corners are almost always background — identify the background cluster
        corners = [
            labels_2d[0,  0],
            labels_2d[0, -1],
            labels_2d[-1, 0],
            labels_2d[-1, -1],
        ]
        background_label = max(set(corners), key=corners.count)
        jersey_label     = 1 - background_label

        return km.cluster_centers_[jersey_label]   # shape (3,) — BGR values

    # ------------------------------------------------------------------ fitting

    def fit_from_crops(self, crops: list) -> None:
        """
        Fits the team-separation model from a sample of player crops.

        Should be called once with crops from the first ~300 frames so both
        teams are well represented.  After fitting, get_player_team() runs on
        any frame without touching the player_id.
        """
        jersey_colors = [self._extract_jersey_color(c) for c in crops]
        jersey_colors = [c for c in jersey_colors if c is not None]

        if len(jersey_colors) < 4:
            print("[TeamAssigner] Not enough valid crops to fit — defaulting all players to team 1.")
            return

        X  = np.array(jersey_colors)           # shape (N, 3)
        km = KMeans(n_clusters=2, random_state=42, n_init=15)
        km.fit(X)
        self._kmeans   = km
        self.is_fitted = True

        # Set display colours to the actual detected jersey colours (BGR)
        for label, center in enumerate(km.cluster_centers_):
            self.team_colors[label + 1] = tuple(int(v) for v in center)

        print(
            f"[TeamAssigner] Fitted on {len(jersey_colors)} crops. "
            f"Team 1 BGR={self.team_colors[1]}, Team 2 BGR={self.team_colors[2]}"
        )

    # ---------------------------------------------------------------- inference

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """
        Predicts which team this player belongs to based on jersey colour.

        Returns 1 or 2.  The player_id argument is accepted for API
        compatibility but is NOT used — predictions are purely pixel-level so
        they stay stable even when ByteTrack resets IDs after a Veo zoom event.
        """
        if not self.is_fitted:
            return 1

        crop  = self.get_crop(frame, player_bbox)
        color = self._extract_jersey_color(crop)
        if color is None:
            return 1

        label = int(self._kmeans.predict(color.reshape(1, -1))[0])
        return label + 1   # KMeans labels 0/1  →  team ids 1/2
