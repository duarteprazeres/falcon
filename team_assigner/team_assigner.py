from sports.common.team import TeamClassifier
import numpy as np

class TeamAssigner:
    """
    Assigns players to teams using the SigLIP + UMAP + KMeans approach
    from the roboflow/sports library, replacing the old raw RGB KMeans.
    """
    def __init__(self, device='cpu'):
        self.team_classifier = TeamClassifier(device=device, batch_size=8)
        self.player_team_dict = {}
        
        # BGR Colors for OpenCV (Pink and Blue)
        self.team_colors = {
            1: (147, 20, 255),  # Pink
            2: (255, 191, 0)    # Blue
        }
        self.is_fitted = False

    def get_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]

    def fit_from_crops(self, crops):
        """
        Fits the classifier on a diverse list of player crops.
        """
        if len(crops) > 0:
            self.team_classifier.fit(crops)
            self.is_fitted = True

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predicts the team for a player. If already predicted, returns from cache.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if not self.is_fitted:
            return 1 # Fallback if no players were found in the first frame

        crop = self.get_crop(frame, player_bbox)
        
        # Guard against empty crops (e.g., if bbox is outside frame)
        if crop.size == 0:
            return 1
            
        team_id = self.team_classifier.predict([crop])[0]
        # TeamClassifier returns 0 or 1. Map to 1 or 2.
        team_id += 1

        self.player_team_dict[player_id] = team_id
        return team_id

