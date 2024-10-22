import numpy as np

class NodeEdgeFeatureEmbeddingBuildier():
    def __init__(self, entity_type, feature_dictionary):
        self.entity_type = entity_type
        self.feature_dictionary = feature_dictionary

    def update_feature_dictionary(self, feature_dictionary):
        self.feature_dictionary.update(feature_dictionary)

    def build_embedding(self, full_feature_keys):
        embedding = []
        if self.entity_type == "node":
            embedding = self.build_node_embedding(full_feature_keys)
        elif self.entity_type == "edge":
            embedding = self.build_edge_embedding(full_feature_keys)
        return embedding

    def build_node_embedding(self, full_feature_keys):
        
        def add_ws_node_features(feature_keys, feats):
            if feature_keys[0] == "centroid":
                feats = np.concatenate([feats, self.feature_dictionary["ws_center"][:2]]).astype(np.float32)
            elif feature_keys[0] == "length":
                feats = np.concatenate([feats, [self.feature_dictionary["ws_length"]]]).astype(np.float32)   #, [np.log(ws_length)]]).astype(np.float32)
            elif feature_keys[0] == "normals":
                feats = np.concatenate([feats, self.feature_dictionary["ws_normal"][:2]]).astype(np.float32)
            if len(feature_keys) > 1:
                feats = add_ws_node_features(feature_keys[1:], feats)
            return feats

        return add_ws_node_features(full_feature_keys, [])
        

    def build_edge_embedding(self, full_feature_keys):
        
        def add_edge_features(feature_keys, feats):
            if feature_keys[0] == "min_dist":
                feats = np.concatenate([feats, self.feature_dictionary["min_dist"]]).astype(np.float32)  #, np.log(distance+1)]).astype(np.float32)
            elif feature_keys[0] == "relative_pos":
                feats = np.concatenate([feats, self.feature_dictionary["rel_pos_1"][:2]]).astype(np.float32)
            elif feature_keys[0] == "centroids_distance":
                feats = np.concatenate([feats, [self.feature_dictionary["centroids_distance"]]]).astype(np.float32)
            elif feature_keys[0] == "angle_centroid_degrees":
                feats = np.concatenate([feats, [self.feature_dictionary["angle_centroid_degrees"]]]).astype(np.float32)
            elif feature_keys[0] == "relative_ang_normal":
                feats = np.concatenate([feats, [self.feature_dictionary["relative_ang_normal"]]]).astype(np.float32)
            if len(feature_keys) > 1:
                feats = add_edge_features(feature_keys[1:], feats)
            return feats
        
        return add_edge_features(full_feature_keys, [])

