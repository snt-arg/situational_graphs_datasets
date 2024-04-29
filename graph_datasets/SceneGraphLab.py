import json, os, sys, time

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)
from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph



class SceneGraphLab():
    def __init__(self):
        with open(os.path.join(os.path.dirname(synthetic_datset_dir),"config", "new_concepts.json")) as f:
            self.synteticdataset_settings = json.load(f)
        self.prepare_dataset()

    def prepare_dataset(self):
        dataset_generator = SyntheticDatasetGenerator(self.synteticdataset_settings, None, None)
        dataset_generator.create_dataset()
        visualize_nxgraph(dataset_generator.graphs["original"][0], image_name = f"test_original_base")

        filtered_nxdataset = dataset_generator.get_filtered_datset(['ws'], ['ws_same_room'])["noise"]
        visualize_nxgraph(filtered_nxdataset[0], image_name = f"test_noise_filtered")

        # extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, 'ws_same_room', "training")
        # self.normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)

sgl = SceneGraphLab()

time.sleep(100)