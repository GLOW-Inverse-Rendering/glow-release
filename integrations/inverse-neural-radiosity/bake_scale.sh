for scene in 'colloc-living-room-1-spot-specular' 'colloc-staircase-1_modified' 'custom_kitchen_ver9_principled_path-spot-specular_more'; do
    mv data/datasets/collocated/${scene}/transforms.json data/datasets/collocated/${scene}/transforms_orig.json
    mv data/datasets/collocated/${scene}_val/transforms.json data/datasets/collocated/${scene}_val/transforms_orig.json
    python3 bake_scale_mat.py data/datasets/collocated/${scene}/transforms_orig.json ../WildLight/datasets/synthetic/${scene}/cameras_sphere.npz data/datasets/collocated/${scene}/transforms.json
    python3 bake_scale_mat.py data/datasets/collocated/${scene}_val/transforms_orig.json ../WildLight/datasets/synthetic/${scene}_val/cameras_sphere.npz data/datasets/collocated/${scene}_val/transforms.json
done
