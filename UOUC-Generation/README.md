This gives the code for generating the UOUC dataset.
To generate, first run Download.py. This downloads the objects into Data/objects.
Then for each group of train (you will have to update SceneGenerator-conf.json in Scene to generate each group), run SceneGenerator_train.py in Scene.
This generates scene templates for each scene.
Then, for every test set, (run SceneGenerator_test_1.py to generate scene templates for first_test) to generate scene template.
After the scenes are generated, update SetObj-conf.json for scenes and save_path to set the scene template path and where to save the rendered files.
The rendering needs blender. We used blender 2.91.
We used 3 gpus to generate the scenes. We distributed the data across them, by setting star and end in SetObject-Scene.py to one of the values of the range in SetObj-conf.json.
Use a gpu by specifying CUDA_VISIBLE_DEVICES while running SetObject-Scene.py. Set the same gpu inside the code.
Thus, we can generate scenes for a group of scenes. 
Then run bounds.py, which adds a bounding box to each object and to scenes.
