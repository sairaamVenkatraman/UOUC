## Image Generation
This gives the code for generating the UOUC dataset.

__Step 1:__

> python `Download.py`. 

This downloads the objects into `Data/objects`. 
_The data is made available by Miguel Zavala, with permission to use his models for the dataset._

__Step 2:__

Set group information [partitioning class id's] in `groups.json`. Update the path information and group information in `SceneGenerator-conf.py` to generate group with
> python SceneGenerator_train.py

This generates scene templates for each scene. Similarly, for test-set `SceneGenerator_test_X.py` is made available.


__Step 3:__

Update SetObj-conf.json for scenes and save_path to set the scene template path and where to save the rendered files. The rendering needs blender. We used blender 2.91.
We used 3 gpus to generate the scenes. Data ditribution can be done, by setting star and end in `SetObject-Scene.py` to one of the values of the range in `SetObj-conf.json.`
> CUDA_VISIBLE_DEVICES=X
> 
> python SetObject-Scene.py

GPU device id is set in the `SetObject-Scene.py` as well


## Bounding Box generation

__Step 4: (Optional)__

To generate bounding boxes for each scene set the path variables in `bounds.py`. The code can the be run.
> python bounds.py


## Question and Answer generation

__Step 5:__

The code generates one question per type for each scene. Set the variables in `qanda/QA/QGenerator.py` to choose the group or test set for generating questions and answers. 
> python QGenerator.py
