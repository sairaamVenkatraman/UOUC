The code for running the experiments on UOUC are here.

Code in the repository uses https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50

Each of model_1, model_2 and model_3 contain code for the models used as baselines.

Please run the train.py for getting resnet50 model. The checkpoint after training is to be used for the models for VQA.

NVIDIA dali is needed to train the models.

Download the qa_file, object_detection_data and qa_dicts from the UOUC downloads for use in training for VQA
