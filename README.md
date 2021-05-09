# Purpose of this repository:
This repository provides an example of painting color over the lips region
based on facial landmark prediction with the Helen dataset, using transfer
learning & fine tuning with a 3rd party pretrained model.

# What's in this repository?
1. A series of notebooks which illustrates the conversion of the raw Helen
dataset to multiple TFRecord files.

- See `tfrec_gen` folder.
- The `helen_ds_target_pd.ipynb` & `helen_ds_img_pds.ipynb` convert the
annotation & images from the Helen dataset to pandas pickle files.
- The `helen_ds_img_target_pd_tfrec.ipynb` converts the pickle files to
TFRecord files.

2. A notebook illustrating transfer learning & fine tuning with a 3rd party
pretrained model. See `transferLearn_fineTune.ipynb`.
3. A Helper class that paints an overlay over the lips region of a facial
image based on ground truth as well as predictions from the learned model. See
`Helper` folder.

# Helen dataset:
You can get the Helen dataset [here](http://www.ifp.illinois.edu/~vuongle2/helen/) [2].

The facial features annotation follows the structure below:
1. 41 points for facial outline.
2. 17 points for nose outline.
3. 20 points each for eyebrows outlines.
4. 20 points each for eyes outlines.
5. 28 points for inner mouth outline.
6. 28 points for outer mouth outline.

# 3rd party pretrained model:
You can get the 3rd party pretrained model [here](https://github.com/nyoki-mtl/keras-facenet) [3].

The 3rd party pretrained model is a facenet model pretrained with the
MS-Celeb-1M dataset.

# Directory structure:
The code in this repository uses the following directory structure. You may
want to manually create the folders before running the notebooks.

The raw Helen dataset should be placed in the `helen_data` folder using the
following directory structure.

The 3rd party pretrained model should be placed in the `facenet/model`
directory.

```
├───chkpt
├───facenet
│   └───model
├───helen_data
│   ├───annotation
│   │   └───annotation
│   ├───helen_1
│   │   └───helen_1
│   ├───helen_2
│   │   └───helen_2
│   ├───helen_3
│   │   └───helen_3
│   ├───helen_4
│   │   └───helen_4
│   └───helen_5
│       └───helen_5
├───helen_pickle
├───Helper
├───saved_fig
├───saved_model
├───tfrec_data
└───tfrec_gen
```

# Results:
You can view the fine tuning results on unseen test set
[here](https://ibb.co/WxLxdrC). The red overlay on the lips region are the
ground truth while the blue ovelay is from the model's prediction.

# Credits & References:
1) [Interactive Facial Feature Localization](http://www.ifp.illinois.edu/~vuongle2/helen/eccv2012_helen_final.pdf)
Vuong Le, Jonathan Brandt, Zhe Lin, Lubomir Boudev, Thomas S. Huang
ECCV2012

2)  Vuong Le, Jonathan Brandt, Zhe Lin, Lubomir Boudev, Thomas S. Huang.
(2012).
Helen dataset.
Retrieved from [http://www.ifp.illinois.edu/~vuongle2/helen/](http://www.ifp.illinois.edu/~vuongle2/helen/)

3) nyoki-mtl (2017) keras-facenet [https://github.com/nyoki-mtl/keras-facenet](https://github.com/nyoki-mtl/keras-facenet)
