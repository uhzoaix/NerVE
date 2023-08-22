# Data Preparation

We aim to process the raw data from [ABC dataset](https://deep-geometry.github.io/abc-dataset/), and make them the proper form for NerVE model. Step files, Feature files and Object files in ABC dataset are enough for our processing.


### Step File Processing

Step file format containing the parametric boundary representation of a CAD model. Here we load `.step` file and sample the parametric curves with [OpenCascade](https://www.opencascade.com/). Specifically we use its Python version [pythonocc-core](https://github.com/tpaviot/pythonocc-core) and [pythonocc-utils](https://github.com/tpaviot/pythonocc-utils). 

- `step_samples.py`

In this file, it samples the parametric curves of a CAD model to form the piecewise-linear(PWL) edges, given the sampling rate(related to the resolution). Also note that only sharp edges(curves) are considered, which are specified in their Feature file.


### Step PWL Edges to NerVE 

- `step_edge2NerVE.py`

Given the curves in the cube grid, it aims to find out ground truth NerVE grid, that is, the cubes which are occupied, the faces which are passed through and mid points of curves inside the cubes. 

Basically, the algorithm can handle piecewise-linear curves as long as all line segments is no longer than the cube edge of grid. And as pointed out in the paper, it may fail to recover when two curves are too close.

### Object Point Cloud

- `obj_point_cloud.py`

Extract the point cloud from `.obj` file as input, and pre-compute K-nearest neighbors for each point.

### Usage

```
python step_samples.py
python step_edge2NerVE.py
python obj_point_cloud.py
```

Just simply run the codes in order. Remember to specify the paths of dataset in the three files.