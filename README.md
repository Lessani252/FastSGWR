FastSGWR: "Enhancing the Computational Efficiency of the SGWR Model and Introducing Its Software Implementation." This article along the python packages (parallel & sequential), and a Graphic User Interface (GUI) tool named 'SGWR Model' is developed based on this article: similarity and geographically weighted regression (SGWR) (https://doi.org/10.1080/13658816.2024.2342319).  

Installation
------------
- pip install sgwr

Citation
--------
If you use this package in your work, please cite the following articles:

1. Lessani, M. Naser, and Zhenlong Li. "SGWR: similarity and geographically weighted regression." International Journal of Geographical Information Science 38, no. 7 (2024): 1232-1255. (https://doi.org/10.1080/13658816.2024.2342319)
2. Lessani, M. Naser, and Zhenlong Li. "Enhancing the Computational Efficiency of the SGWR Model and Introducing Its Software Implementation." Annals of GIS (2025).
   
Overview
--------
The SGWR (Similarity and Geographically Weighted Regression) model is a novel local spatial regression model that extends the conventional GWR by incorporating both geographical proximity and attribute similarity into a composite spatial weight matrix. The  combination of spatial and attribute-based weights is governed by a parameter alpha, which is optimized based on AICc measure.

This Python package includes:
----------------------------
- MPI-enabled parallel implementations of the SGWR
- Serial version of the SGWR model
- Support for Gaussian and bi-square kernels
- Also, supports a combination of adaptive bisquare and gaussian
- Alpha Optimization: Automatically tune the contribution of similarity and spatial proximity.
- Also, users have the ability to either chose standardize or don't standardize their data in parallel version
- Also, users can run the GWR model in both parallel and serial version of this library

Data Format for parallel version (MPI)
--------------------------------------
Input data must be a CSV file with the following column order:
- longitude, latitude, dependent_variable, independent_variable_1, ..., independent_variable_n
- In the attached figure, "y" stands for dependent variable, "x1,x2,...xn" stand for independent variables
- "Longitude" and "Latitude" are the coordinate

![data format](https://github.com/user-attachments/assets/e5e6547d-5eb0-444a-a9be-8b315cbf9997)

Categorical variables
---------------------
Categorical variables must be pre-processed into dummy variables.
Example: For a 3-class variable ("urban", "peri-urban", "rural"), create:
- urban_dummy: 1 if urban, else 0
- peri_urban_dummy: 1 if peri-urban, else 0
- Rural becomes the reference class (excluded)

![Example](https://github.com/user-attachments/assets/08a252df-c9ef-414a-ba30-a41914016e50)


Usage
-----
After preparing your dataset and ensuring all dependencies are installed, the model can be run via the command line:

MPI Commands (parallel):
----------------------------
- fastsgwr run -np x -data path_to_data (by default the kernel is Gaussian function, and doesn't standardize the data)
- fastsgwr run -np x -data path_to_data -standardize (using Gaussian function and standardize the input data)
- fastsgwr run -np x -data path_to_data -bisquare 
- fastsgwr run -np x -data path_to_data -bisquare -standardize
- fastsgwr run -np x -data path_to_data -biga (adaptive bisquare and gaussian)
- fastsgwr run -np x -data path__to_data -gwr (run gwr as well in parallel)
- x: Number of cores
- path_to_data: Path to the CSV dataset

The output will be a CSV file saved in the same input directory, and containing local coefficients and performance metrics.

Serial commands:
----------------------------
- selector = ALPHA(g_coords, g_y, g_x, data, fixed=True, kernel='gaussian') ## for fixed bandwidth and gaussian kernel
- bw, alpha = selector.fit()
- sgwr_model = SGWR(g_coords, g_y, g_x, bw, data, alpha, fixed=True, kernel='gaussian') ## after bandwidth and alpha optimizationi
- result = sgwr_model.fit()
- selector = ALPHA(g_coords, g_y, g_x, data) ## for adaptive bandwidth and bisquare kernel
- bw, alpha = selector.fit()
- sgwr_model = SGWR(g_coords, g_y, g_x, bw, data, alpha) 
- result = sgwr_model.fit()
- if the 'alpha' value set to one (1) here (SGWR(g_coords, g_y, g_x, bw, data, 1)), then the model acts as GWR model.  

Parameter extraction when running in the serial mode:
----------------------------
- result.R2
- result.adj_R2
- result.aicc
- result.aic
- result.params
- result.bse
- result.localR2
- result.filter_tvals()
- result.filter_tvals(alpha=0.05) ### t values with 95% confidence interval
- result.summary()

The GUI tool includes:
---------------------
- Three types of bandwidths are supported: adaptive (bisquare), fixed (Gaussian), and adaptive bisquare for bandwidth optimization, followed by adaptive Gaussian for alpha optimization and model fitting.
- Three options are available for alpha optimization: a pre-defined value, a greedy optimization approach, and a divide-and-conquer strategy.
- Two options are availabel for bandwidth optimization: golden section and pre-defined.
- Users can also choose whether to standardize the data by selecting the "Variable Standardization" option under the 'Additional' settings. By default the tool standardize the data.
- The tool can also run the GWR model. To do so, simply enter a value of one (1) for alpha in the predefined box before running the model.
- The tool also contains the result for GLM in the output result. 
  
![SGWR_screen](https://github.com/user-attachments/assets/fdd29f0b-ee43-4a42-92c2-b80bb2ada358)

We acknowledge that this tool has been developed on top of the MGWR GUI tool.

Installation requirements for the python package
------------------------------------------------
Ensure the following dependencies are installed:

Standard Python Libraries:
- argparse
- typing (Optional)
- itertools (combinations)

MPI and CLI:
- mpi4py
- click

Numerical and Data Handling:
- numpy
- pandas
- scipy (stats, linalg, spatial.distance)

Machine Learning and Metrics:
- scikit-learn (metrics)

Specialized Geospatial Modeling:
- spglm (family, glm, iwls, utils)

Author info
------------
- Code Author: M. Naser Lessani (GIBD)
- Realeased Year: 2025
- Affiliation: Geoinformation and Big Data Research Laboratory (GIBD), Department of Geography, The Pennsylvania State University, University Park, PA, USA (https://giscience.psu.edu/) 
