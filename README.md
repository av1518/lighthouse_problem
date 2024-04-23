# S2: Statistics for Data Science Coursework
# The Lighthouse Problem


### Description
In this project, we tackle the famous lighthouse problem: we infer the position of a lighthouse out to sea just by its measured flash locations along the shore. We use MCMC sampling to do so, and we include the intensity measurements in the second part.

The report is available in the repository folder `report`.



### Installation
Dependencies required to run the project are listed in the `environment.yml` file. To install the necessary Conda environment from these dependencies, run:
```bash
conda env create -f environment.yml
```
If the environment runs into OS-specific dependency issues (this environment was made on a Windows machine), then try installing from the `environment_from_history.yml`. This only contains the installs as typed on the terminal.

Once the environment is created, activate it using:

```bash
conda activate s2
```

### Project Scripts Overview
To obtain all figures and results in the report simply run the following from base of the directory:

```bash
python src/main.py
```

Here's an overview of each script:

| Script                    | Usage                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `main.py`                 | Main script for reproducing the plots and results used in the report, and saves them into the `figures` folder. |
| `main_v.py`        | Scipt that produces the results just for part (v). To run from terminal, change directory to `src` first, and run `python main_v.py`. |
| `main_vii.py`               | Scipt that produces the results just for part (v). To run from terminal, change directory to `src` first, and run `python main_vii.py`                                         |
| `funcs.py`           | Contains all the utility functions used throughout the project.     |

- All the code was ran on a CPU: Ryzen 9.

## Dockerfile Instructions
The user can build and run the solver in a Docker container using the `Dockerfile` provided in the repository. From the root directory, build the image with:

```bash
$ docker build -t stats .
```

This generates a Docker image called `stats`. To deploy and run the package in the container with a local input file, run the following command:

```bash
$ docker run --rm -ti s2
```

This setup uses a virtual Ubuntu environment with Miniconda, installs the necessary packages and activates the environment. 



### Contributing

Contributions are welcome. Please open an issue to discuss significant changes.

### License
This project is open-sourced under the [MIT](https://choosealicense.com/licenses/mit/) License.

