# S2: Statistics for Data Science Coursework
# The Lighthouse Problem


### Description
In this project, we tackle the famous lighthouse problem: we infer the position of a lighthouse out to sea just by its measured flash locations along the shore. We use MCMC sampling to do so, and we include the intensity measurements in the second part.



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



### Contributing

Contributions are welcome. Please open an issue to discuss significant changes.

### License
This project is open-sourced under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Autogeneration tools:
Co-pilot was used in the following way:
- Autocompleting documentation for functions and for scripts
- Understand error messages with "Explain this error" button next to the error message.
- General questions such as:

How can i change a function name and have it change throughout my workspace?  

Answer: In Visual Studio Code, you can use the "Rename Symbol" feature to change a function name throughout your workspace. Here are the steps:

Place your cursor on the function name you want to change.
Right-click the function name, and select "Rename Symbol" from the context menu. Alternatively, you can use the shortcut F2.
Enter the new function name and press Enter.
VS Code will automatically find and update all occurrences of the function name in your workspace. This includes not only the function definition, but also all places where the function is called.

Please note that this feature depends on the language support provided by the installed extensions. If you're working with a language that doesn't have good support, the "Rename Symbol" feature might not work
