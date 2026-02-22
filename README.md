# Predictive ML Project: Repeatable Daily Workflow

## Project Setup and Workflow

This section documents the repeatable daily workflow for setting up and working on this predictive ML project.

### 1. Pull Latest Changes
```
git pull
```

### 2. Create and Activate Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages
```
pip install jupyterlab numpy pandas pyarrow matplotlib seaborn scipy
```

### 4. Start JupyterLab
```
jupyter lab
```

### 5. Edit and Run Your Notebook
- Open or create your notebook (e.g., `yourname_eda.ipynb`).
- Follow the project steps for EDA and ML.

### 6. Save, Commit, and Push Changes
```
git add .
git commit -m "Update notebook and workflow documentation"
git push
```

---

# Quickstart: How to Set Up and Run the EDA Notebook

Follow these steps to set up your environment, obtain the Jupyter notebook, and run your EDA analysis:

1. **Clone the Repository**
    ```sh
    git clone <repo-url>
    cd datafun-06-eda
    ```

2. **Create and Activate a Virtual Environment**
    - On macOS/Linux:
      ```sh
      python3 -m venv .venv
      source .venv/bin/activate
      ```
    - On Windows:
      ```sh
      py -m venv .venv
      .venv\Scripts\activate
      ```

3. **Install Required Packages**
    ```sh
    python3 -m pip install jupyterlab pandas pyarrow matplotlib seaborn ipykernel
    ```

4. **Open JupyterLab or Jupyter Notebook**
    ```sh
    jupyter lab
    # or
    jupyter notebook
    ```

5. **Open and Run the Notebook**
    - Open your notebook file (e.g., `Sasta12_eda.ipynb`).
    - Run each cell in order. The notebook uses a built-in seaborn dataset (`tips`), so no external data file is needed.
    - To use a different dataset, change:
      ```python
      df = sns.load_dataset('tips')
      ```
      to another dataset name, like `'iris'` or `'penguins'`.

6. **Save and Commit Your Work**
    ```sh
    git add .
    git commit -m "Run EDA notebook with seaborn dataset"
    git push
    ```

# Specification for Project 6 EDA Notebook

## Overview

Project 6 is an opportunity to create your own custom
exploratory data analysis (EDA) project using
GitHub, Git, Jupyter, pandas, Seaborn and other popular data analytics tools.

## Deliverable Names

- GitHub Repository:  **datafun-06-eda**
- Documentation:      README.md
- Notebook:           **yourname_eda.ipynb**

## Start a New Project

Follow this common workflow to start a new project.

1. In your browser, create a GitHub project repository with a default README.md. Name the repo as specified above.
2. Clone your new repository down to your machine into your Documents folder.
3. Open your new project repository folder in the Documents folder of your machine in VS Code (if you haven't already).
4. In VS Code, add a useful .gitignore file with a line for .vsode/ and .venv/ and whatever else doesn't need to be tracked in the repository.
5. In VS Code, edit your README.md to record your commands, process, and notes so far.
6. In VS Code, open a terminal - PowerShell if Windows, zsh or bash if Mac/Linux.
7. Use the terminal to git add your files and folders to source control, and git commit your changes with a useful message (e.g. "initial commit"), and git push the changes up to GitHub.
8. Verify your GitHub repository.

## External Dependencies

This project requires at least the following external modules, so a local project virtual environment is recommended.

- jupyterlab: Enables Jupyter notebooks.
- pandas: Handles data manipulation and analysis, focusing on structured (tabular/panel) data.
- pyarrow: Required by pandas; facilitates interaction between pandas and the Arrow format.
- matplotlib: Basic tools for plotting and visualizing data.
- seaborn: Simplifies complex visualizations and statistical plots, built on matplotlib.

## Objective

Perform and publish a custom EDA project to demonstrate skills with Jupyter, pandas, Seaborn and popular tools for data analytics.
The notebook should tell a data story and visually present findings
in a clear and engaging manner.

## Explore Datasets

Choose a dataset for analysis.
You will want a known, clean dataset.
Cleaning data can run 60-80% of the project (or more) - you don't need to
demonstrate cleaning skills for this project.  
The recommended approach is to select one of the other pre-installed datasets in Seaborn.
You can view a list of the Seaborn datasets in the first link below.
The additional links offer a range of options.

- [List of Seaborn Datasets Installed](https://github.com/mwaskom/seaborn-data)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Data.gov](https://www.data.gov/)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

You may use your own data if you have permission and there is no confidential information included.
Be careful with your data selection and ensure you have rights to use the content.

## Requirements

### 1. Create and Manage Project Virtual Environment

This project uses external packages, which are not included in the Python Standard Library - we must install them.
To keep our project separate from all other Python projects,
we will create and manage a local project virtual environment.
We'll install our packages into the local project virtual environment.
For the recommended process with detailed steps and commands, see [PROJECT_VIRTUAL_ENV.md](PROJECT_VIRTUAL_ENV.md).

### 2. Project Start

Make sure Jupyter is installed and working in your project virtual environment.
Document the process and commands you used in your README.md.

Then create, open, and start a new notebook in your root project repository folder:

1. Create the Notebook: In the VS Code Explorer, create a new file i.e., yourname_eda.ipynb. Ensure it has a .ipynb extension.
2. Verify your new notebook is open for editing. If needed, view the project files in VS Code Explorer and double-click the notebook file to open it for editing.
3. Add a Markdown cell at the top of your notebook with the introduction (include the title, author, date and the purpose of the project).

### 3. Import Dependencies (At the Top, After the Introduction)

Add a Python cell next with the import statements for the libraries you will use in the project.
Follow conventional package import organization and alias.
Import each package just once near the top of the file.
Be sure you have INSTALLED any external packages (outside the Python Standard Library) into your active project virtual environment first.

Jupyter Notebook / Python cell example:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```
Execute the cell to ensure everything works.
If you get errors on one of the statements above, the most common issue is that package has not been installed into the active project virtual environment.
When you find you need a new package, first install it into the active project virtual environment and then import it near the top of your Python or Notebook file.


### 5.  Exploratory Data Analysis

Perform a unique exploratory data analysis project using the tools and skills covered previously.

#### Step 1. Data Acquisition

Load the data into a pandas DataFrame.
Use the pd read functions such as pd.read_csv() or pd.read_excel() as appropriate.
To read from the Seaborn dataset, we'll use sns.load_dataset() function and pass in the 'iris' (the name without .csv) to populate our DataFrame.

Jupyter Notebook / Python cell example:

```python
# Load the dataset into a pandas DataFrame - adjust this process for your custom data
df = sns.load_dataset('iris')

# Inspect first rows of the DataFrame
print(df.head())
```

#### Step 2. Initial Data Inspection

Display the first 10 rows of the DataFrame, check the shape, and display the data types of each column using df.head(10), df.shape, and df.dtypes.

Jupyter Notebook / Python cell example:

```python

print(df.head(10))
print(df.shape)
print(df.dtypes)
```
#### Step 3. Initial Descriptive Statistics

Use the DataFrame describe() method to display summary statistics for each column.

Jupyter Notebook / Python cell example:

```python
print(df.describe())
```

#### Step 4. Initial Data Distribution for Numerical Columns

Choose a numerical column and use df['column_name'].hist() to plot a histogram for that specific column.
To show all the histograms for all numerical columns, use df.hist().

Jupyter Notebook / Python cell example:

```python
# Inspect histogram by numerical column
df['sepal_length'].hist()

# Inspect histograms for all numerical columns
df.hist()

# Show all plots
plt.show()
```
Afterwards, use a Markdown cell to document your observations.

#### Step 5. Initial Data Distribution for Categorical Columns

Choose a categorical column and use df['column_name'].value_counts() to display the count of each category.
Use a loop to show the value counts for **all** categorical columns.

Jupyter Notebook / Python cell example:

```python
# Inspect value counts by categorical column
df['species'].value_counts()

# Inspect value counts for all categorical columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    # Display count plot
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Show all plots
plt.show()
```

Afterwards, use a Markdown cell to document your observations.

#### Step 6. Initial Data Transformation and Feature Engineering

Use pandas and other tools to perform transformations.
Transformation may include renaming columns, adding new columns,
or transforming existing data for more in-depth analysis.

For this project, you must:

1. Rename at least one column.
2. Add at least one column.

#### Step 7. Initial Visualizations

Create a variety of chart types using seaborn and matplotlib to showcase different aspects of your data.
For each chart, include the goal - what you want to learn/explore, the type of chart you choose, display the chart, and tell your data story. Use Markdown cells and Python cells. Create at least 3 subsections - each subsection should have the following parts:

1. Goal: The question you are exploring.
2. Chart Type: Tell us what kind of chart you choose to illustrate this goal.
3. Chart: Display the chart.
4. Story: Use Markdown cell(s) to document your observations and insights.

#### Step 8. Initial Storytelling and Presentation

Present your notebook with an opening that introduces yourself and your topic.
Use Markdown section headings to introduce each step.
Interpret the visualizations and statistics to narrate a clear and compelling data story.
Present your findings in a logical and engaging manner.

## Notebook Design

- Begin your notebook with a project summary including the title, author, date, and project's purpose. This provides an immediate understanding of the notebook's objective.
- Ensure your code and presentation are neat, well-organized, and follow good coding practices. This includes proper variable naming, consistent code style, and logical organization of code cells.
- Use Markdown features effectively for formatting, such as section headings, bullet points, and emphasis (bold/italic), to enhance readability.

## Notebook Structure and Documentation

Once the notebook runs without errors, focus on how the notebook content is structured and documented.
Organize your notebook into well-defined sections, each with a clear purpose and header.
Use Markdown cells to provide context, explain your analysis, and share findings. This makes your notebook informative and engaging.
Comment your code cells to explain the purpose and functionality of the code. This is especially important for complex or non-obvious code segments.

## Notebook Execution

Run your notebook entirely to ensure it executes without errors. This includes checking all code cells and ensuring all data visualizations render as expected.
Confirm that your notebook renders correctly on GitHub after pushing, as this ensures your work is viewable by others.

## Evaluation Criteria

- Functionality: The project should be functional and meet all requirements.
- Documentation: The project should be well-written and well-documented.
- Presentation: The project should be presented in a clear and organized manner.
- Professionalism: The project should be submitted on-time and reflect an original, creative effort.

See rubric for additional information.

## Resources

- See [datafun-04-spec](https://github.com/denisecase/datafun-04-spec) for a guided EDA.
- See [JUPYTER.md](https://github.com/denisecase/datafun-04-spec/JUPYTER.md) for Jupyter Notebook keyboard shortcuts and recommendations.
- See [MARKDOWN.md](https://github.com/denisecase/datafun-04-spec/MARKDOWN.md) for Markdown syntax and recommendations.
- See [Plotting graph For IRIS Dataset Using Seaborn And Matplotlib](https://www.tutorialspoint.com/plotting-graph-for-iris-dataset-using-seaborn-and-matplotlib)
- See [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

## Commands and Process Used

Below are the actual commands and process followed for this project:

```
git pull
python3 -m venv .venv
source .venv/bin/activate
pip install jupyterlab numpy pandas pyarrow matplotlib seaborn scipy
jupyter lab
# (edit and run notebook)
git add .
git commit -m "Initial predictive ML project setup and workflow documentation"
git push
```

Notes:
- The virtual environment is created and activated before installing packages.
- All required packages are installed in the local environment.
- JupyterLab is used for interactive development.
- All changes are tracked and pushed to GitHub regularly.
