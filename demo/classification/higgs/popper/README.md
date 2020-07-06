# Performance Validation Workflow with HIGGS
## Using Popper

[Popper](https://github.com/systemslab/popper) is a tool for defining and executing container-native workflows in Docker, as well as other container engines. More details about Popper can be found [here](https://popper.readthedocs.io/).

## Description

This folder contains a `wf.yml` file that defines a Popper workflow for automatically downloading and verifying the complete [HIGGS data set](https://archive.ics.uci.edu/ml/datasets/HIGGS) from UCI (which has 11 million entries), running the benchmark to compare the liblinear library with xLearn and finally generating a report with a chart that shows the results including error bars. 

The benchmark tests the performance of each library by running five times the following set of main tasks:
- Load data set with the help of [Pandas](https://pandas.pydata.org/).
- Generate the trained linear model
- Predict

This is an example of how the chart looks:

![report](https://user-images.githubusercontent.com/33427324/86541248-39be6a00-bec0-11ea-8961-132951ac028f.png)
### Instructions:

1. Clone the repository.
```
git clone https://github.com/aksnzhy/xlearn.git
```

2. Install [docker](https://docs.docker.com/get-docker/).

3. Install the `popper` tool.
```
curl -sSfL https://raw.githubusercontent.com/getpopper/popper/master/install.sh | sh
```
4. Run the workflow.
```
cd xlearn/
popper run -f demo/classification/higgs/popper/wf.yml
```
There is a way to run a single step of the workflow in case you don't want to run the whole thing each time, you only have to add the name of the step at the end like the following example.
```
popper run -f demo/classification/higgs/popper/wf.yml prepare-data
```
When we are having problems with a step there is also an easy way to debug the workflow by opening an interactive shell instead of having to update the YAML file and invoke `popper run` again.
```
popper sh -f demo/classification/higgs/popper/wf.yml prepare-data
```
The example above opens a shell inside the container where other things can be done. More information on this matter can be found [here](https://popper.readthedocs.io/en/latest/sections/getting_started.html#run-your-workflow).
