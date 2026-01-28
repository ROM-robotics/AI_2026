# Anaconda Installation on Ubuntu (with Jupyter & Spyder)

This document provides a step-by-step guide to install **Anaconda**, **Jupyter Notebook / JupyterLab**, and **Spyder IDE** on **Ubuntu Linux**.

---

## 1. Check if Anaconda Is Already Installed

Open a terminal and run:

```bash
conda --version 
```

## 2. If not install install conda form official site or cli

```bash
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
```

## 3. Install
```bash
bash Anaconda3-latest-Linux-x86_64.sh
```

## 4 Follow the instructions:

#### 1.Press Enter to continue

#### 2.Type yes to accept the license

#### 3.Press Enter to accept the default install path:

```bash
/home/username/anaconda3
```

#### 4.When prompted
```bash
  Do you wish the installer to initialize Anaconda3?
```
Type yes!

## 4. Activate Anaconda

Reload the shell configuration:

```bash 
source ~/.bashrc
```
Verify installation:

```bash 
conda --version
```

## 5. Update Conda

```bash
conda update conda -y
```

## 6. Install Jupyter Notebook and JupyterLab

```bash
conda install jupyter jupyterlab -y
```

Run Jupyter Notebook

```bash
jupyter notebook
```

Run JupyterLab

```bash
jupyter lab
```

## 7. Install Spyder IDE

install spider
```bash
conda install spyder -y
```

Run Spyder:

```bash
spyder
```
## 8. Verify Installation

Check Python version:

Check Python version:
```bash 
python --version
```
Check Conda configuration:
```bash
conda info
```

## 9. Create a New Conda Environment

Recommended for project isolation:

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
```

Install Jupyter kernel for the environment:

```
conda install ipykernel -y
python -m ipykernel install --user --name myenv

```

```bash 
pip freeze | geep -v "file://" > requirment.txt

```




