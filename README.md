# Missing Position Prediction + Story Completion Demo System

This is a prototype demonstration system for Human Story Writing Assistance.
Based on the "Missing Position Prediction (MPP)" approach we have proposed.

![スクリーンショット 2021-03-25 22 27 27](https://user-images.githubusercontent.com/2755894/112496809-6278ed80-8dc8-11eb-986c-d666c5a3b3cf.png)

## Requirements

### Preparing an Environment

You can install the required packages as follows.

`pip install -r requirements.txt`

### Model Download

After preparing an environment, please see the `trained_model` directory and download the trained model for running the demo system.

## Run

`streamlit run mppsc-streamlit.py`

## Run on Colaboratory

You can quickly run the system on Colaboratory (Google Colab) by just getting ngrok auth token from https://dashboard.ngrok.com/auth.  
Please open `mppsc_demo_streamlit.ipynb` in Colab, then input your auth token appropriately. 
Run all cells in the file, and now you can access the demo system.
