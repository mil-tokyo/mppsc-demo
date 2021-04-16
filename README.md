# Missing Position Prediction + Story Completion Demo System

This is a prototype demonstration system for Human Story Writing Assistance.
Based on the "Missing Position Prediction (MPP)" approach we have proposed [1][2].

![スクリーンショット 2021-03-25 22 27 27](https://user-images.githubusercontent.com/2755894/112496809-6278ed80-8dc8-11eb-986c-d666c5a3b3cf.png)

You can run this demo system on your own computing environment or on Colaboratory.

## Run on your own Computing Environment

### Requirements

You can install the required packages as follows.

`pip install -r requirements.txt`

After preparing an environment, please see the `trained_model` directory and download the trained model for running the demo system.

### Run

`streamlit run mppsc-streamlit.py`

## Run on Colaboratory

You can quickly run the system on Colaboratory (Google Colab) by just getting ngrok auth token from https://dashboard.ngrok.com/auth.  
Please open `mppsc_demo_streamlit.ipynb` in Colab, then input your auth token appropriately. 
Run all cells in the file, and now you can access the demo system.

## References

1. Yusuke Mori, Hiroaki Yamane, Yusuke Mukuta, Tatsuya Harada, “Finding and Generating a Missing Part for Story Completion,” 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL) (COLING 2020, Workshop), 2020. [[PDF](https://www.aclweb.org/anthology/2020.latechclfl-1.19.pdf)] [[Code](https://github.com/mil-tokyo/missing-position-prediction)]
2. Yusuke Mori, Hiroaki Yamane, Yusuke Mukuta, Tatsuya Harada, “The Nectar of Missing Position Prediction for Story Completion,” Text2Story 2021 (ECIR 2021, Workshop), 2021. [[Video](https://youtu.be/IfnBtOsXq6M)] [(PDF is to appear)]


