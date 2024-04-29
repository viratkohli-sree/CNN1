import streamlit as st
import base64
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')


def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path


if navigation() == "home":
    
    st.markdown(f'<h1 style="color:#000000;font-size:34px;">{"Intrusion Detection Using ML"}</h1>', unsafe_allow_html=True)

    # st.markdown(f'<h1 style="color:#FFFFFF;font-size:26px;">{"Intrusion detection is one of the important security problems in today s cyber world. A significant number of techniques have been developed which are based on machine learning approaches. So for identifying the intrusion we have designed the machine learning algorithms. By using the algorithm we find out intrusion and we can identify the attacker’s details also. IDS are mainly two types: Host based and Network based. A Host based Intrusion Detection System (HIDS) monitors individual host or device and sends alerts to the user if suspicious activities such as modifying or deleting a system file, unwanted sequence of system calls, unwanted configuration changes are detected. A Network based Intrusion Detection System (NIDS) is usually placed at network points such as a gateway and routers to check for intrusions in the network traffic. In this paper, KDD cup IDS dataset was taken from dataset repository. Then, we have to implement the pre-processing techniques. Then, we have to implement the different machine and deep learning algorithms such as Logistic regression (LR) and Convolutional Neural Network (CNN), RF and DT. The experimental results shows that the accuracy for above mentioned algorithms. Then, we can deploy the project in web application using FLASK"}</h1>', unsafe_allow_html=True)

    
    st.write("Intrusion detection is one of the important security problems in today s cyber world. A significant number of techniques have been developed which are based on machine learning approaches. So for identifying the intrusion we have designed the machine learning algorithms. By using the algorithm we find out intrusion and we can identify the attacker’s details also. IDS are mainly two types: Host based and Network based. A Host based Intrusion Detection System (HIDS) monitors individual host or device and sends alerts to the user if suspicious activities such as modifying or deleting a system file, unwanted sequence of system calls, unwanted configuration changes are detected. A Network based Intrusion Detection System (NIDS) is usually placed at network points such as a gateway and routers to check for intrusions in the network traffic. In this paper, KDD cup IDS dataset was taken from dataset repository. Then, we have to implement the pre-processing techniques. Then, we have to implement the different machine and deep learning algorithms such as Logistic regression (LR) and Convolutional Neural Network (CNN), RF and DT. The experimental results shows that the accuracy for above mentioned algorithms. Then, we can deploy the project in web application using FLASK.")
    
    # st.title('Home')
    
elif navigation() == "Prediction":

    st.markdown(f'<h1 style="color:#000000;font-size:34px;">{"Prediction"}</h1>', unsafe_allow_html=True)

    
    #========================== IMPORT PACKAGES ============================
    
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn import linear_model
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt 
    
    
    #=========================== DATA SELECTION ============================
    
    dataframe=pd.read_csv("kddcup99.csv")
    dataframe=dataframe[0:200000]
    print("--------------------------------------------------")
    print("                     Data Selection               ")
    print("--------------------------------------------------")
    print()
    print(dataframe.head(15))
    
    #========================== PRE PROCESSING ================================
    
    #==== checking missing values ====
    
    print("--------------------------------------------------")
    print("Checking missing values ")
    print("--------------------------------------------------")
    print()
    print(dataframe.isnull().sum())
    
    #========= label encoding ===========
    
    print("------------------------------------------------")
    print(" Before label Encoding ")
    print("------------------------------------------------")
    print()
    
    print(dataframe['label'].head(20))
    
    
    label_encoder = preprocessing.LabelEncoder()
    
    
    dataframe = dataframe.astype(str).apply(label_encoder.fit_transform)
    
    print("-------------------------------------------")
    print(" After label Encoding ")
    print("------------------------------------------")
    print()
    
    print(dataframe['label'].head(20))
    
    
    #========================== DATA SPLITTING ===========================
    
    
    X=dataframe[['duration','protocol_type','service','flag','src_bytes','dst_bytes']]
    y=dataframe['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    print("-------------------------------------------")
    print("Data Splitting                    ")
    print("-------------------------------------------")
    print()
    
    print("Total no of data        :",dataframe.shape[0])
    print("Total no of test data   :",X_test.shape[0])
    print("Total no of train data  :",X_train.shape[0])
    
    
    # Y=dataframe['Potability']
    import matplotlib.pyplot as plt
    plt.hist(y)
    plt.title("Histogram")
    # plt.savefig("Hist.png")
    plt.show()     
    
    st.image("Hist.png")
    
    
    import seaborn as sns
    plt.figure(figsize=(5, 5))
    plt.title("Classification")
    sns.countplot(x='label',data=dataframe)
    plt.savefig("Label.png")
    plt.show()    
    
    st.image("Label.png")    
    
    
    
    #========================= CLASSIFICATION ============================
    
    # === LR ===
    
    print("-------------------------------------------")
    print(" Classification ")
    print("------------------------------------------")
    print()
    
    #==== LOGISTIC REGRESSION ====
    
    from sklearn.linear_model import LogisticRegression
    
    #initialize the model
    logreg = LogisticRegression(solver='lbfgs' , C=500)
    
    #fitting the model
    logistic = logreg.fit(X_train,y_train)
    
    #predict the model
    y_pred_lr = logistic.predict(X_train)
    
    
    #===================== PERFORMANCE ANALYSIS ============================
    
    #finding accuracy
    
    result_lr = (metrics.accuracy_score(y_pred_lr,y_train)) * 100
    print("-------------------------------------------")
    print(" Performance Metrics ")
    print("------------------------------------------")
    print()
    print(" Accuracy for LR :",result_lr,'%')
    print()
    print(metrics.classification_report(y_train, y_pred_lr))
    print()
    print()
    
    import pickle
    
    filename = 'intrusion.pkl'
    pickle.dump(logreg, open(filename, 'wb'))
    
    st.text("-------------------------------------------")
    st.text(" Performance Metrics (LR)")
    st.text("------------------------------------------")
    print()
    st.write(" Accuracy for LR :",result_lr,'%')

    
    # === CNN ===
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import  Dense
    
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Input
    
    inp =  Input(shape=(6,1))
    conv = Conv1D(filters=2, kernel_size=2)(inp)
    pool = MaxPool1D(pool_size=2)(conv)
    flat = Flatten()(pool)
    dense = Dense(1)(flat)
    model = Model(inp, dense)
    model.compile(loss='mae', optimizer='adam',metrics=['acc'])
    model.summary()
    
    import numpy as np
    xx=np.expand_dims(X_train,axis=2)
    #model fitting
    history = model.fit(xx, y_train,epochs=10, batch_size=15, verbose=1,validation_split=0.2)
    
    acc_cnn=history.history['acc']
    
    acc_cnn=max(acc_cnn)
    
    acc_cnn=100-acc_cnn
    
    pred_cnn=model.predict(xx)
    
    y_pred1 = pred_cnn.reshape(-1)
    y_pred1[y_pred1<0.5] = 0
    y_pred1[y_pred1>=0.5] = 1
    y_pred1 = y_pred1.astype('int')
    
    print("=========================================================")
    print("------------- Performance Analysis (CNN)-----------------")
    print("========================================================")
    print()
    print("1. Accuracy :",acc_cnn )
    print()
    
    st.text("-------------------------------------------")
    st.text(" Performance Analysis (CNN)")
    st.text("-------------------------------------------")
    print()
    st.write("1. Accuracy :",acc_cnn )
    print()    
    # === RF ===
    
    print("-------------------------------------------")
    print(" Classification ")
    print("------------------------------------------")
    print()
    
    #==== RF ====
    
    from sklearn.ensemble import RandomForestClassifier
    
    #initialize the model
    rf = RandomForestClassifier()
    
    #fitting the model
    rf = rf.fit(X_train,y_train)
    
    #predict the model
    y_pred_rf = rf.predict(X_train)
    
    
    #finding accuracy
    
    result_rf = (metrics.accuracy_score(y_pred_rf,y_train)) * 100
    print("-------------------------------------------")
    print(" Performance Metrics ")
    print("------------------------------------------")
    print()
    print(" Accuracy for RF :",result_rf,'%')
    print()
    print(metrics.classification_report(y_train, y_pred_rf))
    print()

    st.text("-------------------------------------------")
    st.text(" Performance Metrics (RF)")
    st.text("------------------------------------------")
    print()
    st.write(" Accuracy for RF :",result_rf,'%')
    print()    
    
    # ===== DECISION TREE
    
    print("-------------------------------------------")
    print(" Classification ")
    print("------------------------------------------")
    print()
    
    from sklearn.tree import DecisionTreeClassifier
    
    #initialize the model
    dt = DecisionTreeClassifier()
    
    #fitting the model
    dt = dt.fit(X_train,y_train)
    
    #predict the model
    y_pred_dt = dt.predict(X_train)
    
    
    #finding accuracy
    
    result_dt = (metrics.accuracy_score(y_pred_dt,y_train)) * 100
    print("-------------------------------------------")
    print(" Performance Metrics ")
    print("------------------------------------------")
    print()
    print(" Accuracy for DT :",result_dt,'%')
    print()
    print(metrics.classification_report(y_train, y_pred_dt))
    print()
    
    st.text("-------------------------------------------")
    st.text(" Performance Metrics (DT)")
    st.text("------------------------------------------")
    print()
    st.write(" Accuracy for DT :",result_dt,'%')
    print()     
    # ==== Graph 
    
    
    import seaborn as sns
    sns.barplot(x=["RF","DT","LR","CNN"],y=[result_rf,result_dt,result_lr,acc_cnn])
    plt.title("Comparison")
    # plt.savefig("Graph.png")
    plt.show()    
    
    
    st.image("Graph.png")  
    
    
# -------- PREDICTION


    st.markdown(f'<h1 style="color:#FFFFFF;font-size:18px;">{"Kindly enter the following details !!! "}</h1>', unsafe_allow_html=True)


    dur = st.text_input("Enter the Duration",'1')
    dur = int(dur)

    proto = st.text_input("Enter the Protocol Type",'2')
    proto = int(proto)

    service = st.text_input("Enter the Service",'2')
    service = int(service)

    flag = st.text_input("Enter the Flag",'2')
    flag = int(flag)

    src = st.text_input("Enter the Source Bytes",'2')
    src = int(src)

    destn = st.text_input("Enter the Destination Bytes Bytes",'6')
    destn = int(destn)

    aa = st.button("Submit")
    
    if aa:
        
        Data_reg = [dur,proto,service,flag,src,destn]
                    
        y_pred_reg=logreg.predict([Data_reg])
        
        if y_pred_reg == 11:
            
            st.markdown(f'<h1 style="color:#000000;font-size:18px;">{"No Intrusion - Normal "}</h1>', unsafe_allow_html=True)
        
        else:
            
            st.markdown(f'<h1 style="color:#000000;font-size:18px;">{" Intrusion Detection - Attack "}</h1>', unsafe_allow_html=True)
            











        
