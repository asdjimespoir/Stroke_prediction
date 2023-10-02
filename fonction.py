import cv2
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from pathlib import Path
from scipy import ndimage
import plotly.express as px
import imageio.v2 as imageio
from PIL import Image,ImageOps
from skimage import morphology
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit_authenticator as stauth
from tensorflow.keras.optimizers import SGD
from streamlit_option_menu import option_menu
from sklearn.compose import ColumnTransformer #data preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder #data preprocessing


def predict_patient(scan, model):
    scan = np.array(scan)
    prediction = model.predict(np.expand_dims(scan, axis=0))[0]
    pred = prediction[0]
    return pred

def predict_tabulaire_data(model,feature):
    with open("./Models/Steptab.pkl", "rb") as file:
        ct = pickle.load(file)

    cat_ct = ct["cat_ct"]
    num_ct = ct["num_ct"]
    

    feature = np.array(feature)
    feature = feature.astype(object)
    feat_encoded = cat_ct.transform(feature)
    feat_scaled = num_ct.transform(feat_encoded)
    
    #feature = feat_scaled[:,[0,1,2,3,5,8,9,13,16,18,19,20]]
    feature = feat_scaled.astype(float)

    prediction = model.predict(feature)
    prediction = prediction[0][0]
    return prediction

def imagePred(image,model):
    size = (128, 128)
    image = cv2.resize(image, size , interpolation=cv2.INTER_AREA)
    image = np.array(image).flatten()
    image = image/255
    
    loss='binary_crossentropy'
    metrics = ['accuracy']
    optimizer = tf.keras.optimizers.legacy.SGD()
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    
    pred = model.predict(np.expand_dims(image, axis=0))
    pred = pred[0][0]
    return pred
 
def histogramme(df):
    #st.sidebar.write("Menu de la Page 1")
    # Histogramme
    #st.markdown(" ## Histogram")
    n_bins = st.number_input(
        label = "Nombre de bins",
        min_value = 20,
        max_value = 100,
        value = 70
    )
    #histo_title = st.text_input("Histogramme")
    fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
    ax1[0].hist(df['age'], bins=n_bins )
    ax1[0].set_title("Age")
    ax1[1].hist(df['bmi'], bins=n_bins, color="r" )
    ax1[1].set_title("BMI")
    #ax1[1, 0].hist(df['stroke'], bins=n_bins, color="g" )
    #ax1[1, 0].set_title("Stroke")
    ax1[2].hist(df['avg_glucose_level'], bins=n_bins, color="orange" )
    ax1[2].set_title("Avg Glucose Level")
    #plt.tight_layout()
    st.pyplot(fig1)

def histogramme_to_choice(df,num_col,cat_col):
    # Histogramme to choice
    #st.markdown("## Plotly Chart")
    #num_col = df.select_dtypes(exclude=['object']).columns.to_list()
    #cat_col = df.select_dtypes(include=['object']).columns.to_list()
    var_x = st.selectbox("Variable en Abscisse", num_col)
    var_y = st.selectbox("Variable en Ordonnée", num_col)
    var_color = st.selectbox("Variables pour colorier les points", cat_col)

    fig2 = px.scatter(df,
                     x= var_x,
                     y= var_y,
                     color = var_color,
                     title = str(var_x)+" vs "+str(var_y)+" by " +str(var_color))
    st.plotly_chart(fig2)

def countplot(df,num_col,cat_col):
    # Display the countplot
    #st.markdown("## Count Histogram")
    sns.set_style("darkgrid")
    fig3, ax3 = plt.subplots(figsize=(10,5))
    strok = df.loc[df['stroke']==1]
    list_col = st.selectbox("Selectionner un élément", ["gender","ever_married","hypertension","work_type","smoking_status","Residence_type","heart_disease"])
    sns.countplot(strok, x = list_col,palette='inferno')
    st.pyplot(fig3)
    
def matrice_corr(df,num_col,cat_col):
    # Matrice corr
    #st.markdown("## Matrice de correlation")
    fig4, ax4 = plt.subplots(figsize=(15,7))
    #df_encoded = df[num_col]#pd.get_dummies(df)
    sns.heatmap(df[num_col].corr(),annot=True)
    st.pyplot(fig4)

def resize_image(image):
    image = ImageOps.exif_transpose(image)  # Corrects the image orientation based on EXIF data.
    image = image.resize((800,600), Image.LANCZOS)  # Specify the resampling filter (e.g., LANCZOS).  #ANTIALIAS NEAREST BILINEAR BICUBIC BOX HAMMING LANCZOS .Resampling.
    return image

def plot_scan_from_path(slices_path, patient_id):
    """Plot 40 slices for a patient ID"""
    num_rows = 4
    num_columns = 10
    factor = 1.2
    f, axarr = plt.subplots(
        num_rows,
        num_columns,
        figsize=(num_columns*factor, num_rows*factor),
    )
    f.suptitle(f"Patient {patient_id}", y=1.1)
    image_id = 1
    for i in range(num_rows):
        for j in range(num_columns):
          try:
            img = imageio.imread(f'{slices_path}/{patient_id} ({image_id}).jpg')
          except Exception as e:
            print(e)
            img = np.zeros((2,2))
          finally:
            axarr[i, j].imshow(img, cmap='gray')
            axarr[i, j].axis('off')
            image_id += 1
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

def remove_noise(image, display=False):
  """Remove slice noise"""
  # morphology.dilation crée une segmentation de l'image
  # Si un pixel se trouve entre l'origine et le bord d'un carré de taille
  # 3x3, le pixel appartient à la même classe
  segmentation = morphology.dilation(image, np.ones((3, 3)))
  segmentation[segmentation < 25] = 0
  segmentation[segmentation > 25] = 1
  labels, label_nb = ndimage.label(segmentation)
  label_count = np.bincount(labels.ravel().astype(int))

  # La taille de label_count est le nombre de classes/segmentations trouvées.
  # La première classe n'est pas utilisée puisqu'il s'agit de l'arrière-plan.
  label_count[0] = 0

  # Un masque avec la classe ayant le plus de pixels est créé
  # puisqu'il doit représenter le cerveau
  mask = labels == label_count.argmax()

  # Améliorer le masque cérébral
  mask = morphology.dilation(mask, np.ones((5, 5)))
  mask = ndimage.binary_fill_holes(mask)
  mask = morphology.dilation(mask, np.ones((3, 3)))

  # Puisque les pixels du masque sont des zéros et des uns,
  # il est possible de multiplier l'image originale pour ne conserver que la région du cerveau
  masked_image = mask * image

  if display:
    plt.figure(figsize=(10, 2.5))
    plt.subplot(141)
    plt.imshow(image, cmap=plt.cm.bone)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(mask, cmap=plt.cm.bone)
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(masked_image, cmap=plt.cm.bone)
    plt.title('Clean Image')
    plt.axis('off')

  return masked_image

def resize_scan(scan):
  """Resize the CT scan to a desired uniform size across all axis"""
  # Set the desired depth
  desired_depth = 64
  desired_width = 128
  desired_height = 128
  # Get current depth
  current_depth = scan.shape[-1]
  current_width = scan.shape[0]
  current_height = scan.shape[1]
  # Compute depth factor
  depth = current_depth / desired_depth
  width = current_width / desired_width
  height = current_height / desired_height
  depth_factor = 1 / depth
  width_factor = 1 / width
  height_factor = 1 / height
  # Rotate
  scan = ndimage.rotate(scan, 90, reshape=False)
  # Resize across z-axis
  scan = ndimage.zoom(scan, (width_factor, height_factor, depth_factor), order=1)
  return scan

def normalize_scan(scan):
  """Normalize the scan to the interval [0, 1]"""
  min = 0
  max = 255
  scan[scan < min] = min
  scan[scan > max] = max
  scan = (scan - min) / (max - min)
  scan = scan.astype("float32")
  return scan

def plot_scan_from_dataset(num_rows, num_columns, width, height, data, title):
    """Plot a scan from dataset"""
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    f.suptitle(title, y=1.1)
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    #plt.show()
    st.pyplot(f)
