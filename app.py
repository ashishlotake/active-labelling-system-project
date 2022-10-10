import glob
from array import array
from signal import signal
import PIL
import zipfile
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import shutil
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# from streamlit.script_runner import StopException, RerunException
# import pyautogui
from symbol import shift_expr

import tensorflow as tf
from tensorflow import keras
import joblib
from tensorflow.keras.preprocessing.image import load_img, save_img
#######################################
#######################################

import base64


###########################################################
# load immutable stuffs
###########################################################
# this will make sure the daat is not loaded every time run app


@st.cache(allow_output_mutation=True)
def load():
    best_model = keras.models.load_model("resnet50_no_augmentation.keras")
    idx_class = joblib.load("class_idx")
    class_name = tuple(idx_class.values())

    classifier_model_v2 = joblib.load("GRAD-CAM.pkl")
    last_conv_layer_model = joblib.load("last_conv_layer_model.pkl")

    return last_conv_layer_model, classifier_model_v2, class_name, best_model, idx_class


last_conv_layer_model, classifier_model_v2, class_name, best_model, idx_class = load()

last_conv_layer_name = "conv5_block3_out"
classifier_layer_names = ["global_average_pooling2d", "batch_normalization",
                          "dropout", "dense", "batch_normalization_1", "dropout_1",
                          "dense_1"
                          ]

###########################################################
# Set up title .
###########################################################

# Title.
st.markdown("<div style='text-align: center; '><i><h1 >Active Labelling System</h1>Author :- Ashish Lotake</i></div",
            unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color:gray ; font-size:15px'><i>Please ensure that all images have unique name.</i></p>",
            unsafe_allow_html=True)

# upload images
uploaded_file = st.file_uploader('',
                                 type=['png', 'jpg', 'jpeg'],
                                 accept_multiple_files=True)

######################################
# sidebar
######################################

with st.sidebar:
    # st.markdown("<h1 style='text-align: center; '> Labelling System</h1>",
    #             unsafe_allow_html=True)
    # st.header("All the images model can label for now")
    st.selectbox("See all 101 images categories supported", class_name)
    # st.markdown("<h1 style='text-align:center; '> Author: Ashish Lotake</h5>",
    #             unsafe_allow_html=True)
    # st.write("Intruction on 'How to use ?'")
    st.markdown("<h3 style='text-align: center; '><i> How to use?</i></h3>",
                unsafe_allow_html=True)
    st.markdown("<p><i>1. Upload images, and ensure all have unique name <br><br>2. Use first slider to adjust images per row. <br><br>3. Use second slide to define threshold<br> <small>( threshold:- minumum accepctable model score for image, below which you need to label images)</small><br><br>4. YOU NEED TO MAKE SURE TO LABEL ALL THE IMAGES ON THE SCREEN, ONLY THEN PRESS SUBMIT<br><br>5. If happy with result dont press submit, directly press  proceed <br><br>6. After Pressing proceed just press Downlaod zip to downlaod labelled data .<br><br>7. If not happy, label all images on screen then press submit, thne downlaod to download labelled data and then train</i></p>",
                unsafe_allow_html=True)
    st.image('destination_path.png')
    


###########################################################
# function
###########################################################


# @st.cache(allow_output_mutation=True)
def get_img_array(img_path, target_size):

    # laod image and resize it
    img = keras.utils.load_img(img_path, target_size=target_size)
    # convert tensor to array
    array = keras.utils.img_to_array(img)
    # making sure channle last by expanding domension
    array = np.expand_dims(array, axis=0)
    # preprocess images as per requied model
    array = keras.applications.resnet50.preprocess_input(array)
    return array


# @st.cache(allow_output_mutation=True)
def decode_prediction(preds):
    # find the top three classes predicted for this image
    top_preds = []
    _pred = sorted([i for i in (preds[0])], reverse=True)[:3]
    for i in _pred:
        pred_idx = np.where(preds[0] == i)[0][0]
        pred_class = idx_class[pred_idx]
        top_preds.append((pred_class, i))

    return top_preds


###########################################################
# model interpretibility
###########################################################

# @st.cache(persist=True)
def grad_cam(last_conv_layer_model, img_a, classifier_model_v2):
    with tf.GradientTape() as tape:

        # compute the activation of last conv layer
        last_conv_layer_output = last_conv_layer_model(img_a)

        # then make the 'tape' watch it
        tape.watch(last_conv_layer_output)

        # retrive the activation channel corresponding to the top predicted class
        preds = classifier_model_v2(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

        # this is the gradient of the top predicted class with regard to the output feature map of the last convolutional layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # Now we apply pooling and importance weighting to the gradient tensor to obtain our heatmap of class activation.
        # This is a vector where each entry is the mean intensity of the gradient for a given channel.
        # It quantifies the importance of each channel with regard to the top predicted class
        pooled_grad = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_layer_output = last_conv_layer_output.numpy()[0]

        # multiply each channel in the output of the last convolutional layer by "ho importance this channel is"
        for i in range(pooled_grad.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grad[i]

    return last_conv_layer_output


def visualize_cnn_output(classifier_model_v2,
                         last_conv_layer_name,
                         classifier_layer_names,
                         img_a, preds, image_path):
    '''
    classifier_model_v2 --> Model with only classifier layers from the original model
    last_conv_layer_name --> last convolutional layer
    classifier_layer_names --> all the layeres after the last convolutional layer
    img_a --> Image converted to array form
    preds --> model prediction for img_a
    image_path --> location of image on the disk. This is for plotting heatmap on top of image
    '''

    last_conv_layer_output = grad_cam(
        last_conv_layer_model, img_a, classifier_model_v2)

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.6 + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    return superimposed_img



###########################################################
# helper function
###########################################################


# define widgest

def my_widget_selectbox(key_):
    return col.selectbox("", class_name, key=key_)


@st.cache(allow_output_mutation=True)
def helper_func(uploaded_file):
    full_image_info = []
    for imgsss in uploaded_file:
        img_a = get_img_array(imgsss, (180, 180))
        preds = best_model.predict(img_a)
        label = decode_prediction(preds)[0]
        full_image_info.append((imgsss, img_a, preds, label))
    full_image_info = sorted(
        full_image_info, key=lambda t: t[-1][::-1], reverse=True)
    return full_image_info


def display_grid(super_imposed_imgs, n_col):

    super_final = [super_imposed_imgs[i:i + n_col]
                   for i in range(0, len(super_imposed_imgs), n_col)]

    for images_info in super_final:

        col_2 = st.columns(n_col)

        for i, col in enumerate(col_2):

            try:
                img_a, preds, imges_, label = images_info[i]
                finl_img = visualize_cnn_output(
                    classifier_model_v2, last_conv_layer_name, classifier_layer_names, img_a, preds, imges_)
                # super_imposed_imgs.append(finl_img)
                col.image(finl_img)
                col.write(label)
            except:
                pass

###########################################################
# folder to save images which need to be trained
###########################################################


def create_folder():
    try:
        shutil.rmtree("images_to_retrain")
        os.mkdir("images_to_retrain")
    except:
        os.mkdir("images_to_retrain")


def create_folder_label():
    try:
        shutil.rmtree("labelled_data")
        os.mkdir("labelled_data")
    except:
        pass
        os.mkdir("labelled_data")


#########################################################
# to creat a zip file so user can downlaod 
#########################################################
def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, mode='w') as zipf:
        len_dir_path = len(folder_path)
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len_dir_path:])


###########################################################
# Set up main app.
###########################################################
if len(uploaded_file) == 0:
    st.error("Please upload images see the magic")
else:
    if len([up_img.name for up_img in uploaded_file]) != len(set([up_img.name for up_img in uploaded_file])):
        st.error("Please check if all the uploaded images have unique name")
    else:


        # Add in sliders.

        super_imposed_imgs = []
        user_interest_images = []
        st.markdown("<p style='text-align: center;font-size:18px '><i>How many images to display per row?</i></p>",
                    unsafe_allow_html=True)

        n_col = st.slider("", 1, 6, 4, 1)

        full_image_info = helper_func(uploaded_file)

        # confidence
        st.markdown("<p style='text-align: center;font-size:18px '><i>Confidence threshold<br>What is the minimum acceptable confidence level below which you want to label images ?</i></p>",
                    unsafe_allow_html=True)


        confidence_threshold = st.slider(
            "", 0.0, 1.0, 0.85, 0.005)

        # # result
        for (imgsss, img_a, preds, label) in full_image_info:
            if confidence_threshold > label[-1]:
                user_interest_images.append((imgsss, label))

                super_imposed_imgs.append((img_a, preds, imgsss, label))

        user_final = [user_interest_images[i:i + n_col]
                      for i in range(0, len(user_interest_images), n_col)]

            

        # st.markdown( "<p style='text-align: center; font-size:14px '>IMPORTANT:- LABEL ALL THE IMAGES ON THE SCREEN and IF YOU ARE HAPPY WITH MODEL CLASSIFICATION MOVE SLIDER SO NO IMAGES ARE ON SCREEN ONLY AFTER THAN PRESS PROCEED <br>If you cannot see any images on screen means model have classified them correctly with your defined threshold </p>", unsafe_allow_html=True)



        with st.form(key='User_answer', clear_on_submit=True):

            # st.markdown(
            #     f"<p style='text-align: center; font-size:13px '>Images for which model has confidence score below {confidence_threshold*100} %</p>", unsafe_allow_html=True)

            # st.markdown("<p style='text-align: center; color:gray ; font-size:13px '>Here,<span style='color:rgb(9, 171, 59); font-family:Courier New'>('helicopter', 94.5272)</span>; implies that model is <span style=' color:rgb(9, 171, 59); font-family:Courier New'>94.5272% confident</span> that's it's a <span style=' color:rgb(9, 171, 59); font-family:Courier New'>helicopter</span>.</p>",
            #             unsafe_allow_html=True)
            

            for images_info in user_final:

                col_2 = st.columns(n_col)

                for i, col in enumerate(col_2):

                    try:
                        imgess, label = images_info[i]
                        col.image(imgess)
                        col.write((label[0], str(label[-1]*100)[:7]))
                        # col.write(imgess.name)
                        my_widget_selectbox(key_=imgess.name)
                        col.widget_tags()
                    except:
                        pass
            


            submitted = st.form_submit_button('Submit')

        img_to_retrain = {}
        create_folder()
        create_folder_label()
        if submitted:
            # create_folder()
            # create_folder_label()
            user_reply = {k: v for k, v in st.session_state.items(
            ) if (k != "FormSubmitter:User_answer-Submit") and (k != "button")}
            st.write(user_reply)

            ## get the label from user ans now create a set then move imamges to respective folder
            label_name = set(user_reply.values())
            # st.write(label_name)
            # for i in label_name:
            #     if os.path.isdir('images_to_retrain/'+i):
            #         pass
            #     else: 
            #         os.mkdir('images_to_retrain/'+i)

            retrain_folder = pathlib.Path("images_to_retrain")
            
            for single_img in uploaded_file:
                if single_img.name in st.session_state.keys():
                    # st.image(single_img)
                    img_name = single_img.name
                    folder_name = st.session_state[single_img.name]
                    # st.write(folder_name)
                    save_img_name = folder_name+"*-*"+img_name

                    # here i will store files for retraining, in a directory.
                    for imggss in glob.glob("images_to_retrain/*.*"):
                        # st.write(imggss)
                        if imggss.split("*-*")[-1] == save_img_name.split('*-*')[-1]:
                            # here i am renaming, lets say user changes its mind and change the label of a images
                            # so take this into account, i will jsut rename image to new label

                            new_name = "images_to_retrain/" + save_img_name
                            # st.write(imggss, new_name)
                            os.rename(imggss, new_name)
                        else:
                            with open(os.path.join(retrain_folder, save_img_name), "wb") as f:
                                f.write(single_img.getbuffer())
                    else:
                        with open(os.path.join(retrain_folder, save_img_name), "wb") as f:
                            f.write(single_img.getbuffer())
    

   # ###########################################################
   # # model training pahse
   # ###########################################################
        st.markdown(
            f"<p style='text-align: center; font-size:13px '><i>LABEL ALL THE IMAGES ON THE SCREEN  <br> IF YOU ARE HAPPY WITH MODEL CLASSIFICATION DONT PRESS SUBMIT, PRESS PROCEED</p>", unsafe_allow_html=True)

        procees_btn = st.button("Proceed")
        if st.session_state.get('button') != True:
            st.session_state['button'] = procees_btn
        
        if st.session_state['button'] == True:

        

            # ge tall the images and their labels and name
            # we will use this  to move images so user can download them
            all_img_ls = {}
            for (imgsss, img_a, preds, label) in full_image_info:
                all_img_ls[imgsss.name] = label

            # st.write(all_img_ls)

            # now lets move the images to directory which users wants to train

            img_to_train_or_no_check_dict = {}
            for all_info in user_interest_images:            
                img_to_train_or_no_check_dict[all_info[0].name] = all_info[-1]

            # st.write(img_to_train_or_no_check_dict)


    ##############################################################################################
    # moving all the images for which user is happy to labelled folder
    ##############################################################################################

            # find all the images above user defined threshold and move them to labelled folder so to downlaod 
            img_to_move_2_label = dict(all_img_ls.items() - img_to_train_or_no_check_dict.items())
            # st.write(img_to_move_2_label)

            moving_folder = pathlib.Path("labelled_data")
            for images in uploaded_file:
                # st.image(images)
                try:
                    f_name,_ = img_to_move_2_label[images.name]
                    if os.path.isdir('labelled_data/'+f_name):
                        with open(os.path.join(moving_folder, f_name+"/"+images.name), "wb") as f:
                            f.write(images.getbuffer())
                    else: 
                        os.mkdir('labelled_data/'+f_name)
                        with open(os.path.join(moving_folder,  f_name+"/"+images.name), "wb") as f:
                            f.write(images.getbuffer())      
                except:
                    pass

    ##############################################################################################

    ##############################################################################################
    # moving all the images for training and to labelled folder, so user and download
    # these are alos the images, that user self labelled
    ##############################################################################################

            ## here we will go through all the images, and if my previous model have high propability then i will not train that image
            all_imgs = glob.glob("images_to_retrain/*.*")


            for img in all_imgs:

                # img_n = img.split('/')[-1].split('*-*')[-1]
                # st.write(all_img_ls[img_n])

                folder_n = img.split('/')[-1].split('*-*')[0]
                # st.write(folder_n)
        
                if os.path.isdir('images_to_retrain/'+folder_n):
                    pass
                else: 
                    os.mkdir('images_to_retrain/'+folder_n)

                if os.path.isdir('labelled_data/'+folder_n):
                    pass
                else: 
                    os.mkdir('labelled_data/'+folder_n)
                try:
                    shutil.copy(img, 'original_data'+'/'+folder_n)
                    shutil.copy(img, 'labelled_data'+'/'+folder_n)
                    shutil.move(img, 'images_to_retrain'+'/'+folder_n)
                except:
                    os.remove(img)

    ##############################################################################################


    ############################################################
    ## Creating zip and let user download 
    ############################################################

            zip_directory("labelled_data", "labelled_data.zip")

            with open('labelled_data.zip', 'rb') as f:
                # Defaults to 'application/octet-stream'
                st.download_button('Download Zip', f, file_name='labelled_data.zip')

            os.remove("labelled_data.zip")


    ############################################################
    ## training the model
    ############################################################

            if st.button('Training'):
                st.session_state['button'] = False

                if len(tf.config.list_physical_devices('GPU')) ==0:
                    # st.error("Cant train No GPU avaliable")
                    st.warning("Training")

                else:
                    st.warning("Training")
                    resenet50_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= keras.applications.resnet50.preprocess_input)

                    train_set = resenet50_gen.flow_from_directory("original_data", target_size=(180, 180), batch_size=32)

                    callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

                    previous_model = keras.models.load_model("resnet50_no_augmentation.keras")

                    previous_model.fit(train_set, epochs=10,callbacks=callbacks)
                    previous_model.save("resnet50_no_augmentation.keras")







            # st.warning("Trainig")
            
            # # loading and preprocessing images
            # resenet50_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= keras.applications.resnet50.preprocess_input)
            # data_2_train = resenet50_gen.flow_from_directory("images_to_retrain")
            # best_model.fit_generator(resenet50_gen)


            # for img_re in glob.glob("images_to_retrain/*"):
            #     st.image(img_re)
        


    ############################################################
    ## model interpretability
    ############################################################

        st.markdown("<h2 style='text-align: center; '><i>Model Interpretibility</i></h4>",
                    unsafe_allow_html=True)

        st.markdown(
            f"<p style='text-align: center; '><i>Images Highlighted with models' region on interest</i></p>", unsafe_allow_html=True)

        # st.write("Here the images have been higlightes")

        if st.button("View CNN region of interest"):
            # st.write('d')
            with st.expander("Collapse/Expand", expanded=True):
                display_grid(super_imposed_imgs, n_col)
