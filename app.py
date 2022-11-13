# To run use
# $ streamlit run yolor_streamlit_demo.py
#from yolor import *
from yolo_v7 import *
import tempfile
import cv2
import torch

from utils.hubconf import custom
from utils.plots import plot_one_box

from models.models import *
from utils.datasets import *
from utils.general import *

import streamlit as st
from PIL import ImageColor

def main():
    #타이틀
    st.title("Object Tracking Dashboard YOLOV7 - CapstoneDesign -")
    #사이드바 타이틀
    st.sidebar.title('설정')

    # #구분선
    st.sidebar.markdown('---')

    # path to model
    path_model_file = st.sidebar.text_input(
    'Yolov7 모델경로',
    'C:\coding\yolov7streamlit-main\yolov7streamlit-main\yolov7.pt'
    )

     #구분선
    st.sidebar.markdown('---')

    # Class txt
    path_to_class_txt = st.sidebar.file_uploader(
    '클래스 파일:', type=['txt']
    )
    # #구분선
    st.sidebar.markdown('---')

    if path_to_class_txt is not None:

        options = st.sidebar.radio(
            '선택:', ('Webcam', 'Image', 'Video', 'RTSP'), index=1)

        gpu_option = st.sidebar.radio(
            'PU 선택:', ('CPU', 'GPU'))
        if not torch.cuda.is_available():
            st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
        else:
            st.sidebar.success(
                'GPU is Available on this Device, Choose GPU for the best performance',
                icon="✅"
            )
        st.sidebar.markdown('---')

        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )

        st.sidebar.markdown('---')
























    # st.markdown(
    # """
    # <style>
    # [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    #     width: 350px;
    # }
    # [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    #     width: 350px;
    #     margin-left: -350px;
    # }
    # </style>
    # """,
    # unsafe_allow_html=True,
    # )






















































    # #구분선
    # st.sidebar.markdown('---')
    # #사이드바 confidence 조절 슬라이드바
    # confidence = st.sidebar.slider('Confidence',min_value = 0.0, max_value = 1.0, value=0.25)
    # #구분선
    # st.sidebar.markdown('---')

    # #체크박스
    # save_img = st.sidebar.checkbox('동영상 저장')
    # enable_GPU = st.sidebar.checkbox('GPU 사용')
    # use_webcam = st.sidebar.checkbox('웹캠 사용')
    # custom_classes = st.sidebar.checkbox('Custom classes 사용')
    

    # assigned_class_id = []

    # #custom classes 선택함수 / custom classes 체크 선택하면 나옴
    # if custom_classes:
    #         assigned_class = st.sidebar.multiselect('Custom classes를 선택하시오', list(names), default='person')
    #         for each in assigned_class:
    #             assigned_class_id.append(names.index(each))

    # #구분선
    # st.sidebar.markdown('---')
    # #비디오 업로드
    # video_file_buffer = st.sidebar.file_uploader("동영상를 업로드 하세요", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    # #디폴트 비디오
    # DEMO_VIDEO = 'test.mp4'
    # tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    # ##웹캠, 비디오 업로드
    # if not video_file_buffer:
    #     if use_webcam:
    #         vid = cv2.VideoCapture(0, cv2.CAP_ARAVIS)
    #         tfflie.name = 0
    #     else:
    #         vid = cv2.VideoCapture(DEMO_VIDEO)
    #         tfflie.name = DEMO_VIDEO
    #         dem_vid = open(tfflie.name,'rb')
    #         demo_bytes = dem_vid.read()
    
    #         st.sidebar.text('업로드 된 동영상')
    #         st.sidebar.video(demo_bytes)

    # else:
    #     tfflie.write(video_file_buffer.read())
    #     # print("No Buffer")
    #     dem_vid = open(tfflie.name,'rb')
    #     demo_bytes = dem_vid.read()
    
    #     st.sidebar.text('업로드 된 동영상')
    #     st.sidebar.video(demo_bytes)

    # print(tfflie.name)

    # stframe = st.empty()
    # #구분선
    # st.sidebar.markdown('---')




    # #구분선
    # st.markdown("<hr/>", unsafe_allow_html=True)
    # kpi1, kpi2, kpi3 = st.columns(3)


    # with kpi1:
    #     st.markdown("**Frame Rate**")
    #     kpi1_text = st.markdown("0")

    # with kpi2:
    #     st.markdown("**Tracked Objects**")
    #     kpi2_text = st.markdown("0")

    # with kpi3:
    #     st.markdown("**Total Count**")
    #     kpi3_text = st.markdown("0")

    # #구분선
    # st.markdown("<hr/>", unsafe_allow_html=True)

    # load_yolov7_and_process_each_frame('yolov7', tfflie.name, enable_GPU, save_img, confidence, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stframe)
    # #load_yolor_and_process_each_frame( tfflie.name, enable_GPU, confidence, assigned_class_id, kpi1_text, kpi2_text, kpi3_text, stframe)

    # st.text('Video is Processed')
    # vid.release()






if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass