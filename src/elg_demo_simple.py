#!/usr/bin/env python3
"""Main script for gaze direction inference from webcam feed."""
import argparse
import os
import queue
import threading
import time

import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf

from datasources import Video, Webcam
from models import ELG
import util.gaze

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')

    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 2

        # Define webcam stream data source
        # Change data_format='NHWC' if not using CUDA
        if args.from_video:
            assert os.path.isfile(args.from_video)
            data_source = Video(args.from_video,
                                tensorflow_session=session, batch_size=batch_size,
                                data_format='NCHW' if gpu_available else 'NHWC',
                                eye_image_shape=(108, 180))
        else:
            data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))

        # Define model
        if args.from_video:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
        else:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=1,
                num_modules=2,
                num_feature_maps=32,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )

        # Record output frames to file if requested
        if args.record_video:
            video_out = None
            video_out_queue = queue.Queue()
            video_out_should_stop = False
            video_out_done = threading.Condition()

            def _record_frame():
                global video_out
                last_frame_time = None
                out_fps = 30
                out_frame_interval = 1.0 / out_fps
                while not video_out_should_stop:
                    frame_index = video_out_queue.get()
                    if frame_index is None:
                        break
                    assert frame_index in data_source._frames
                    frame = data_source._frames[frame_index]['bgr']
                    h, w, _ = frame.shape
                    if video_out is None:
                        video_out = cv.VideoWriter(
                            args.record_video, cv.VideoWriter_fourcc(*'H264'),
                            out_fps, (w, h),
                        )
                    now_time = time.time()
                    if last_frame_time is not None:
                        time_diff = now_time - last_frame_time
                        while time_diff > 0.0:
                            video_out.write(frame)
                            time_diff -= out_frame_interval
                    last_frame_time = now_time
                video_out.release()
                with video_out_done:
                    video_out_done.notify_all()
            record_thread = threading.Thread(target=_record_frame, name='record')
            record_thread.daemon = True
            record_thread.start()

        # Begin visualization thread
        infer = model.inference_generator()

        def _visualize_output():

            if args.fullscreen:
                cv.namedWindow('vis', cv.WND_PROP_FULLSCREEN)
                cv.setWindowProperty('vis', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

            while True:
                
                output = next(infer)
                frame_index = output['frame_index'][0]
                if frame_index not in data_source._frames:
                    continue
                frame = data_source._frames[frame_index]
                bgr = frame['bgr']
                faces = [None] * len(frame['faces'])
                for face_index, face in enumerate(frame['faces']):
                    faces[face_index] = {
                        'face_index': face_index,
                        'face_coordinate': face,
                        'emotion': {},
                        'eyegaze': {
                            'right': {},
                            'left': {},
                        }
                    }
                    cv.rectangle(
                        bgr, tuple(np.round(face[:2]).astype(np.int32)),
                        tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                        color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                    )

                for eye_index in output['eye_index']:
                    # Decide which landmarks are usable
                    heatmaps_amax = np.amax(
                        output['heatmaps'][eye_index, :].reshape(-1, 18), axis=0)
                    can_use_eye = np.all(heatmaps_amax > 0.7)
                    if not can_use_eye:
                        continue
                    
                    eye = frame['eyes'][eye_index]
                    eye_image = eye['image']
                    eye_side = eye['side']
                    eye_landmarks = output['landmarks'][eye_index, :]
                    eye_radius = output['radius'][eye_index][0]
                    if eye_side == 'left':
                        eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]

                    # Transform predictions
                    eye_landmarks = np.concatenate([eye_landmarks,
                                                    [[eye_landmarks[-1, 0] + eye_radius,
                                                      eye_landmarks[-1, 1]]]])
                    eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                                       'constant', constant_values=1.0))
                    eye_landmarks = (eye_landmarks *
                                     eye['inv_landmarks_transform_mat'].T)[:, :2]
                    eye_landmarks = np.asarray(eye_landmarks)
                    # eyelid_landmarks = eye_landmarks[0:8, :]
                    # iris_landmarks = eye_landmarks[8:16, :]
                    iris_centre = eye_landmarks[16, :]
                    eyeball_centre = eye_landmarks[17, :]
                    eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                                    eye_landmarks[17, :])

                    i_x0, i_y0 = iris_centre
                    e_x0, e_y0 = eyeball_centre
                    theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                    phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                            -1.0, 1.0))
                    current_gaze = np.array([theta, phi])
                    # util.gaze.draw_gaze(bgr, iris_centre, current_gaze, length=240.0, thickness=3)
                    
                    face_index = int(eye_index / 2)
                    faces[face_index]['eyegaze'][eye_side] = {
                        'vector_x_component': -np.sin(current_gaze[1]),
                        'vector_y_component': -np.sin(current_gaze[0]),
                        'iris_centre_x': np.round(i_x0).astype(np.int32),
                        'iris_centre_y': np.round(i_y0).astype(np.int32),
                    }

                for face in faces:
                    # 假设已经检测到人脸F(其宽和高分别为w和h)，
                    # 其左眼和右眼的中心坐标分别为C1 = (x1, y1)和C2 = (x2, y2)(根据landmark推算得到)，
                    # gaze算法输出的左右眼视线方向分别为 D1 = (v_x1, v_y1)和D2 = (v_x2, v_y2)(忽略z方向的值)，
                    # 则确定视线汇聚点的方法如下:
                    # Ø  确定圆心: O = (C1 + C2) / 2
                    # Ø  确定半径:  R = max(w, h)
                    # Ø  计算平均视线: D = (D1 + D2) / 2
                    # Ø  计算视线汇聚点坐标:  P = O + R*D

                    left_eyegaze = face['eyegaze']['left']
                    right_eyegaze = face['eyegaze']['right']
                    if (not left_eyegaze) and (not right_eyegaze):
                        continue
                    if not left_eyegaze:
                        left_eyegaze = right_eyegaze
                    if not right_eyegaze:
                        right_eyegaze = left_eyegaze
                    x1 = left_eyegaze['iris_centre_x']
                    y1 = left_eyegaze['iris_centre_y']
                    x2 = right_eyegaze['iris_centre_x']
                    y2 = right_eyegaze['iris_centre_y']
                    O = ((x1 + x2)/2, (y1 + y2)/2)
                    face_width, face_height = tuple(np.round(face['face_coordinate'][2:]).astype(np.int32))
                    frame_height, frame_width, _ = bgr.shape
                    R = max(frame_height * frame_height/face_height,
                            frame_width * frame_width/face_width)/2
                    v_x1 = left_eyegaze['vector_x_component']
                    v_y1 = left_eyegaze['vector_y_component']
                    v_x2 = right_eyegaze['vector_x_component']
                    v_y2 = right_eyegaze['vector_y_component']
                    D = ((v_x1 + v_x2)/2, (v_y1 + v_y2)/2)
                    P = (int((O[0] + R * D[0])), int((O[1] + R * D[1])))

                    cv.circle(
                        bgr, P,
                        radius=5, color=(0, 0, 255), thickness=5)

                    cv.arrowedLine(
                        bgr,
                        (x1, y1),
                        P, (55, 255, 155), thickness=3
                    )

                    cv.arrowedLine(
                        bgr,
                        (x2, y2),
                        P, (55, 255, 155), thickness=3
                    )

                # Record frame?
                if args.record_video:
                    video_out_queue.put_nowait(frame_index)
                
                cv.imshow('vis', bgr)
                # Quit?
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return

                
        # visualize_thread = threading.Thread(target=_visualize_output, name='visualization')
        # visualize_thread.daemon = True
        # visualize_thread.start()

        _visualize_output()

        # Close video recording
        if args.record_video and video_out is not None:
            video_out_should_stop = True
            video_out_queue.put_nowait(None)
            with video_out_done:
                video_out_done.wait()
