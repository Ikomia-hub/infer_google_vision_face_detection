import copy
from ikomia import core, dataprocess
from google.cloud import vision
import os
import io
import cv2


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGoogleVisionFaceDetectionParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.window_size = 25
        self.conf_thres = 0.3
        self.google_application_credentials = ''

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.window_size = int(params["window_size"])
        self.conf_thres = float(params["conf_thres"])
        self.google_application_credentials = str(params["google_application_credentials"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["conf_thres"] = str(self.conf_thres)
        params["google_application_credentials"] = str(self.google_application_credentials)
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGoogleVisionFaceDetection(dataprocess.CKeypointDetectionTask):

    def __init__(self, name, param):
        dataprocess.CKeypointDetectionTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferGoogleVisionFaceDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None
        self.classes = ["face"]
        self.landmark_dict = {
            "LEFT_EYE": 0, "RIGHT_EYE": 1, "LEFT_OF_LEFT_EYEBROW": 2, 
            "RIGHT_OF_LEFT_EYEBROW": 3, "LEFT_OF_RIGHT_EYEBROW": 4, 
            "RIGHT_OF_RIGHT_EYEBROW": 5, "MIDPOINT_BETWEEN_EYES": 6, 
            "NOSE_TIP": 7, "UPPER_LIP": 8, "LOWER_LIP": 9, 
            "MOUTH_LEFT": 10, "MOUTH_RIGHT": 11, "MOUTH_CENTER": 12,
            "NOSE_BOTTOM_RIGHT": 13, "NOSE_BOTTOM_LEFT": 14, 
            "NOSE_BOTTOM_CENTER": 15, "LEFT_EYE_TOP_BOUNDARY": 16,
            "LEFT_EYE_RIGHT_CORNER": 17, "LEFT_EYE_BOTTOM_BOUNDARY": 18,
            "LEFT_EYE_LEFT_CORNER": 19, "RIGHT_EYE_TOP_BOUNDARY": 20,
            "RIGHT_EYE_RIGHT_CORNER": 21, "RIGHT_EYE_BOTTOM_BOUNDARY": 22,
            "RIGHT_EYE_LEFT_CORNER": 23, "LEFT_EYEBROW_UPPER_MIDPOINT": 24,
            "RIGHT_EYEBROW_UPPER_MIDPOINT": 25, "LEFT_EAR_TRAGION": 26,
            "RIGHT_EAR_TRAGION": 27, "FOREHEAD_GLABELLA": 28, 
            "CHIN_GNATHION": 29, "CHIN_LEFT_GONION": 30, 
            "CHIN_RIGHT_GONION": 31, "LEFT_CHEEK_CENTER": 32, 
            "RIGHT_CHEEK_CENTER": 33
        }
        self.skeleton = [
            # Eyes
            [self.landmark_dict["LEFT_EYE"], self.landmark_dict["RIGHT_EYE"]],
            # Eyebrows
            [self.landmark_dict["LEFT_OF_LEFT_EYEBROW"], self.landmark_dict["RIGHT_OF_LEFT_EYEBROW"]],
            [self.landmark_dict["LEFT_OF_RIGHT_EYEBROW"], self.landmark_dict["RIGHT_OF_RIGHT_EYEBROW"]],
            # Nose
            [self.landmark_dict["NOSE_TIP"], self.landmark_dict["MIDPOINT_BETWEEN_EYES"]],
            [self.landmark_dict["NOSE_BOTTOM_LEFT"], self.landmark_dict["NOSE_BOTTOM_CENTER"]],
            [self.landmark_dict["NOSE_BOTTOM_RIGHT"], self.landmark_dict["NOSE_BOTTOM_CENTER"]],
            [self.landmark_dict["NOSE_TIP"], self.landmark_dict["NOSE_BOTTOM_CENTER"]],
            # Mouth
            [self.landmark_dict["UPPER_LIP"], self.landmark_dict["MOUTH_LEFT"]],
            [self.landmark_dict["LOWER_LIP"], self.landmark_dict["MOUTH_RIGHT"]],
            [self.landmark_dict["MOUTH_RIGHT"], self.landmark_dict["UPPER_LIP"]],
            [self.landmark_dict["MOUTH_LEFT"], self.landmark_dict["LOWER_LIP"]],
            # Face contour
            [self.landmark_dict["LEFT_EAR_TRAGION"], self.landmark_dict["LEFT_OF_LEFT_EYEBROW"]],
            [self.landmark_dict["RIGHT_EAR_TRAGION"], self.landmark_dict["RIGHT_OF_RIGHT_EYEBROW"]],
            [self.landmark_dict["LEFT_EAR_TRAGION"], self.landmark_dict["CHIN_LEFT_GONION"]],
            [self.landmark_dict["RIGHT_EAR_TRAGION"], self.landmark_dict["CHIN_RIGHT_GONION"]],
            [self.landmark_dict["CHIN_LEFT_GONION"], self.landmark_dict["CHIN_GNATHION"]],
            [self.landmark_dict["CHIN_RIGHT_GONION"], self.landmark_dict["CHIN_GNATHION"]]
        ]
        self.palette = [
            # Eyes
            [255, 102, 102],
            # Eyebrows
            [255, 178, 102], [255, 178, 102],
            # Nose
            [153, 255, 153], [230, 230, 0], [230, 230, 0], [230, 230, 0],
            # Mouth
            [153, 204, 255], [153, 204, 255], [153, 204, 255], [153, 204, 255], 
            # Face contour
            [255, 255, 255],[255, 255, 255],[255, 255, 255],[255, 255, 255], [255, 255, 255], [255, 255, 255]
        ]


    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()

        # Get input :
        input = self.get_input(0)
        src_image = input.get_image()

        # Get parameters :
        param = self.get_param_object()

        # Set Keypoints links
        keypoint_links = []
        for (start_pt_idx, end_pt_idx), color in zip(self.skeleton, self.palette):
            link = dataprocess.CKeypointLink()
            link.start_point_index = start_pt_idx
            link.end_point_index = end_pt_idx
            link.color = color
            keypoint_links.append(link)
        self.set_keypoint_links(keypoint_links)
        self.set_object_names(self.classes)

        if self.client is None:
            if param.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = param.google_application_credentials
            self.client = vision.ImageAnnotatorClient()

        # Convert the NumPy array to a byte stream
        is_success, image_buffer = cv2.imencode(".jpg", src_image)
        byte_stream = io.BytesIO(image_buffer)

        # Convert the byte stream to bytes
        image_bytes = byte_stream.getvalue()

        # Create an Image object for Google Vision API
        image = vision.Image(content=image_bytes)

        # Inference
        response = self.client.face_detection(image=image)
        faces = response.face_annotations
        # # Process output
        for i, face in enumerate(faces):
            if face.detection_confidence < param.conf_thres: # skip detections with lower score
                continue
            # Get box coordinates
            vertices = [(vertex.x,vertex.y) for vertex in face.bounding_poly.vertices]
            x_box = vertices[0][0]
            y_box = vertices[0][1]
            w = vertices[1][0] - x_box
            h = vertices[2][1] - y_box

            # Get points coordinates
            landmarks_objects = face.landmarks
            kpts_data  = [(landmark.position.x, landmark.position.y) for landmark in landmarks_objects]

            # Set Keypoints links
            keypts = []
            kept_kp_id = []
            for link in self.get_keypoint_links():
                kp1, kp2 = kpts_data[link.start_point_index], kpts_data[link.end_point_index]
                x1, y1 = kp1
                x2, y2 = kp2
                if link.start_point_index not in kept_kp_id:
                    kept_kp_id.append(link.start_point_index)
                    keypts.append(
                        (link.start_point_index, dataprocess.CPointF(float(x1), float(y1))))
                if link.end_point_index not in kept_kp_id:
                    kept_kp_id.append(link.end_point_index)
                    keypts.append(
                        (link.end_point_index, dataprocess.CPointF(float(x2), float(y2))))

            # # Display
            self.add_object(i, 0, face.detection_confidence, float(x_box), float(y_box), w, h, keypts)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGoogleVisionFaceDetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_google_vision_face_detection"
        self.info.short_description = "Face detection using Google cloud vision API."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Google"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://cloud.google.com/vision/docs/detecting-faces"
        # Code source repository
        self.info.repository = "https://github.com/googleapis/python-vision"
        # Keywords used for search
        self.info.keywords = "Face detection,Google,Cloud,Vision AI"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferGoogleVisionFaceDetection(self.info.name, param)
