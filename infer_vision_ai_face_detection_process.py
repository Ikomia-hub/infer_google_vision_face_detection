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
class InferVisionAiFaceDetectionParam(core.CWorkflowTaskParam):

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
class InferVisionAiFaceDetection(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Add input/output of the algorithm here
        # Example :  self.add_input(dataprocess.CImageIO())
        #           self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferVisionAiFaceDetectionParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None
        self.names = ["face"]
        self.set_names(self.names)

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

        # Process output
        for i, face in enumerate(faces):
            # Get coordinates
            vertices = [(vertex.x,vertex.y) for vertex in face.bounding_poly.vertices]
            x1 = vertices[0][0]
            y1 = vertices[0][1]
            w = vertices[1][0] - x1
            h = vertices[2][1] - y1
            # Display
            self.add_object(i+1, 0, face.detection_confidence, float(x1), float(y1), w, h)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferVisionAiFaceDetectionFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_vision_ai_face_detection"
        self.info.short_description = "your short description"
        self.info.description = "your description"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create algorithm object
        return InferVisionAiFaceDetection(self.info.name, param)
